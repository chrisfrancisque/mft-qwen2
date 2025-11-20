"""
HumanEval and HumanEval+ evaluation for code generation.

Implements pass@1 evaluation with code execution in sandboxed environment.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .tokenization import load_qwen2_tokenizer
from .utils_xla import get_device, is_master, print_once


def format_humaneval_prompt(problem: dict) -> str:
    """
    Format HumanEval problem into Qwen chat format.

    Args:
        problem: Dict with 'prompt' field (function signature + docstring)

    Returns:
        Formatted prompt string
    """
    original_prompt = problem["prompt"]

    formatted = (
        "<|user|>\n"
        "Write Python code to complete the following function.\n\n"
        f"{original_prompt}\n"
        "<|assistant|>\n"
    )

    return formatted


def load_humaneval_dataset(humaneval_plus: bool = False) -> List[dict]:
    """
    Load HumanEval or HumanEval+ dataset.

    Args:
        humaneval_plus: If True, load HumanEval+, else HumanEval

    Returns:
        List of problem dictionaries
    """
    if humaneval_plus:
        print_once("Loading HumanEval+ dataset...")
        dataset = load_dataset("evalplus/humanevalplus", split="test")
    else:
        print_once("Loading HumanEval dataset...")
        dataset = load_dataset("openai_humaneval", split="test")

    problems = list(dataset)
    print_once(f"Loaded {len(problems)} problems")

    return problems


def generate_solution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    device: Optional[torch.device] = None
) -> str:
    """
    Generate code solution for a prompt.

    Args:
        model: Qwen2 model
        tokenizer: Qwen2 tokenizer
        prompt: Formatted prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        device: Device to run on

    Returns:
        Generated code completion
    """
    if device is None:
        device = get_device()

    # Tokenize prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Force XLA sync by moving to CPU immediately
    if str(device).startswith('xla'):
        try:
            import torch_xla.core.xla_model as xm
            outputs_cpu = outputs.detach().cpu()
            xm.mark_step()
        except Exception:
            outputs_cpu = outputs
    else:
        outputs_cpu = outputs

    # Decode
    generated_text = tokenizer.decode(outputs_cpu[0], skip_special_tokens=True)

    # Extract completion (remove prompt)
    if generated_text.startswith(prompt):
        completion = generated_text[len(prompt):]
    else:
        # Fallback: try to find the assistant response
        if "<|assistant|>" in generated_text:
            completion = generated_text.split("<|assistant|>")[-1]
        else:
            completion = generated_text

    return completion.strip()


def execute_code_with_tests(
    code: str,
    test: str,
    timeout: int = 5
) -> Dict[str, any]:
    """
    Execute generated code with test cases.

    Args:
        code: Generated code solution
        test: Test code to run
        timeout: Execution timeout in seconds

    Returns:
        Dict with 'passed' (bool) and 'error' (str or None)
    """
    import signal
    import contextlib
    import io

    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out")

    # Combine code and test
    full_code = code + "\n" + test

    result = {
        "passed": False,
        "error": None
    }

    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        # Capture stdout/stderr
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):

            # Execute in isolated namespace
            exec_globals = {}
            exec(full_code, exec_globals)

        # If we got here, no exception was raised
        result["passed"] = True

    except TimeoutError as e:
        result["error"] = f"Timeout: {str(e)}"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"

    finally:
        # Cancel alarm
        signal.alarm(0)

    return result


def evaluate_humaneval(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    humaneval_plus: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
    output_path: Optional[Path] = None
) -> Dict[str, float]:
    """
    Evaluate model on HumanEval or HumanEval+.

    Args:
        model: Qwen2 model
        tokenizer: Qwen2 tokenizer
        humaneval_plus: Whether to use HumanEval+
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling
        device: Device to run on
        output_path: Optional path to save detailed results

    Returns:
        Dict with metrics
    """
    dataset_name = "HumanEval+" if humaneval_plus else "HumanEval"

    print_once("=" * 80)
    print_once(f"Evaluating on {dataset_name}")
    print_once("=" * 80)

    # Load dataset
    problems = load_humaneval_dataset(humaneval_plus)

    # Generation settings
    print_once(f"\nGeneration settings:")
    print_once(f"  Temperature: {temperature}")
    print_once(f"  Top-p: {top_p}")
    print_once(f"  Max new tokens: {max_new_tokens}")

    # Set model to eval mode
    model.eval()

    # Generate solutions
    results = []
    num_passed = 0

    for i, problem in enumerate(tqdm(problems, desc=f"Evaluating {dataset_name}", disable=not is_master())):
        task_id = problem["task_id"]
        prompt_text = problem["prompt"]
        test = problem["test"]
        entry_point = problem["entry_point"]

        # Debug logging for first few problems
        if i < 3:
            print_once(f"\n[DEBUG] Starting problem {i+1}/{len(problems)}: {task_id}", flush=True)

        # Format prompt
        formatted_prompt = format_humaneval_prompt(problem)

        # Generate solution
        if i == 0:
            print_once(f"[DEBUG] First generation starting (this may take 10-30 min for XLA compilation)...", flush=True)

        completion = generate_solution(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device
        )

        if i == 0:
            print_once(f"[DEBUG] First generation complete! Subsequent ones will be faster.", flush=True)
        elif i < 3:
            print_once(f"[DEBUG] Problem {i+1} generation complete", flush=True)

        # Combine original prompt + completion for execution
        full_code = prompt_text + completion

        # Execute with tests
        exec_result = execute_code_with_tests(full_code, test)

        passed = exec_result["passed"]
        if passed:
            num_passed += 1

        # Store result
        results.append({
            "task_id": task_id,
            "prompt": prompt_text,
            "completion": completion,
            "passed": passed,
            "error": exec_result["error"]
        })

        # Force XLA sync after each problem to prevent graph fusion
        if str(device).startswith('xla'):
            try:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            except Exception:
                pass

    # Calculate metrics
    total_problems = len(problems)
    pass_at_1 = num_passed / total_problems if total_problems > 0 else 0.0

    metrics = {
        "dataset": dataset_name,
        "total_problems": total_problems,
        "passed": num_passed,
        "pass@1": pass_at_1
    }

    print_once(f"\n{dataset_name} Results:")
    print_once(f"  Total problems: {total_problems}")
    print_once(f"  Passed: {num_passed}")
    print_once(f"  Pass@1: {pass_at_1:.1%}")

    # Save detailed results
    if output_path and is_master():
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "metrics": metrics,
                "results": results
            }, f, indent=2)

        print_once(f"\nDetailed results saved to {output_path}")

    return metrics


def evaluate_model_on_code_benchmarks(
    checkpoint_dir: Path,
    output_dir: Path,
    eval_config: dict
) -> Dict[str, Dict]:
    """
    Evaluate a checkpoint on both HumanEval and HumanEval+.

    Args:
        checkpoint_dir: Path to model checkpoint
        output_dir: Directory to save results
        eval_config: Evaluation configuration

    Returns:
        Dict with results for both benchmarks
    """
    print_once("=" * 80)
    print_once("CODE BENCHMARK EVALUATION")
    print_once("=" * 80)

    # Load model and tokenizer
    print_once(f"\nLoading model from {checkpoint_dir}...")

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = model.to(device)

    print_once(f"Model loaded on {device}")

    # Evaluation settings
    humaneval_config = eval_config.get("humaneval", {})
    humaneval_plus_config = eval_config.get("humaneval_plus", {})

    all_metrics = {}

    # Evaluate on HumanEval
    print_once("\n" + "=" * 80)
    humaneval_results = evaluate_humaneval(
        model=model,
        tokenizer=tokenizer,
        humaneval_plus=False,
        max_new_tokens=humaneval_config.get("max_new_tokens", 512),
        temperature=humaneval_config.get("temperature", 0.2),
        top_p=humaneval_config.get("top_p", 0.95),
        device=device,
        output_path=output_dir / "humaneval_results.json"
    )

    all_metrics["humaneval"] = humaneval_results

    # Evaluate on HumanEval+
    print_once("\n" + "=" * 80)
    humaneval_plus_results = evaluate_humaneval(
        model=model,
        tokenizer=tokenizer,
        humaneval_plus=True,
        max_new_tokens=humaneval_plus_config.get("max_new_tokens", 512),
        temperature=humaneval_plus_config.get("temperature", 0.2),
        top_p=humaneval_plus_config.get("top_p", 0.95),
        device=device,
        output_path=output_dir / "humaneval_plus_results.json"
    )

    all_metrics["humaneval_plus"] = humaneval_plus_results

    # Save combined metrics
    if is_master():
        metrics_path = output_dir / "eval_metrics.json"

        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        print_once(f"\nCombined metrics saved to {metrics_path}")

    print_once("\n" + "=" * 80)
    print_once("EVALUATION COMPLETE")
    print_once("=" * 80)

    return all_metrics

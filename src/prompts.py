"""
Unified prompt formatting for instruction-following.

Formats examples into Qwen-compatible chat format with user/assistant markers.
"""

def format_instruction_prompt(instruction: str, input_text: str = "", output: str = "") -> dict:
    """
    Format an instruction-following example into Qwen chat format.

    Args:
        instruction: The instruction/task description
        input_text: Optional additional input context
        output: The expected output/completion

    Returns:
        dict with 'prompt' (input portion) and 'text' (full prompt+output)
    """
    # Build prompt (input portion only)
    if input_text and input_text.strip():
        prompt = (
            "<|user|>\n"
            f"Instruction: {instruction}\n\n"
            f"Input:\n{input_text}\n"
            "<|assistant|>\n"
        )
    else:
        prompt = (
            "<|user|>\n"
            f"Instruction: {instruction}\n"
            "<|assistant|>\n"
        )

    # Full text includes output and EOS
    full_text = prompt + output + "<|endoftext|>"

    return {
        "prompt": prompt,
        "output": output,
        "text": full_text
    }


def format_chat_messages(messages: list) -> dict:
    """
    Format a list of chat messages into Qwen format.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str}

    Returns:
        dict with 'prompt' and 'text'
    """
    parts = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")

    # Last message should be assistant, extract it as output
    if messages[-1]["role"] == "assistant":
        output = messages[-1]["content"]
        prompt = "".join(parts[:-1]) + "<|assistant|>\n"
    else:
        # If last message is user, output is empty
        output = ""
        prompt = "".join(parts)

    full_text = "".join(parts) + "<|endoftext|>"

    return {
        "prompt": prompt,
        "output": output,
        "text": full_text
    }

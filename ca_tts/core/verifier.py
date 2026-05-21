"""Expert verification module for CA-TTS."""
import os
import io
import base64
import re
import ast
import enum
import sys
from typing import Any, List, Union, Optional, Dict, Tuple
import torch
from .algorithms import prepare_inputs

try:
    import google.generativeai as genai
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Missing required libraries. Please run: pip install google-generativeai pillow")
    sys.exit(1)

from .prompts import (
    SELF_CONSISTENCY_VERIFIER_PROMPT,
    SELF_REFLECTION_VERIFIER_PROMPT
)

try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except TypeError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set your API key: export GOOGLE_API_KEY='your_key_here'")
    pass


class VerificationMode(enum.Enum):
    """Defines different verification modes."""
    SELF_CONSISTENCY = "self_consistency"
    SELF_CORRECTION = "self_correction"
    SELF_REFLECTION = "self_reflection"


PROMPT_MAP = {
    VerificationMode.SELF_CONSISTENCY: SELF_CONSISTENCY_VERIFIER_PROMPT.strip(),
    VerificationMode.SELF_CORRECTION: SELF_REFLECTION_VERIFIER_PROMPT.strip(),
    VerificationMode.SELF_REFLECTION: SELF_REFLECTION_VERIFIER_PROMPT.strip(),
}

def extract_output(response_text: str, mode: Any) -> Any:
    """
    Extract and parse output from model response based on verification mode.

    Args:
        response_text: Raw string response from the model.
        mode: The VerificationMode being used.

    Returns:
        - List[float]: For SELF_CONSISTENCY mode, returns probabilities_list.
        - str: For other modes, returns the raw response string.
        - str: If parsing fails in SELF_CONSISTENCY mode, returns error message string.
    """

    # 1. Isolate the Assistant's reply
    #    Use case-insensitive re.split to find "assistant" and subsequent newlines
    #    This prevents us from parsing example lists in the prompt (user/system)
    parts = re.split(r"assistant\s*\n+", response_text, maxsplit=1, flags=re.IGNORECASE)

    assistant_reply = ""
    if len(parts) == 2:
        # Found "assistant" separator, take everything after it
        assistant_reply = parts[1]
    else:
        # No "assistant" separator found.
        # Possibly because the model's reply is very short, e.g., just "[0.95, 0.05]"
        # In this case, assume the entire response_text is the reply (lower risk)
        assistant_reply = response_text

    if mode == VerificationMode.SELF_CONSISTENCY:

        # 2. Find *all* non-greedy matches of lists in assistant_reply
        #    re.DOTALL (re.S) makes '.' also match newlines
        try:
            matches = re.findall(r"(\[.*?\])", assistant_reply, re.DOTALL)
        except Exception:
            matches = []  # In case assistant_reply is None or other type

        if not matches:
            # 3. If no '[]' blocks found in 'assistant' reply
            return (
                f"Error: No list format found in 'assistant' response (no '[]' matched).\n"
                f"Assistant response (truncated): {assistant_reply[-500:]}"
            )

        # 4. Extract the *last* matched list string
        #    This is most likely the model's final answer, even if it output other lists before
        list_string = matches[-1].strip()

        # 5. Try to parse the extracted string as a list
        try:
            parsed_probs = ast.literal_eval(list_string)

            # 6. Validate parsed result
            if not (isinstance(parsed_probs, list) and all(isinstance(x, (int, float)) for x in parsed_probs)):
                return f"Error: 'Probabilities' list is not a valid numeric list: {list_string}"

            # 7. Return only the probability list
            return parsed_probs

        except (ValueError, SyntaxError) as e:
            # 8. If ast.literal_eval fails (e.g., matched "[p_1, ..., p_n]")
            return (
                f"Error: Failed to parse last list extracted from 'assistant' response.\n"
                f"Extracted string='{list_string}'\n"
                f"Assistant response (truncated)='{assistant_reply[-500:]}'\n"
                f"Error: {e}"
            )

    elif mode == VerificationMode.SELF_CORRECTION:
        return assistant_reply  # Return isolated reply

    elif mode == VerificationMode.SELF_REFLECTION:
        """
        Expected format:
        Critique: Based on this question, your answer is "{model_answer}", <critique_text>

        Goal: Extract <critique_text>
        """
        critique_reply = assistant_reply.strip()

        pattern = re.compile(
            r'Critique: Based on this question, your answer is ".*?"\s*,\s*(.*)',
            re.DOTALL
        )

        match = pattern.search(critique_reply)

        if match:
            critique_text = match.group(1).strip()

            if critique_text:
                return critique_text
            else:
                return (
                    f"Error: Response format correct, but 'Critique' text is empty.\n"
                    f"Response (truncated): {critique_reply[-500:]}"
                )
        else:
            # Response string doesn't match expected format
            # Return isolated reply, not full response_text
            return critique_reply

    else:
        raise ValueError(f"Unknown verification mode: {mode}")


def _pil_to_base64(image: Image.Image) -> str:
    """Convert PIL.Image object to base64-encoded JPEG string."""
    buffered = io.BytesIO()
    # Ensure saving as RGB mode to avoid certain format issues
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def verify(
    image: Image.Image,
    question: str,
    verification_mode: Any,
    option_list: Optional[List[str]] = None,
    model_answer: Optional[str] = None,
    expert_model_name: str = "gemini-2.5-pro-06-17",
    local_generator: Optional[Any] = None,
    local_processor: Optional[Any] = None,
    image_path: Optional[str] = None,
    sampling_params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Verify input using expert model (local or API-based).

    Args:
        image: PIL.Image object.
        question: The text question.
        verification_mode: VerificationMode enum member.
        option_list: (SELF_CONSISTENCY only) List of candidate options.
        model_answer: (SELF_REFLECTION only) Model answer for critique.
        expert_model_name: Model name to use.
                          If starts with 'gpt-' or 'gemini-', use API.
                          Otherwise, use local model.
        local_generator: (Local model required)
        local_processor: (Local model required)
        image_path: (Local model required)
        sampling_params: (Local model required)

    Returns:
        Parsed output (type depends on verification_mode).
    """

    # --- Logic 1: Local model execution ---
    if not (expert_model_name.startswith("gpt-") or expert_model_name.startswith("gemini-") or expert_model_name.startswith("72") or expert_model_name.startswith("max")):

        print(f"--- Using mode: {verification_mode.name} (Local Model: {expert_model_name}) ---")

        # Check required components for local model
        if not all([local_generator, local_processor, image_path, sampling_params, torch, prepare_inputs]):
            missing = [
                "local_generator" if not local_generator else None,
                "local_processor" if not local_processor else None,
                "image_path" if not image_path else None,
                "sampling_params" if not sampling_params else None,
                "torch" if not torch else None,
                "prepare_inputs" if not prepare_inputs else None,
            ]
            missing_str = ", ".join(filter(None, missing))
            return f"Error: Local verification failed. Missing components: {missing_str}"

        # 1. Get instruction prompt
        prompt_text = PROMPT_MAP.get(verification_mode)
        if prompt_text is None:
            raise ValueError(f"No prompt found for {verification_mode}.")

        # 2. Format prompt
        if verification_mode == VerificationMode.SELF_CONSISTENCY:
            if not option_list:
                return "Error: SELF_CONSISTENCY mode requires 'option_list' to format prompt."
            try:
                prompt_text = prompt_text.format(options_list=str(option_list))
            except KeyError:
                return f"Error: SELF_CONSISTENCY prompt missing '{{options_list}}' placeholder."

        elif verification_mode == VerificationMode.SELF_REFLECTION:
            if model_answer is None:
                 return "Error: SELF_REFLECTION mode requires 'model_answer' to format prompt."
            try:
                prompt_text = prompt_text.format(question=question, model_answer=model_answer)
            except KeyError:
                return f"Error: SELF_REFLECTION prompt missing '{{model_answer}}' placeholder."

        # 3. Prepare local model input
        full_prompt_text = f"{question}\n\n{prompt_text}"

        try:
            inputs = prepare_inputs(local_processor, image_path, full_prompt_text)
            inputs = {k: v.to(local_generator.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        except Exception as e:
            return f"Error: Local 'prepare_inputs' failed: {e}"

        # 4. Call local generator
        try:
            # Use deterministic sampling for verification
            local_sampling_params = sampling_params.copy()
            local_sampling_params['do_sample'] = False

            trace = local_generator.generate_sample(inputs, local_sampling_params)
            response_text = trace['full_text']

            print(f"Raw Local response:\n{'-'*20}\n{response_text}\n{'-'*20}")

        except Exception as e:
            return f"Error: Local 'generate_sample' failed: {e}"

        # 5. Extract output
        parsed_output = extract_output(response_text, verification_mode)

        # 6. Length validation
        if verification_mode == VerificationMode.SELF_CONSISTENCY:
            if not isinstance(parsed_output, list):
                return f"Parsing output failed: {parsed_output}"

            parsed_probs = parsed_output

            if len(option_list) != len(parsed_probs):
                return (
                    f"Error: Options list (length {len(option_list)}) "
                    f"does not match Probabilities list (length {len(parsed_probs)}).\n"
                    f"Response: {response_text}"
                )

            print(f"Parsed probabilities: {parsed_probs}")
            print(f"Input options: {option_list}")
            return parsed_probs, option_list

        # For (SELF_CORRECTION, SELF_REFLECTION)
        return parsed_output

    # --- Logic 2: API execution (OpenAI) ---
    try:
        from openai import OpenAI
        from openai import APIError
    except ImportError:
        return "Error: 'openai' library required. Please run 'pip install openai'"

    print(f"--- Using mode: {verification_mode.name} (OpenAI: {expert_model_name}) ---")

    # 1. Initialize OpenAI client
    try:
        api_key = os.environ.get("EXPERT_API_KEY")
        base_url = os.environ.get("EXPERT_API_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            return "Error: EXPERT_API_KEY environment variable not set. Please set it before using API-based verification."
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        return f"Error: Failed to initialize OpenAI client. Please check EXPERT_API_KEY environment variable: {e}"

    # 2. Get instruction prompt
    prompt_text = PROMPT_MAP.get(verification_mode)
    if prompt_text is None:
        raise ValueError(f"No prompt found for {verification_mode}.")

    # 3. Format prompt based on mode
    if verification_mode == VerificationMode.SELF_CONSISTENCY:
        if not option_list:
            return "Error: SELF_CONSISTENCY mode requires 'option_list' to format prompt."
        try:
            prompt_text = prompt_text.format(options_list=str(option_list))
        except KeyError:
            return f"Error: SELF_CONSISTENCY prompt missing '{{options_list}}' placeholder."

    elif verification_mode == VerificationMode.SELF_REFLECTION:
        if model_answer is None:
             return "Error: SELF_REFLECTION mode requires 'model_answer' to format prompt."
        try:
            prompt_text = prompt_text.format(question=question, model_answer=model_answer)
        except KeyError:
            return f"Error: SELF_REFLECTION prompt missing '{{model_answer}}' placeholder."

    # 4. Build multimodal API payload
    base64_image = _pil_to_base64(image)
    content_parts = []

    if prompt_text:
        content_parts.append({
            "type": "text",
            "text": prompt_text
        })

    content_parts.append({
        "type": "text",
        "text": f"User question: {question}"
    })

    content_parts.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
            "detail": "high"
        }
    })

    messages = [
        {
            "role": "user",
            "content": content_parts
        }
    ]

    if expert_model_name.startswith("72"):
        expert_model_name = "qwen2.5-vl-72b-instruct"

    if expert_model_name.startswith("max"):
        expert_model_name = "qwen-vl-max-latest"

    # 5. Call API
    try:
        response = client.chat.completions.create(
            model=expert_model_name,
            messages=messages,
            max_tokens=2048
        )

        response_text = response.choices[0].message.content

        print(f"Raw API response:\n{'-'*20}\n{response_text}\n{'-'*20}")

    except APIError as e:
        if "AuthenticationError" in str(e):
             return "Error: OpenAI API key invalid or not set. Please check EXPERT_API_KEY."
        return f"Error: OpenAI API call failed: {e}"
    except Exception as e:
        return f"Error: Unexpected error occurred: {e}"

    # 6. Extract and return output
    parsed_output = extract_output(response_text, verification_mode)

    # 7. Length validation
    if verification_mode == VerificationMode.SELF_CONSISTENCY:
        if not isinstance(parsed_output, list):
            return f"Parsing output failed: {parsed_output}"

        parsed_probs = parsed_output

        if len(option_list) != len(parsed_probs):
            return (
                f"Error: Options list (length {len(option_list)}) "
                f"does not match Probabilities list (length {len(parsed_probs)}).\n"
                f"Response: {response_text}"
            )

        print(f"Parsed probabilities: {parsed_probs}")
        print(f"Input options: {option_list}")
        return parsed_probs, option_list

    # For (SELF_CORRECTION, SELF_REFLECTION)
    return parsed_output

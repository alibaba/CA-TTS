"""Core algorithms for confidence computation, voting, and diffusion noise."""
import logging
import warnings
import torch
import math
import numpy as np
import time
import json
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Tuple

from PIL import Image
from transformers import AutoProcessor
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

logger = logging.getLogger(__name__)

# ---------------- InternVL image helpers ----------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_internvl_transform(input_size: int):
    """Create the default InternVL transform."""
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def dynamic_preprocess_for_internvl(image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448,
                                    use_thumbnail: bool = False) -> List[Image.Image]:
    """
    Split the input image into multiple square tiles while preserving aspect ratio.
    Used for InternVL models that require dynamic resolution preprocessing.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    def find_closest_aspect_ratio(target_aspect_ratio: float, ratios: List[Tuple[int, int]]) -> Tuple[int, int]:
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = orig_width * orig_height
        for ratio in ratios:
            r = ratio[0] / ratio[1]
            ratio_diff = abs(target_aspect_ratio - r)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_internvl_image(image_path: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    """Load and preprocess an image into InternVL pixel_values tensor."""
    image = load_image(image_path)
    transform = build_internvl_transform(input_size=input_size)
    images = dynamic_preprocess_for_internvl(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def prepare_internvl_inputs(
    tokenizer,
    image_path: str,
    prompt_text: str,
    input_size: int = 448,
    max_num: int = 12,
) -> Dict[str, Any]:
    """
    Prepare inputs for InternVL chat-style inference.
    Returns a dict compatible with InternVLGenerator.
    """
    pixel_values = load_internvl_image(image_path, input_size=input_size, max_num=max_num)
    return {
        "pixel_values": pixel_values,
        "question": prompt_text,
        "tokenizer": tokenizer,
        "num_patches_list": [pixel_values.shape[0]]
    }

def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a local path.
    Uses context manager to ensure file handles are properly closed.
    """
    try:
        with Image.open(image_path) as img:
            image = img.copy()
    except Exception as e:
        logger.error(f"Error opening local image file: {e}")
        raise
            
    return image.convert("RGB")

def prepare_inputs(
        processor: AutoProcessor,
        image_path: str,
        prompt_text: str
    ) -> Dict[str, Any]:
        """Prepare model inputs using Qwen-VL's apply_chat_template."""
        image = load_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        return inputs

def prepare_noise_inputs(
    processor: AutoProcessor,
    noisy_image: Image.Image,
    prompt_text: str
) -> Dict[str, Any]:
    """
    Prepare model inputs for a noisy image.
    Accepts a PIL.Image object instead of an image path.
    """
    messages = [
        {
            "role": "user",
            "content": [
        {"type": "image", "image": noisy_image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    return inputs

# --- Answer Extraction ---
def extract_answer(text: str, format: str = 'mathvision') -> Optional[str]:
    """
    Extract a 'boxed' answer from text, supporting different dataset formats.

    Args:
        text: The model's output text.
        format: Answer format ('default' or 'mathvision').
    """
    
    search_key = ""
    if format == 'mathvision':
        # MathVision 格式: \boxed{...}
        search_key = "\\boxed"
    elif format == 'default':
        # 原始 DeepConf 逻辑
        search_key = "boxed"
    else:
        # 默认回退到原始逻辑
        search_key = "boxed"

    if search_key in text:
        # 从最后一个匹配项开始查找，以获取最终答案
        ans_part = text.split(search_key)[-1]
        
        # 查找第一个 '{'
        start_idx = ans_part.find('{')
        if start_idx == -1:
            # 兼容没有花括号的旧格式, e.g., "boxed $ANSWER$"
            if format == 'default':
                return ans_part.split("$")[0].strip()
            return None # MathVision 格式必须有花括号

        # --- 使用堆栈处理嵌套花括号 ---
        # e.g., \boxed{\frac{1}{2}}
        ans_part = ans_part[start_idx+1:] # 获取 '{' 之后的内容
        stack = 1
        answer = ""
        for char in ans_part:
            if char == '{':
                stack += 1
                answer += char
            elif char == '}':
                stack -= 1
                if stack == 0:
                    break # 找到匹配的 '}'
                answer += char
            else:
                answer += char
        
        if stack != 0:
            return None # 括号不匹配
        
        extracted = answer.strip()
        # 如果提取的结果是空字符串或字符串 "None"，也返回 None
        if not extracted or extracted == "None":
            return None
        
        return extracted
    
    return None

# --- Confidence Computation ---

def compute_token_confidence(
    scores: Tuple[torch.Tensor], 
    top_k: int = 20
) -> List[float]:
    """
    [Re-implemented] Simulates the vLLM (DeepThink) 'logprobs' confidence logic.

    This function computes the "Negative Mean Log-Probability" (NMLP) of the
    Top-K tokens. This serves as a robust measure of the distribution's
    "peakedness" or certainty at each generation step.

    This implementation correctly mirrors the vLLM code's *logic* (averaging 
    the top K logprobs), not its misleading comment.

    Args:
        scores: The 'scores' tuple from model.generate(). Each element is
                a tensor of shape [batch_size, vocab_size].
        top_k: The number of top tokens to average (e.g., 20, to match
               the vLLM 'logprobs=20' parameter).

    Returns:
        A list of confidence scores (NMLP), rounded to 3 decimal places.
    """
    
    token_confs = []
    if not scores or top_k <= 0:
        logger.warning(f"Warning: 'scores' tuple is empty or 'top_k' <= 0. Returning empty list.")
        return []

    for step_scores in scores:
        # step_scores shape is [batch_size, vocab_size], e.g., [1, 32000]
        
        try:
            # 1. Convert Logits to Log-Probabilities
            # Use .float() for numerical stability
            log_probs = torch.log_softmax(step_scores.float(), dim=-1)
            
            # 2. Get the Top-K log probabilities
            # top_k_logprobs will have shape [batch_size, top_k]
            # We ensure top_k is not larger than the vocab size
            k = min(top_k, log_probs.shape[-1])
            top_k_logprobs, _ = torch.topk(log_probs, k=k, dim=-1)
            
            # 3. Calculate the mean of these K values
            # .mean() will reduce the tensor to a scalar
            mean_logprob = torch.mean(top_k_logprobs)
            
            # 4. Confidence = Negative Mean Log-Probability (NMLP)
            conf = -mean_logprob.item()
            
            # 5. Round to 3 decimal places to mimic the vLLM logic
            token_confs.append(round(conf, 3))

        except Exception as e:
            logger.error(f"Error processing step_scores tensor: {e}. Appending 'inf'.")
            token_confs.append(float('inf'))
    
    # print(f"confidences: {token_confs}")

    return token_confs

def compute_token_confidence_vllm(logprobs: List[Dict]) -> List[float]:
    """
    Compute token confidence from vLLM-format logprobs.

    Args:
        logprobs: vLLM logprobs list (List[Dict[token_id, LogprobObject]])
                  where LogprobObject has a .logprob attribute.

    Returns:
        Confidence list per token (negative mean logprob).
    """
    confs = []
    for token_logprobs in logprobs:
        if token_logprobs:
            # vLLM returns {token_id: Logprob object} dicts; compute mean of top-k logprobs
            mean_logprob = np.mean([lp.logprob for lp in token_logprobs.values()])
            confs.append(round(-mean_logprob, 3))
    return confs

def compute_sliding_window_confidence(confs: List[float], window_size: int) -> List[float]:
    """Compute sliding window average confidence."""
    if len(confs) < window_size:
        return [sum(confs) / len(confs)] if confs else [0]
    
    sliding_means = []
    for i in range(len(confs) - window_size + 1):
        window = confs[i:i + window_size]
        # Round to match reference implementation
        sliding_means.append(round(sum(window) / len(window), 3))
    return sliding_means

# --- Trace-level Confidence Computation ---

def calculate_mean_confidence(trace: Dict[str, Any]) -> float:
    """Calculate mean confidence from 'confs' in trace."""
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            return np.mean(confs) if confs else 0.0
        return 0.0
    except Exception:
        return 0.0

def calculate_tail_confidence(trace: Dict[str, Any], tail_tokens: int = 20) -> float:
    """Calculate mean confidence of the last N tokens."""
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
            return np.mean(tail_confs) if tail_confs else 0.0
        return 0.0
    except Exception:
        return 0.0

def calculate_bottom_window_confidence(trace: Dict[str, Any], window_size: int = 20, bottom_percent: float = 0.1) -> float:
    """
    Compute sliding window confidence and return the mean of bottom x% windows.

    Args:
        trace: Trace dict containing a 'confs' list.
        window_size: Sliding window size.
        bottom_percent: 0.1 = bottom 10%, -1 = minimum window.
    """
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            if len(confs) < window_size:
                return np.mean(confs) if confs else 0.0
            
            window_means = []
            current_sum = sum(confs[:window_size])
            window_means.append(current_sum / window_size)
            
            for i in range(1, len(confs) - window_size + 1):
                current_sum = current_sum - confs[i-1] + confs[i + window_size - 1]
                window_means.append(current_sum / window_size)
            
            if not window_means:
                return 0.0
            
            if bottom_percent == -1:  # Min window
                return min(window_means)
            
            num_bottom = max(1, int(len(window_means) * bottom_percent))
            if num_bottom == 1:
                return min(window_means)
            else:
                bottom_means = np.partition(window_means, num_bottom-1)[:num_bottom]
                return np.mean(bottom_means)
        
        return 0.0
    except Exception:
        return 0.0


def _apply_verifier_boost(
    scores: Dict[str, float],
    num_samples: int,
    tau: float,
    verifier_output: Optional[List[float]],
    options_list: Optional[List[str]]
) -> Dict[str, float]:
    """
    Helper function: Applies a boost to voting/weight scores based on verifier output.

    Args:
        scores: The original vote counts or weight sums (e.g., {'A': 10.5, 'B': 2.0}).
        num_samples: The total number of samples (len(answers)).
        tau: The boost strength (0.0 to 1.0).
        verifier_output: The probability list from the verifier (e.g., [0.8, 0.2]).
        options_list: The list of option names corresponding to the probabilities (e.g., ['A', 'B']).

    Returns:
        The new scores after applying the boost.
    """
    
    # --- 1. Validate inputs for boosting ---
    is_valid_input = (
        verifier_output is not None and
        options_list is not None and
        len(verifier_output) == len(options_list) and
        num_samples > 0 and
        tau > 0.0
    )

    if not is_valid_input:
        # Warn if user intended to boost (tau > 0) but provided invalid inputs
        if tau > 0.0:
            warnings.warn(
                "Verifier boost requested (tau > 0) but 'verifier_output' or 'options_list' "
                "are missing, empty, or mismatched. Skipping boost."
            )
        return scores # Return original scores if boost cannot be applied

    # --- 2. Calculate total boost mass ---
    # This is the total "boost vote" mass to be distributed.
    total_boost_mass = int(tau * num_samples)
    
    if total_boost_mass == 0:
        # Boost strength too low or too few samples, resulting in zero integer boost.
        return scores 

    # --- 3. Normalize verifier probabilities ---
    prob_sum = sum(verifier_output)
    if prob_sum <= 1e-6: # Avoid division by zero
        warnings.warn("Verifier probabilities sum to zero. Cannot apply boost.")
        return scores
        
    # Copy original scores to avoid mutating the original dict
    boosted_scores = scores.copy()

    # --- 4. Distribute boost weights ---
    for option_str, prob in zip(options_list, verifier_output):
        # Distribute the total_boost_mass according to the verifier's probability
        # e.g., total_boost_mass=10, prob=0.8 -> boost_value = (0.8 / 1.0) * 10 = 8
        boost_value = (prob / prob_sum) * total_boost_mass
        
        # Add the distributed boost value to the original score
        boosted_scores[option_str] = boosted_scores.get(option_str, 0.0) + boost_value
        
    return boosted_scores


def apply_external_boost_and_vote(
    vote_scores: Dict[str, float],
    boost_answer: str,
    boost_value: float
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Apply an external answer boost (from SR or SCk) to existing vote scores and find the new winner.

    Args:
        vote_scores: Original {answer: score} dictionary.
        boost_answer: Answer to boost (from SR or SCk).
        boost_value: Boost value to add (e.g., 0.5 * num_samples).

    Returns:
        (winning_answer, new_vote_scores) tuple.
    """
    # 1. Handle empty input
    if not vote_scores:
        if boost_answer and boost_value > 0:
            new_scores = {boost_answer: boost_value}
            return boost_answer, new_scores
        else:
            return None, {}

    # 2. Check if boost is needed
    if not boost_answer or boost_value == 0:
        try:
            winner = max(vote_scores.keys(), key=lambda x: vote_scores[x])
            return winner, vote_scores
        except ValueError:
             return None, {}

    # 3. Copy and apply boost
    new_scores = vote_scores.copy()
    boost_answer_str = str(boost_answer)
    new_scores[boost_answer_str] = new_scores.get(boost_answer_str, 0.0) + boost_value

    # 4. Find new winner
    new_winner = max(new_scores.keys(), key=lambda x: new_scores[x])
    
    return new_winner, new_scores


# --- Voting Functions ---

def simple_majority_vote(
    answers: List[str],
    verify_mode: bool = False,
    tau: float = 0.0,
    verifier_output: Optional[List[float]] = None,
    options_list: Optional[List[str]] = None
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Simple majority vote. Applies verifier boost if verify_mode=True.
    Returns: (winning_answer, final_scores_dict)
    """
    if not answers:
        return None, {}
    
    # Filter out None and empty answers
    valid_answers = [a for a in answers if a is not None and str(a).strip() != "None" and str(a).strip()]
    if not valid_answers:
        return None, {}
        
    vote_counts = Counter(valid_answers)
    num_samples = len(valid_answers)
    
    final_scores: Dict[str, float] = {str(k): float(v) for k, v in vote_counts.items()}
    
    if verify_mode:
        final_scores = _apply_verifier_boost(
            final_scores, 
            num_samples, 
            tau, 
            verifier_output, 
            options_list
        )
        
    if not final_scores:
        return None, {}
        
    winner = max(final_scores.keys(), key=lambda x: final_scores[x])
    return winner, final_scores

def weighted_majority_vote(
    answers: List[str],
    weights: List[float],
    verify_mode: bool = False,
    tau: float = 0.0,
    verifier_output: Optional[List[float]] = None,
    options_list: Optional[List[str]] = None
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Weighted majority vote. Returns: (winning_answer, final_scores_dict)
    """
    if not answers:
        return None, {}
    
    # Filter out None and empty answers
    filtered_data = [(a, w) for a, w in zip(answers, weights)
                     if a is not None and str(a).strip() != "None" and str(a).strip()]
    if not filtered_data:
        return None, {}
        
    answer_weights = {}
    num_samples = len(filtered_data)
    
    for answer, weight in filtered_data:
        answer_str = str(answer)
        answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    
    final_scores = answer_weights
    
    if verify_mode:
        final_scores = _apply_verifier_boost(
            final_scores, 
            num_samples, 
            tau, 
            verifier_output, 
            options_list
        )
        
    if not final_scores:
        return None, {}
        
    winner = max(final_scores.keys(), key=lambda x: final_scores[x])
    return winner, final_scores

def weighted_majority_vote_with_noise_penalty(
    answers: List[str],
    weights: List[float],
    noise_steps: List[int],
    penalty_type: str = 'none',
    linear_penalty_factor: float = 0.1,
    max_noise_step: int = 999,
    verify_mode: bool = False,
    tau: float = 0.0,
    verifier_output: Optional[List[float]] = None,
    options_list: Optional[List[str]] = None
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Weighted majority vote with noise penalty. Returns: (winning_answer, final_scores_dict)
    """
    if not answers:
        return None, {}
    
    # Filter out None and empty answers
    filtered_data = [(a, w, ns) for a, w, ns in zip(answers, weights, noise_steps)
                     if a is not None and str(a).strip() != "None" and str(a).strip()]
    if not filtered_data:
        return None, {}
            
    answer_weights = {}
    num_samples = len(filtered_data)
    
    num_noise_samples = len(filtered_data) - 1 
    
    for k, (answer, base_weight, noise_step) in enumerate(filtered_data):
        base_weight = float(base_weight)
        adjusted_weight = base_weight

        if penalty_type == 'linear':
            penalty_strength = float(noise_step) / max_noise_step
            penalty_amount = linear_penalty_factor * penalty_strength 
            adjusted_weight = base_weight - penalty_amount
        elif penalty_type == 'percentage_k':
            if num_noise_samples > 0:
                penalty_ratio = float(k) / num_noise_samples
                adjusted_weight = base_weight * (1.0 - penalty_ratio)
        
        final_weight = max(0.0, adjusted_weight)
        answer_str = str(answer)
        answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + final_weight

    final_scores = answer_weights
    
    if verify_mode:
        final_scores = _apply_verifier_boost(
            final_scores, 
            num_samples, 
            tau, 
            verifier_output, 
            options_list
        )

    if not final_scores:
        return None, {}
    
    winner = max(final_scores.keys(), key=lambda x: final_scores[x])
    return winner, final_scores

def filter_top_confidence(traces: List[Dict[str, Any]], confidence_type: str = 'tail', top_percent: float = 0.1) -> List[Dict[str, Any]]:
    """Filter traces by confidence, keeping top %."""
    if not traces:
        return []
    
    # 1. Compute confidence
    confidences = []
    for trace in traces:
        if confidence_type == 'mean':
            conf = calculate_mean_confidence(trace)
        elif confidence_type == 'tail':
            conf = calculate_tail_confidence(trace)
        elif confidence_type == 'bottom_window':
            conf = calculate_bottom_window_confidence(trace)
        elif confidence_type == 'min_window':
            # Use bottom_percent=-1 to get min_window
            conf = calculate_bottom_window_confidence(trace, bottom_percent=-1)
        else:
            # Default
            conf = calculate_mean_confidence(trace)
        confidences.append(conf)
    
    # 2. Get threshold
    threshold = np.percentile(confidences, (1 - top_percent) * 100)

    # 3. Filter
    filtered_traces = []
    for trace, conf in zip(traces, confidences):
        if conf >= threshold:
            filtered_traces.append(trace)
    
    return filtered_traces

def append_confidences_to_jsonl(
    output_path: str,
    answers: List[str],
    mean_confidences: List[Union[float, np.float64]],
    tail_confidences: List[Union[float, np.float64]],
    bottom_window_confidences: List[Union[float, np.float64]],
    min_window_confidences: List[Union[float, np.float64]],
    meta_info: Optional[Dict[str, Any]] = None
):
    """
    Pack four confidence lists and answers into a JSON Lines entry and append to file.

    Args:
        output_path: Path to the JSON Lines output file.
        answers: Answer list for each sample.
        ..._confidences: Four confidence lists.
        meta_info: Optional metadata dict (e.g., id, noise_level).
    """

    def convert_np_to_float(data_list):
        return [float(x) for x in data_list]

    data_entry = {
        "timestamp": time.time(),
        "answers": answers, # <-- 2. 添加答案到字典
        "mean_confidences": convert_np_to_float(mean_confidences),
        "tail_confidences": convert_np_to_float(tail_confidences),
        "bottom_window_confidences": convert_np_to_float(bottom_window_confidences),
        "min_window_confidences": convert_np_to_float(min_window_confidences),
    }
    
    if meta_info:
        # 将元信息合并到数据字典中，方便追踪
        data_entry.update(meta_info)

    # 2. 将字典序列化为 JSON 字符串
    json_line = json.dumps(data_entry, ensure_ascii=False)
    
    # 3. 以追加模式 ('a') 写入文件，并添加换行符
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json_line + '\n')
        # print(f"Successfully appended data to {output_path}")
    except IOError as e:
        print(f"Error writing to file {output_path}: {e}")

def compute_voting_results(
    traces: List[Dict[str, Any]], 
    add_noise: bool,
    output_path: str,
    verify_mode: bool = False,
    tau: float = 0.0,
    verifier_output: Optional[List[float]] = None,
    options_list: Optional[List[str]] = None,
    meta_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    (已修改)
    Computes voting results across all voting methods.
    Returns a dict containing 'vote_scores' for downstream SR/SCk augmentation.
    """
    # Filter out None, empty strings, and the string "None"
    valid_traces = [
        trace for trace in traces
        if trace.get('extracted_answer') is not None
        and str(trace.get('extracted_answer')).strip() != "None"
        and str(trace.get('extracted_answer')).strip()
    ]

    base_methods = [
        'majority', 'mean_confidence_weighted', 'tail_confidence_weighted',
        'bottom_window_weighted', 'min_window_weighted',
        'top10_tail_filtered', 'top10_bottom_window_filtered'
    ]
    # Base keys + boosted variants
    all_method_keys = base_methods + [f"{m}_boosted" for m in base_methods]

    # Placeholder keys for SR and SCk augmentation results
    sr_sck_keys = [
        'sc_plus_sr_majority_boosted',
        'sc_plus_sr_mean_confidence_weighted_boosted',
        'sc_plus_sr_plus_sck_majority_boosted',
        'sc_plus_sr_plus_sck_mean_confidence_weighted_boosted'
    ]
    all_method_keys.extend(sr_sck_keys)

    if not valid_traces:
        return {method: None for method in all_method_keys}

    answers = [trace['extracted_answer'] for trace in valid_traces]

    ANALYSIS_WINDOW_SIZE = 20
    ANALYSIS_TAIL_TOKENS = 50

    mean_confidences = [calculate_mean_confidence(trace) for trace in valid_traces]
    tail_confidences = [calculate_tail_confidence(trace, tail_tokens=ANALYSIS_TAIL_TOKENS) for trace in valid_traces]
    bottom_window_confidences = [calculate_bottom_window_confidence(trace, window_size=ANALYSIS_WINDOW_SIZE) for trace in valid_traces]
    min_window_confidences = [calculate_bottom_window_confidence(trace, window_size=ANALYSIS_WINDOW_SIZE, bottom_percent=-1) for trace in valid_traces]

    if add_noise and output_path:
        # Assumes valid_traces (and thus answers/confidences) are ordered by noise step (t=0, ..., t=999)
        try:
            append_confidences_to_jsonl(
                output_path=output_path,
                answers=answers,
                meta_info=meta_info,
                mean_confidences=mean_confidences,
                tail_confidences=tail_confidences,
                bottom_window_confidences=bottom_window_confidences,
                min_window_confidences=min_window_confidences,
            )
        except Exception as e:
            logger.warning("Failed to write noise confidence log to %s: %s", output_path, e)

    pre_boost_args = {"verify_mode": False, "tau": 0.0, "verifier_output": verifier_output, "options_list": options_list}
    post_boost_args = {"verify_mode": verify_mode, "tau": tau, "verifier_output": verifier_output, "options_list": options_list}

    if add_noise:
        noise_steps = [trace.get('noise_step', 0) for trace in valid_traces]
        penalty_type_to_use = 'linear'
        vote_function_pre_boost = lambda ans, wts: weighted_majority_vote_with_noise_penalty(ans, wts, noise_steps=noise_steps, penalty_type=penalty_type_to_use, **pre_boost_args)
        vote_function_post_boost = lambda ans, wts: weighted_majority_vote_with_noise_penalty(ans, wts, noise_steps=noise_steps, penalty_type=penalty_type_to_use, **post_boost_args)
        logger.debug("Using noise-penalized voting function (type: %s)", penalty_type_to_use)
    else:
        vote_function_pre_boost = lambda ans, wts: weighted_majority_vote(ans, wts, **pre_boost_args)
        vote_function_post_boost = lambda ans, wts: weighted_majority_vote(ans, wts, **post_boost_args)
    
    voting_results = {}
    
    # 1. Simple Majority Vote
    majority_answer_pre, majority_scores_pre = simple_majority_vote(answers, **pre_boost_args)
    voting_results['majority'] = {
        'answer': majority_answer_pre,
        'num_votes': len(answers),
        'confidence': None,
        'vote_scores': majority_scores_pre 
    }
    majority_answer_post, majority_scores_post = simple_majority_vote(answers, **post_boost_args)
    voting_results['majority_boosted'] = {
        'answer': majority_answer_post,
        'num_votes': len(answers),
        'confidence': None,
        'vote_scores': majority_scores_post 
    }
    
    # 2. Mean Confidence Weighted
    if any(c > 0 for c in mean_confidences):
        mean_weighted_answer_pre, mean_weighted_scores_pre = vote_function_pre_boost(answers, mean_confidences)
        voting_results['mean_confidence_weighted'] = {
            'answer': mean_weighted_answer_pre,
            'num_votes': len(answers),
            'confidence': np.mean(mean_confidences),
            'vote_scores': mean_weighted_scores_pre 
        }
        mean_weighted_answer_post, mean_weighted_scores_post = vote_function_post_boost(answers, mean_confidences)
        voting_results['mean_confidence_weighted_boosted'] = {
            'answer': mean_weighted_answer_post,
            'num_votes': len(answers),
            'confidence': np.mean(mean_confidences),
            'vote_scores': mean_weighted_scores_post 
        }
    
    # 3. Tail Confidence Weighted
    if any(c > 0 for c in tail_confidences):
        tail_weighted_answer_pre, tail_weighted_scores_pre = vote_function_pre_boost(answers, tail_confidences)
        voting_results['tail_confidence_weighted'] = {
            'answer': tail_weighted_answer_pre,
            'num_votes': len(answers),
            'confidence': np.mean(tail_confidences),
            'vote_scores': tail_weighted_scores_pre 
        }
        tail_weighted_answer_post, tail_weighted_scores_post = vote_function_post_boost(answers, tail_confidences)
        voting_results['tail_confidence_weighted_boosted'] = {
            'answer': tail_weighted_answer_post,
            'num_votes': len(answers),
            'confidence': np.mean(tail_confidences),
            'vote_scores': tail_weighted_scores_post 
        }
    
    # 4. Bottom Window Confidence Weighted
    if any(c > 0 for c in bottom_window_confidences):
        bottom_weighted_answer_pre, bottom_weighted_scores_pre = vote_function_pre_boost(answers, bottom_window_confidences)
        voting_results['bottom_window_weighted'] = {
            'answer': bottom_weighted_answer_pre,
            'num_votes': len(answers),
            'confidence': np.mean(bottom_window_confidences),
            'vote_scores': bottom_weighted_scores_pre 
        }
        bottom_weighted_answer_post, bottom_weighted_scores_post = vote_function_post_boost(answers, bottom_window_confidences)
        voting_results['bottom_window_weighted_boosted'] = {
            'answer': bottom_weighted_answer_post,
            'num_votes': len(answers),
            'confidence': np.mean(bottom_window_confidences),
            'vote_scores': bottom_weighted_scores_post 
        }
    
    # 5. Minimum Window Confidence Weighted
    if any(c > 0 for c in min_window_confidences):
        min_window_answer_pre, min_window_scores_pre = vote_function_pre_boost(answers, min_window_confidences)
        voting_results['min_window_weighted'] = {
            'answer': min_window_answer_pre,
            'num_votes': len(answers),
            'confidence': np.mean(min_window_confidences),
            'vote_scores': min_window_scores_pre 
        }
        min_window_answer_post, min_window_scores_post = vote_function_post_boost(answers, min_window_confidences)
        voting_results['min_window_weighted_boosted'] = {
            'answer': min_window_answer_post,
            'num_votes': len(answers),
            'confidence': np.mean(min_window_confidences),
            'vote_scores': min_window_scores_post 
        }
    
    # 6. Top 10% Tail Confidence Filtered + Weighted
    top_tail_traces = filter_top_confidence(valid_traces, 'tail', 0.1)
    if top_tail_traces:
        # ... (定义 top_tail_vote_func_pre/post - 不变) ...
        top_tail_answers = [trace['extracted_answer'] for trace in top_tail_traces]
        top_tail_confidences = [calculate_tail_confidence(trace) for trace in top_tail_traces]
        if add_noise:
            top_tail_noise_steps = [trace.get('noise_step', 0) for trace in top_tail_traces]
            top_tail_vote_func_pre = lambda ans, wts: weighted_majority_vote_with_noise_penalty(ans, wts, noise_steps=top_tail_noise_steps, penalty_type=penalty_type_to_use, **pre_boost_args)
            top_tail_vote_func_post = lambda ans, wts: weighted_majority_vote_with_noise_penalty(ans, wts, noise_steps=top_tail_noise_steps, penalty_type=penalty_type_to_use, **post_boost_args)
        else:
            top_tail_vote_func_pre = lambda ans, wts: weighted_majority_vote(ans, wts, **pre_boost_args)
            top_tail_vote_func_post = lambda ans, wts: weighted_majority_vote(ans, wts, **post_boost_args)

        if any(c > 0 for c in top_tail_confidences):
            top_tail_answer_pre, top_tail_scores_pre = top_tail_vote_func_pre(top_tail_answers, top_tail_confidences)
            voting_results['top10_tail_filtered'] = {
                'answer': top_tail_answer_pre,
                'num_votes': len(top_tail_answers),
                'confidence': np.mean(top_tail_confidences),
                'vote_scores': top_tail_scores_pre 
            }
            top_tail_answer_post, top_tail_scores_post = top_tail_vote_func_post(top_tail_answers, top_tail_confidences)
            voting_results['top10_tail_filtered_boosted'] = {
                'answer': top_tail_answer_post,
                'num_votes': len(top_tail_answers),
                'confidence': np.mean(top_tail_confidences),
                'vote_scores': top_tail_scores_post 
            }
    
    # 7. Top 10% Bottom Window Confidence Filtered + Weighted
    top_bottom_traces = filter_top_confidence(valid_traces, 'bottom_window', 0.1)
    if top_bottom_traces:
        # ... (定义 top_bottom_vote_func_pre/post - 不变) ...
        top_bottom_answers = [trace['extracted_answer'] for trace in top_bottom_traces]
        top_bottom_confidences = [calculate_bottom_window_confidence(trace) for trace in top_bottom_traces]
        if add_noise:
            top_bottom_noise_steps = [trace.get('noise_step', 0) for trace in top_bottom_traces]
            top_bottom_vote_func_pre = lambda ans, wts: weighted_majority_vote_with_noise_penalty(ans, wts, noise_steps=top_bottom_noise_steps, penalty_type=penalty_type_to_use, **pre_boost_args)
            top_bottom_vote_func_post = lambda ans, wts: weighted_majority_vote_with_noise_penalty(ans, wts, noise_steps=top_bottom_noise_steps, penalty_type=penalty_type_to_use, **post_boost_args)
        else:
            top_bottom_vote_func_pre = lambda ans, wts: weighted_majority_vote(ans, wts, **pre_boost_args)
            top_bottom_vote_func_post = lambda ans, wts: weighted_majority_vote(ans, wts, **post_boost_args)
            
        if any(c > 0 for c in top_bottom_confidences):
            top_bottom_answer_pre, top_bottom_scores_pre = top_bottom_vote_func_pre(top_bottom_answers, top_bottom_confidences)
            voting_results['top10_bottom_window_filtered'] = {
                'answer': top_bottom_answer_pre,
                'num_votes': len(top_bottom_answers),
                'confidence': np.mean(top_bottom_confidences),
                'vote_scores': top_bottom_scores_pre 
            }
            top_bottom_answer_post, top_bottom_scores_post = top_bottom_vote_func_post(top_bottom_answers, top_bottom_confidences)
            voting_results['top10_bottom_window_filtered_boosted'] = {
                'answer': top_bottom_answer_post,
                'num_votes': len(top_bottom_answers),
                'confidence': np.mean(top_bottom_confidences),
                'vote_scores': top_bottom_scores_post 
            }

    # Fill in all missing keys (including SR/SCk placeholders)
    for key in all_method_keys:
        if key not in voting_results:
            voting_results[key] = {
                'answer': None,
                'num_votes': 0,
                'confidence': None,
                'vote_scores': {}
            }
            
    return voting_results


def add_diffusion_noise(image_tensor: torch.Tensor, noise_step: Union[int, torch.Tensor]) -> torch.Tensor:
    """
    Applies noise to the input image tensor (x_0) corresponding to a specific time step (t)
    in the forward process of a Denoising Diffusion Probabilistic Model (DDPM).

    This function simulates the non-iterative calculation of the noisy image x_t from the 
    original image x_0, using the reparameterization trick:
    q(x_t|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon.

    Args:
        image_tensor (torch.Tensor): The clean input image tensor (x_0).
        noise_step (Union[int, torch.Tensor]): The time step 't' in the diffusion schedule 
                                               to corrupt the image to. 
                                               The valid range for this step is [0, 999], 
                                               as the total number of steps (num_steps) is 1000. 
                                               t=0 is minimal noise; t=999 is maximal noise.

    Returns:
        torch.Tensor: The noisy image tensor (x_t) after applying noise up to 'noise_step'.
    """
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_step = int(noise_step) if not isinstance(noise_step, int) else noise_step
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step) 

    return image_tensor_cd


def prepare_noisy_samples(image_tensor: torch.Tensor, num_samples: int) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Generates a list of noisy image tensors, ranging from the original image (t=0)
    to maximally noisy (t=999), based on a Self-Consistency strategy.

    Args:
        image_tensor (torch.Tensor): The clean input image tensor (x_0).
        num_samples (int): The number of noisy samples to generate (excluding the clean one).

    Returns:
        Tuple[List[torch.Tensor], List[int]]:
            1. noisy_samples: A list of image tensors (size num_samples + 1), 
                              starting with the clean image.
            2. all_t_steps:   A corresponding list of noise steps (t) used for 
                              each image (size num_samples + 1), starting with 0.
    """
    # DDPM parameters
    num_steps = 1000
    max_step_index = num_steps - 1  # 999

    # Determine noise step indices for noisy samples (t > 0)
    noise_steps: List[int] = []

    for i in range(1, num_samples + 1):
        step_value = i * (max_step_index / num_samples)
        t_step = int(torch.round(torch.tensor(step_value)).item())
        # Ensure t_step is at least 1 and at most 999
        t_step = max(1, min(max_step_index, t_step))
        noise_steps.append(t_step)

    # Create complete step list starting with t=0 (clean image)
    all_t_steps: List[int] = [0] + noise_steps

    logger.debug("Calculated all noise steps (t): %s", all_t_steps)

    # Generate list of noisy images
    # First element is the clean image (t=0)
    noisy_samples: List[torch.Tensor] = [image_tensor]

    # Iterate through t > 0 steps to generate noisy images
    for t_step in noise_steps:
        # Note: image_tensor is x_0, add_diffusion_noise generates new noise internally
        noisy_image = add_diffusion_noise(image_tensor.clone(), t_step)
        noisy_samples.append(noisy_image)

    return noisy_samples, all_t_steps

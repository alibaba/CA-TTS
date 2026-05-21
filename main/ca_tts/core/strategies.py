# core/strategies.py
from .generation import BaseGenerator
from ..outputs import DeepThinkOutput # 假设 outputs.py 在上一级
from .algorithms import compute_voting_results
from typing import Dict, Any, List
import time
from tqdm import tqdm

def run_self_consistency(
    generator: BaseGenerator,
    inputs: Dict[str, Any],
    sampling_params: Dict[str, Any],
    num_samples: int = 16
) -> DeepThinkOutput:
    """
    运行自洽性 (DeepConf 离线模式)。
    生成 N 个样本并进行投票。
    """
    print(f"Running Self-Consistency with {num_samples} samples...")
    output = DeepThinkOutput()
    output.mode = "self_consistency"
    
    traces = []
    start_time = time.time()
    
    for i in range(num_samples):
        # 我们可以为每次采样设置不同的 seed
        sample_params = sampling_params.copy()
        sample_params['seed'] = int(time.time_ns()) + i # 简单 seed 示例
        
        trace = generator.generate_sample(inputs, sample_params)
        traces.append(trace)
        print(f"  Sample {i+1}/{num_samples} done. Answer: {trace['extracted_answer']}")

    output.generation_time = time.time() - start_time
    output.all_traces = traces
    output.total_traces_count = len(traces)
    output.total_tokens = sum(t['num_tokens'] for t in traces)

    # --- Voting ---
    print("Computing voting results...")
    voting_start = time.time()
    output.voting_results = compute_voting_results(traces)

    # Change final_answer to a dictionary containing all voting results
    # 1. Initialize final_answer as an empty dictionary
    output.final_answer = {}

    # 2. Iterate through all keys in voting_results (e.g., 'majority', 'weighted_confidence')
    for voting_method, results in output.voting_results.items():
        # 3. Check if the 'answer' key exists in the results for this method
        if 'answer' in results:
            # 4. Store the answer string in the new dictionary
            #    Key = Voting method name (e.g., 'majority')
            #    Value = Answer (e.g., 'A')
            output.final_answer[voting_method] = results['answer']
        else:
            # Record the voting method even if no 'answer' was found
            output.final_answer[voting_method] = None
        
    output.processing_time = time.time() - voting_start
    output.total_time = time.time() - start_time
    
    output.print_summary()
    return output


def run_self_correction(
    generator: BaseGenerator,
    inputs: Dict[str, Any],
    sampling_params: Dict[str, Any],
    correction_prompt_template: str = "Question: {question}\nInitial Answer: {answer}\nCritique: Please review this answer. Is it correct? If not, provide the corrected answer in the format 'Final Answer: \\boxed{{...}}'."
) -> DeepThinkOutput:
    """
    运行自我修正流程。
    1. 生成初始答案。
    2. 使用修正提示词生成最终答案。
    """
    print("Running Self-Correction...")
    output = DeepThinkOutput()
    output.mode = "self_correction"
    start_time = time.time()

    # --- 1. 生成初始答案 ---
    print("Generating initial answer...")
    initial_trace = generator.generate_sample(inputs, sampling_params)
    initial_answer = initial_trace.get("generated_text", "") # 获取完整生成
    
    # --- 2. 构建修正提示词 ---
    # 假设原始 prompt 在 inputs["text_prompt"]
    original_question = inputs.get("text_prompt", "[Question content not found]")
    correction_prompt = correction_prompt_template.format(question=original_question, answer=initial_answer)
    
    # 为修正提示词准备新的输入
    # 注意：这需要 generator.processor.tokenizer
    tokenizer = generator.processor.tokenizer if hasattr(generator.processor, 'tokenizer') else generator.processor
    
    correction_inputs_tokenized = tokenizer(
        correction_prompt, 
        return_tensors='pt'
    )
    
    # 如果是 LVLM，我们必须保留 'pixel_values'
    if 'pixel_values' in inputs:
        correction_inputs_tokenized['pixel_values'] = inputs['pixel_values']
    
    # --- 3. 生成修正答案 ---
    print("Generating corrected answer...")
    final_trace = generator.generate_sample(correction_inputs_tokenized, sampling_params)
    
    output.all_traces = [initial_trace, final_trace] # 存储两个步骤的 trace
    output.final_answer = final_trace.get('extracted_answer', 'No answer found after correction.')
    output.total_time = time.time() - start_time
    
    print(f"Initial Answer: {initial_trace.get('extracted_answer')}")
    print(f"Final Answer: {output.final_answer}")
    
    return output

def run_self_reflection(
    generator: BaseGenerator,
    inputs: Dict[str, Any],
    sampling_params: Dict[str, Any],
    reflection_prompt_template: str = "Question: {question}\nProposed Answer: {answer}\n\nPlease reflect on the reasoning process for this answer. Identify any potential errors, logical fallacies, or missing steps. Do not provide a new answer, just provide your reflection.",
    final_answer_prompt_template: str = "Question: {question}\nInitial Answer: {answer}\nReflection/Critique: {reflection}\n\nBased on the reflection, provide a new, final, and improved answer. Put the final answer in the format '\\boxed{{...}}'."
) -> DeepThinkOutput:
    """
    运行自我反思流程 (三步)。
    1. 生成初始答案。
    2. 生成对初始答案的“反思” (Critique)。
    3. 结合问题、答案和反思，生成最终答案。
    """
    print("Running Self-Reflection (3-step)...")
    output = DeepThinkOutput()
    output.mode = "self_reflection"
    start_time = time.time()

    # --- 1. 生成初始答案 ---
    print("Generating initial answer...")
    initial_trace = generator.generate_sample(inputs, sampling_params)
    initial_answer_text = initial_trace.get("generated_text", "") # 获取完整生成
    
    # 假设原始 prompt 在 inputs["text_prompt"]
    original_question = inputs.get("text_prompt", "[Question content not found]")

    # 获取 tokenizer
    tokenizer = generator.processor.tokenizer if hasattr(generator.processor, 'tokenizer') else generator.processor

    # --- 2. 生成反思 (Reflection) ---
    print("Generating reflection...")
    reflection_prompt = reflection_prompt_template.format(question=original_question, answer=initial_answer_text)
    
    reflection_inputs = tokenizer(
        reflection_prompt, 
        return_tensors='pt'
    )
    if 'pixel_values' in inputs:
        reflection_inputs['pixel_values'] = inputs['pixel_values'] # 保留图像

    reflection_trace = generator.generate_sample(reflection_inputs, sampling_params)
    reflection_text = reflection_trace.get("generated_text", "[Reflection failed]")

    # --- 3. 生成最终答案 ---
    print("Generating final answer based on reflection...")
    final_prompt = final_answer_prompt_template.format(
        question=original_question, 
        answer=initial_answer_text,
        reflection=reflection_text
    )
    
    final_inputs = tokenizer(
        final_prompt, 
        return_tensors='pt'
    )
    if 'pixel_values' in inputs:
        final_inputs['pixel_values'] = inputs['pixel_values'] # 保留图像

    final_trace = generator.generate_sample(final_inputs, sampling_params)

    # --- 总结 ---
    output.all_traces = [initial_trace, reflection_trace, final_trace] # 存储所有步骤
    output.final_answer = final_trace.get('extracted_answer', 'No answer found after reflection.')
    output.total_time = time.time() - start_time
    
    print("\n--- Reflection Summary ---")
    print(f"Initial Answer: {initial_trace.get('extracted_answer')}")
    print(f"Reflection: {reflection_text}")
    print(f"Final Answer: {output.final_answer}")
    
    return output
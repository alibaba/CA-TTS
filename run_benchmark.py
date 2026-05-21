import os
import json
import argparse
from tqdm import tqdm
from typing import Dict, Any, List, Union, Optional
import sys
import multiprocessing
import numpy as np
import time

from PIL import Image
# Important: Do not import any libraries that might indirectly import torch / initialize CUDA at module top level (e.g., modelscope).
# In multiprocessing spawn mode, child processes will import this module first; if CUDA is already initialized,
# setting CUDA_VISIBLE_DEVICES later in the worker will not take effect, causing all processes to run on physical GPU 0.
# Therefore, delay such imports to inside worker_main, after setting CUDA_VISIBLE_DEVICES.

PROMPT_MATH_VISION = """Please solve the problem step by step and put your answer in one '\\boxed{{}}'. If it is a multiple choice question, only one letter is allowed in the '\\boxed{{}}'.
{Question}"""

PROMPT_MATH_VISION_ZERO = """{Question}"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Self-Consistency on a VL model (Qwen-VL example) with multiprocessing")
    
    parser.add_argument(
        "--task_name", 
        type=str, 
        required=True, 
        help="The task name to specify the dataset and corresponding prompt template"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to the output JSON Lines (.jsonl) file to save the final answers"
    )
    parser.add_argument(
        "--noise_log_path", 
        type=str, 
        default=None, 
        help="Path to the output JSONL file to save noise-step-wise confidences and answers."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Model ID from Hugging Face Hub"
    )
    
    parser.add_argument(
        "--model_family",
        type=str,
        default="qwen",
        choices=["qwen", "internvl"],
        help="Model family to run: qwen (HF) or internvl (modelscope)"
    )
    
    parser.add_argument(
        "--gpu_ids", 
        type=str, 
        required=True, 
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--processes_per_gpu", 
        type=int, 
        default=1, 
        help="Number of processes to launch per GPU"
    )

    parser.add_argument(
        "--add_noise",
        type=str,
        default='False', 
        choices=['True', 'False'], 
        help="If 'True', run self-consistency on noisy image inputs instead of text sampling."
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=8, 
        help="Number of samples for self-consistency (default: 8)"
    )

    parser.add_argument(
        "--verify_mode",
        type=str,
        default='False', 
        choices=['True', 'False'], 
        help="If 'True', add a gemini verifier."
    )
    
    parser.add_argument(
        "--verify_tau", 
        type=float, 
        default=0.0, 
        help="The boost strength (0.0 to 1.0) of verifier"
    )

    parser.add_argument(
        "--verify_expert_model",
        type=str,
        default="gpt-5-mini-0807-global",
        help="The expert model to use for verification (e.g., 'gpt-5-mini...', 'GRPO-Qwen', 'Qwen-Expert')."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the dataset JSON file. If not provided, must be specified in task config."
    )
    
    args = parser.parse_args()
    
    if isinstance(args.add_noise, str):
        args.add_noise = args.add_noise.lower() == 'true'
    if isinstance(args.verify_mode, str):
        args.verify_mode = args.verify_mode.lower() == 'true'

    return args


def load_input_json(input_json_path: str):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def worker_main(
    worker_id: int,
    gpu_id: str,
    samples_chunk: List[Dict[str, Any]],
    results_queue: multiprocessing.Queue,
    args: argparse.Namespace,
    sampling_params: Dict[str, Any]
):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Local imports
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
    # Delayed import of modelscope (which may internally import torch and trigger CUDA initialization)
    from modelscope import AutoModel, AutoTokenizer
    
    from ca_tts.core.generation import TransformersGenerator, InternVLGenerator
    from ca_tts.outputs import DeepThinkOutput
    from ca_tts.core.algorithms import (
        add_diffusion_noise, 
        prepare_noisy_samples, 
        load_image, 
        prepare_inputs, 
        prepare_noise_inputs,
        compute_voting_results,
        apply_external_boost_and_vote,
        prepare_internvl_inputs
    )
    from ca_tts.core.verifier import verify, VerificationMode

    torch.cuda.set_device(0)
    print("add noise: " + str(args.add_noise))
    # Note: cuda:0 here refers to the 0th device in the "current process's visible device list", which typically maps to the physical gpu_id
    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        device_name = "Unknown"
    print(f"Worker {worker_id} started. CUDA_VISIBLE_DEVICES={gpu_id}, local cuda:0={device_name}. Processing {len(samples_chunk)} samples.")

    generator = None
    processor = None
    expert_generator = None
    expert_processor = None

    try:
        # 1. Load main model
        print(f"Worker {worker_id}: Loading main model from {args.model} (family={args.model_family})...")
        if args.model_family == "internvl":
            model = AutoModel.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True
            ).eval().cuda()
            processor = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
            generator = InternVLGenerator(model=model, processor_or_tokenizer=processor)
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model, 
                dtype=torch.bfloat16, 
                device_map="cuda:0",
                attn_implementation="flash_attention_2"
            )
            # model = Qwen3VLForConditionalGeneration.from_pretrained(
            #     args.model,
            #     dtype="auto", 
            #     device_map="auto",
            #     # attn_implementation=args.attn_implementation
            # )
            processor = AutoProcessor.from_pretrained(args.model)
            generator = TransformersGenerator(model=model, processor_or_tokenizer=processor)
        print(f"Worker {worker_id}: Main model loaded.")

        # 2. Load or configure Expert Verifier model based on args

        # Logic 1: "GRPO" -> use main model
        if args.verify_expert_model.startswith("GRPO"):
            print(f"Worker {worker_id}: Verification model '{args.verify_expert_model}' uses main generator.")
            expert_generator = generator
            expert_processor = processor

        # Logic 2: "Qwen" -> load new Qwen model
        elif args.verify_expert_model.startswith("Qwen"):
            expert_model_path = args.model  # Use the same model as expert by default
            print(f"Worker {worker_id}: Loading EXPERT verifier model from {expert_model_path}...")
            try:
                # Note: This will load a second model on the same GPU (cuda:0)
                expert_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    expert_model_path, 
                    dtype=torch.bfloat16, 
                    device_map="cuda:0",
                    attn_implementation="flash_attention_2"
                )
                expert_processor = AutoProcessor.from_pretrained(expert_model_path)
                expert_generator = TransformersGenerator(model=expert_model, processor_or_tokenizer=expert_processor)
                print(f"Worker {worker_id}: Expert verifier model loaded.")
            except Exception as e:
                print(f"Worker {worker_id} FAILED to initialize expert model: {e}")
                raise e  # Strict failure

        # Logic 3: "gpt-..." or other -> expert_generator remains None, will use API
        else:
             print(f"Worker {worker_id}: Verification model '{args.verify_expert_model}' (API) will be used.")

    except Exception as e:
        print(f"Worker {worker_id} FAILED during model initialization: {e}")
        results_queue.put(None)
        return  # Exit worker

    def run_evaluation_sample(
        generator: TransformersGenerator, 
        processor: AutoProcessor, 
        image_path: str, 
        prompt_text: str, 
        raw_question: str,
        num_samples: int, 
        sampling_params: Dict[str, Any],
        gt_answer: Optional[str],
        sample_id: str,
        noise_log_path: Optional[str]
    ) -> DeepThinkOutput:
        
        output = DeepThinkOutput()
        overall_start_time = time.time()
        
        BOOST_HYPERPARAMETER = 0.5
        boost_value = BOOST_HYPERPARAMETER * num_samples
        print(f"External boost value set to: {boost_value} (0.5 * {num_samples})")

        use_noise = args.add_noise and args.model_family != "internvl"
        if args.add_noise and not use_noise:
            print("InternVL pipeline does not support noisy image generation. Falling back to text self-consistency.")

        # --- 1. Generation Phase ---
        if use_noise:
            print(f"Running Noise-based Self-Consistency with {num_samples} samples...")
            output.mode = "noise_self_consistency"
            traces = []
            gen_start_time = time.time()

            base_inputs = prepare_inputs(processor, image_path, prompt_text)
            if 'pixel_values' not in base_inputs:
                raise ValueError("Could not find 'pixel_values' in processor output.")
            
            clean_image_tensor = base_inputs['pixel_values'].to(generator.model.device)

            num_noisy_samples_to_gen = num_samples - 1
            (noisy_image_tensors, all_t_steps) = prepare_noisy_samples(
                clean_image_tensor, 
                num_noisy_samples_to_gen
            )

            if len(noisy_image_tensors) != num_samples:
                print(f"Warning: Expected {num_samples} total samples, but prepare_noisy_samples returned {len(noisy_image_tensors)}")

            for i in range(len(noisy_image_tensors)):
                img_tensor = noisy_image_tensors[i]
                t_step_for_this_sample = all_t_steps[i]

                sample_inputs = base_inputs.copy()
                sample_inputs['pixel_values'] = img_tensor
                sample_inputs = {k: v.to(generator.model.device) if isinstance(v, torch.Tensor) else v for k, v in sample_inputs.items()}

                gen_params = sampling_params.copy()
                gen_params['seed'] = int(time.time_ns()) + i

                trace = generator.generate_sample(sample_inputs, gen_params)
                trace['noise_step'] = t_step_for_this_sample
                traces.append(trace)
                print(f"   Noise Sample {i+1}/{num_samples} (t={t_step_for_this_sample}) done. Answer: {trace['extracted_answer']}")
            
            output.generation_time = time.time() - gen_start_time
            output.all_traces = traces

        else:
            print(f"Running Text-based Self-Consistency with {num_samples} samples...")
            output.mode = "self_consistency"
            traces = []
            gen_start_time = time.time()

            if isinstance(generator, InternVLGenerator):
                inputs = prepare_internvl_inputs(
                    tokenizer=processor,
                    image_path=image_path,
                    prompt_text=prompt_text
                )
            else:
                inputs = prepare_inputs(processor, image_path, prompt_text)
                inputs = {k: v.to(generator.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            traces = generator.generate_sample_batch(
                inputs, 
                sampling_params, 
                num_samples_per_input=num_samples
            )
            
            print(f"   Batch of {num_samples} generated. Example: {traces[0]['extracted_answer']}")

            output.generation_time = time.time() - gen_start_time
            output.all_traces = traces

        output.total_traces_count = len(output.all_traces)
        output.total_tokens = sum(t['num_tokens'] for t in output.all_traces)

        # --- 2. Verification Phase ---
        verifier_output, options_list = [], []
        
        if args.verify_mode == True:
            all_answers = [t['extracted_answer'] for t in output.all_traces]
            options_list = sorted(list(set(ans for ans in all_answers if ans and ans.strip())))

            if not options_list:
                print("Worker: No valid answers extracted from traces. Skipping verifier.")
                verifier_output, options_list = [], []
            else:
                print(f"Worker: Found {len(options_list)} unique answers for verifier: {options_list}")
                
                if "allowed in the '\\boxed{{}}'." in prompt_text:
                     question_for_verifier_sc = prompt_text.split("allowed in the '\\boxed{{}}'.")[-1].strip()
                else:
                     question_for_verifier_sc = raw_question

                image_pil = load_image(image_path=image_path)

                MAX_RETRIES = 3
                verify_result = None

                for attempt in range(MAX_RETRIES):
                    print(f"Worker: Calling verifier (Attempt {attempt + 1}/{MAX_RETRIES})...")
                    try:
                        verify_result = verify(
                            image=image_pil,
                            question=question_for_verifier_sc,
                            verification_mode=VerificationMode.SELF_CONSISTENCY,
                            option_list=options_list,
                            model_answer=None,

                            # --- New/modified parameters ---
                            expert_model_name=args.verify_expert_model,

                            # Pass the correct local model selected based on args
                            # (if using API, these will be None)
                            local_generator=expert_generator,
                            local_processor=expert_processor,

                            image_path=image_path,
                            sampling_params=sampling_params
                        )
                        
                        if isinstance(verify_result, tuple) and len(verify_result) == 2:
                            verifier_output, _ = verify_result
                            print(f"Worker: Verifier success. Probs: {verifier_output}")
                            break
                        else:
                            print(f"Worker WARNING: Verifier attempt {attempt + 1} failed (not a tuple).")
                            print(f"Verifier Response (Error Msg): {verify_result}")
                            verify_result = None
                    
                    except Exception as e:
                        print(f"Worker ERROR: Verifier attempt {attempt + 1} raised an exception: {e}")
                        verify_result = None

                if verify_result is None:
                    print(f"Worker CRITICAL: Verifier failed all {MAX_RETRIES} attempts.")
                    verifier_output = [0.0] * len(options_list)
                    print(f"Setting verifier output to zeros: {verifier_output}")

        # --- 3. Base Voting Phase ---
        print("Computing *base* voting results...")
        voting_start_time = time.time()
        
        meta_info_for_log = {"id": sample_id, "gt_answer": gt_answer}
        
        output.voting_results = compute_voting_results(
            output.all_traces, 
            add_noise=use_noise,
            output_path=noise_log_path,
            meta_info=meta_info_for_log,
            verify_mode=args.verify_mode, 
            tau=args.verify_tau, 
            verifier_output=verifier_output, 
            options_list=options_list
        )

        if args.verify_mode == True:
            # --- 4. Self-Reflection (SR) Enhanced Voting ---
            print(f"\n--- Running SR Enhanced Voting (Boost: {boost_value}) ---")
            
            base_maj_key = 'majority_boosted' if args.verify_mode else 'majority'
            base_mean_key = 'mean_confidence_weighted_boosted' if args.verify_mode else 'mean_confidence_weighted'
            
            base_maj_result = output.voting_results.get(base_maj_key)
            base_mean_result = output.voting_results.get(base_mean_key)

            base_answer_for_sr = base_maj_result['answer'] if base_maj_result else None
            critique = "N/A (No base answer for SR)"
            sr_answer = None

            # if base_answer_for_sr:
            #     try:
            #         #  -----------------self-reflection-----------------
            #         print(f"SR: Getting critique for answer: {base_answer_for_sr}")
            #         critique = generator.generate_self_reflection(
            #             image_path=image_path,
            #             question=raw_question,
            #             voted_answer=base_answer_for_sr,

            #             # Pass verifier parameters
            #             expert_model_name=args.verify_expert_model,
            #             local_generator=expert_generator,  # (defined at the top of worker_main)
            #             local_processor=expert_processor,  # (same as above)
            #             sampling_params=sampling_params
            #         )

            #         # (function returns "Error: ..." on failure)
            #         if isinstance(critique, str) and critique.startswith("Error:"):
            #             raise Exception(critique)

            #         print(f"SR: Critique received: {critique[:100]}...")
            #     except Exception as e:
            #         critique = f"Error generating critique: {e}"
            #         print(f"SR ERROR: {critique}")

            #     try:
            #         sr_prompt_text = (
            #             f"{prompt_text}\n\n"
            #             f"--- Previous Analysis ---\n"
            #             f"According to the previous question '{raw_question}'"
            #             f"A previous attempt resulted in the answer '{base_answer_for_sr}'.\n"
            #             f"Here is a critique of that attempt: {critique}\n\n"
            #             f"--- New Task ---\n"
            #             f"Based on the original question and this critique, please re-evaluate and provide the final, correct answer. "
            #             f"Put your final answer in one '\\boxed{{}}'."
            #         )

            #         print(f"SR: Re-prompting model with critique...")
            #         sr_inputs = prepare_inputs(processor, image_path, sr_prompt_text)

            #         sr_sampling_params = sampling_params.copy()
            #         sr_sampling_params['do_sample'] = False

            #         sr_trace = generator.generate_sample(sr_inputs, sr_sampling_params)
            #         sr_answer = sr_trace['extracted_answer']
            #         print(f"SR: Model's re-evaluated (SR) answer: {sr_answer}")

            #     except Exception as e:
            #         print(f"SR ERROR: Re-prompting failed: {e}")
            #         sr_answer = None
            
            if base_maj_result:
                sr_maj_ans, sr_maj_scores = apply_external_boost_and_vote(
                    vote_scores=base_maj_result['vote_scores'],
                    boost_answer=sr_answer,
                    boost_value=boost_value
                )
                output.voting_results['sc_plus_sr_majority_boosted'] = {
                    'answer': sr_maj_ans,
                    'num_votes': len(output.all_traces),
                    'confidence': None,
                    'vote_scores': sr_maj_scores,
                    'critique': critique,
                    'sr_answer': sr_answer
                }
            
            if base_mean_result:
                sr_mean_ans, sr_mean_scores = apply_external_boost_and_vote(
                    vote_scores=base_mean_result['vote_scores'],
                    boost_answer=sr_answer,
                    boost_value=boost_value
                )
                output.voting_results['sc_plus_sr_mean_confidence_weighted_boosted'] = {
                    'answer': sr_mean_ans,
                    'num_votes': len(output.all_traces),
                    'confidence': base_mean_result.get('confidence'),
                    'vote_scores': sr_mean_scores,
                    'critique': critique,
                    'sr_answer': sr_answer
                }

            # --- 5. Self-Check (SCk) Enhanced Voting ---
            print(f"\n--- Running SCk Enhanced Voting (Boost: {boost_value}) ---")

            sck_answer = None

            # -----------------self-check-----------------
            try:
                sck_result = generator.self_check(
                    image_path=image_path,
                    prompt_text=prompt_text,
                    sampling_params=sampling_params,
                    with_reference=False,
                    contrastive_alpha=0.1
                )

                if sck_result.get("status") == "success":
                    sck_answer = sck_result['original_trace'].get('extracted_answer')
                    print(f"SCk: Self-Check completed. Answer: {sck_answer}")
                else:
                    print(f"Worker WARNING: Self-Check failed or returned error: {sck_result.get('error')}")

            except Exception as e:
                print(f"Worker ERROR: Self-Check failed: {e}")
            
            sr_maj_result = output.voting_results.get('sc_plus_sr_majority_boosted')
            if sr_maj_result:
                sck_maj_ans, sck_maj_scores = apply_external_boost_and_vote(
                    vote_scores=sr_maj_result['vote_scores'],
                    boost_answer=sck_answer,
                    boost_value=boost_value
                )
                output.voting_results['sc_plus_sr_plus_sck_majority_boosted'] = {
                    'answer': sck_maj_ans,
                    'num_votes': len(output.all_traces),
                    'confidence': None,
                    'vote_scores': sck_maj_scores,
                    'sck_answer': sck_answer
                }
            
            sr_mean_result = output.voting_results.get('sc_plus_sr_mean_confidence_weighted_boosted')
            if sr_mean_result:
                sck_mean_ans, sck_mean_scores = apply_external_boost_and_vote(
                    vote_scores=sr_mean_result['vote_scores'],
                    boost_answer=sck_answer,
                    boost_value=boost_value
                )
                output.voting_results['sc_plus_sr_plus_sck_mean_confidence_weighted_boosted'] = {
                    'answer': sck_mean_ans,
                    'num_votes': len(output.all_traces),
                    'confidence': sr_mean_result.get('confidence'),
                    'vote_scores': sck_mean_scores,
                    'sck_answer': sck_answer
                }

        # --- 6. Final Answer Extraction ---
        print("\n--- Final Answer Extraction ---")
        output.final_answer = {}
        for voting_method, results in output.voting_results.items():
            if results:
                output.final_answer[voting_method] = results.get('answer', None)
            else:
                output.final_answer[voting_method] = None

        output.processing_time = time.time() - voting_start_time
        output.total_time = time.time() - overall_start_time

        output.print_summary()
        return output

    def process_sample(
        generator: TransformersGenerator, 
        processor: AutoProcessor, 
        sample: Dict[str, Any], 
        prompt_text: str, 
        num_samples: int, 
        sampling_params: Dict[str, Any],
        args: argparse.Namespace
    ) -> Dict[str, Any]:
    
        image_path = sample['image'][0]
        
        gt_answer = sample["answer"]
        sample_id = sample["id"]
        
        deepthink_output = run_evaluation_sample(
            generator=generator,
            processor=processor,
            image_path=image_path,
            prompt_text=prompt_text,
            raw_question=sample["problem"],
            num_samples=num_samples,
            sampling_params=sampling_params,
            gt_answer=gt_answer,
            sample_id=sample_id,
            noise_log_path=args.noise_log_path
        )
        
        result = {
            "id": sample["id"],
            "question": sample["problem"],
            "image": sample["image"][0],
            "final_answer": deepthink_output.final_answer,
            "gt": sample["answer"],
            "critique_used_for_sr": deepthink_output.voting_results
                .get('sc_plus_sr_majority_boosted', {})
                .get('critique', 'N/A')
        }
        return result

    try:
        
        if args.task_name == "math-vision":
            QUESTION_PROMPT = PROMPT_MATH_VISION_ZERO
        elif args.task_name == "math-vista":
            QUESTION_PROMPT = PROMPT_MATH_VISION_ZERO
        elif args.task_name == "mm-vet":
            QUESTION_PROMPT = PROMPT_MATH_VISION
        elif args.task_name == "mmstar":
            QUESTION_PROMPT = PROMPT_MATH_VISION
        elif args.task_name == "mmmu":
            QUESTION_PROMPT = PROMPT_MATH_VISION
        else:
            raise ValueError(f"Unknown task_name in worker: {args.task_name}")

        for sample in tqdm(samples_chunk, desc=f"Worker {worker_id} (GPU {gpu_id})", position=worker_id, ncols=100):
            prompt_text = QUESTION_PROMPT.format(Question=sample['problem'])
            
            try:
                result = process_sample(
                    generator, 
                    processor, 
                    sample, 
                    prompt_text, 
                    num_samples=args.num_samples,
                    sampling_params=sampling_params,
                    args=args
                )
                results_queue.put(result)
                
            except Exception as e:
                print(f"Worker {worker_id} failed on sample {sample.get('id')}: {e}")
                error_result = {
                    "id": sample.get('id'),
                    "error": str(e)
                }
                results_queue.put(error_result)

        results_queue.put(None)
        print(f"Worker {worker_id} finished and sent 'None' sentinel.")

    except Exception as e:
        print(f"Worker {worker_id} FAILED to initialize: {e}")
        results_queue.put(None)

def main():
    args = parse_arguments()

    if args.dataset_path:
        input_json_path = args.dataset_path
    else:
        raise ValueError(
            f"No dataset path provided. Please specify --dataset_path, e.g.:\n"
            f"  --dataset_path data/MathVision/testmini_convert.json"
        )

    input_samples = load_input_json(input_json_path)

    gpu_ids = args.gpu_ids.split(',')
    worker_gpu_map = []
    for gpu_id in gpu_ids:
        worker_gpu_map.extend([gpu_id] * args.processes_per_gpu)

    total_workers = len(worker_gpu_map)
    if total_workers == 0:
        print("Error: No workers to launch. Check --gpu_ids and --processes_per_gpu.")
        return

    print(f"Total GPUs specified: {len(gpu_ids)}")
    print(f"Processes per GPU: {args.processes_per_gpu}")
    print(f"Total workers to launch: {total_workers}")
    print(f"Worker-to-GPU mapping: {worker_gpu_map}")

    sample_chunks = np.array_split(input_samples, total_workers)

    sampling_params = {
        'do_sample': True,
        'temperature': 1.0,
        'top_p': 1.0,
        'top_k': 40,
        'max_new_tokens': 8192,
        'confidence_window_size': 50
    }

    results_queue = multiprocessing.Queue()
    processes = []

    for i in range(total_workers):
        gpu_id_for_worker = worker_gpu_map[i]
        chunk_for_worker = sample_chunks[i].tolist()

        p = multiprocessing.Process(
            target=worker_main,
            args=(
                i,
                gpu_id_for_worker,
                chunk_for_worker,
                results_queue,
                args,
                sampling_params
            )
        )
        processes.append(p)
        p.start()

    print(f"Main process is listening for results... Writing to {args.output}")
    workers_finished_count = 0
    total_samples_processed = 0

    try:
        with open(args.output, 'w', encoding='utf-8') as f_out:
            while workers_finished_count < total_workers:
                result = results_queue.get()

                if result is None:
                    workers_finished_count += 1
                    print(f"A worker has finished. {workers_finished_count}/{total_workers} workers complete.")
                else:
                    json.dump(result, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    total_samples_processed += 1

                    if total_samples_processed % 50 == 0:
                        print(f"Main process has written {total_samples_processed} samples so far...")

    except Exception as e:
        print(f"CRITICAL ERROR during result collection and writing: {e}")
        print("Attempting to terminate worker processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()

    print("All workers have signaled completion. Waiting for processes to join...")
    for p in processes:
        p.join()

    print(f"Processing completed. All {total_samples_processed} results saved line-by-line to: {args.output}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
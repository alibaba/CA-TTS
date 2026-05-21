"""Core generation classes for model inference."""
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List, Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoProcessor, GenerationConfig

from .algorithms import compute_token_confidence, calculate_bottom_window_confidence, extract_answer, prepare_noisy_samples, load_image, prepare_inputs, prepare_noise_inputs, create_noisy_image
from .verifier import verify, VerificationMode

logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    """Abstract base class for generators."""

    def __init__(self, model, processor_or_tokenizer, device: str = 'cuda'):
        # When using device_map="auto", the model is already on its devices.
        # Calling model.to(device) here will break accelerate's hooks.
        # We simply store the model as-is.
        self.model = model
        self.processor = processor_or_tokenizer

        # Get the device from the model itself (e.g., 'cuda:0' or 'cpu')
        # where the inputs need to be sent.
        self.device = self.model.device


    @abstractmethod
    def generate_sample(self, inputs: Dict[str, Any], sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a single sample and return a trace containing confidence scores.
        """
        pass

class TransformersGenerator(BaseGenerator):
    """
    Generator using Hugging Face Transformers.
    Supports multimodal (LVLM) and text-only (LLM) models.
    """
    def __init__(self, model: PreTrainedModel, processor_or_tokenizer: Union[PreTrainedTokenizer, AutoProcessor], device: str = 'cuda'):
        super().__init__(model, processor_or_tokenizer, device)

    @torch.inference_mode()
    def generate_sample(self, inputs: Dict[str, Any], sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a single sample and return a trace containing logits_scores.
        """
        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
        
        gen_params = sampling_params.copy()

        window_size = gen_params.pop('confidence_window_size', 20)
        seed = gen_params.pop('seed', None)

        # Ensure we can always get scores
        gen_params['output_scores'] = True
        gen_params['return_dict_in_generate'] = True

        if seed is not None and gen_params.get('do_sample', False):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


        # 1. Generate
        outputs = self.model.generate(**inputs_on_device, **gen_params)

        # 2. Decode
        input_ids_len = inputs_on_device['input_ids'].shape[1]
        generated_ids = outputs.sequences
        generated_token_ids = generated_ids[:, input_ids_len:]

        full_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_text = self.processor.batch_decode(generated_token_ids, skip_special_tokens=True)[0]

        # 3. Compute confidence
        token_confs = compute_token_confidence(
            outputs.scores,
        )
        min_conf = calculate_bottom_window_confidence(token_confs, window_size)
        avg_conf = np.mean(token_confs) if token_confs else 0.0

        # 4. Extract answer
        answer = extract_answer(generated_text)

        # 5. Build trace
        trace = {
            "full_text": full_text,
            "generated_text": generated_text,
            "extracted_answer": answer,
            "token_ids": generated_token_ids[0].cpu().tolist(),
            "num_tokens": len(generated_token_ids[0]),
            'confs': token_confs,
            "min_conf": min_conf,
            "avg_conf": avg_conf,

            # Store logits (tuple, each element shape [batch, vocab_size])
            # Transfer to CPU to save GPU memory
            "logits_scores": tuple(s.cpu() for s in outputs.scores)
        }
        
        return trace
    
    @torch.inference_mode()
    def generate_sample_batch(
        self, 
        inputs: Dict[str, Any], 
        sampling_params: Dict[str, Any],
        num_samples_per_input: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple samples at once and return a list of traces.

        This function handles two cases:
        1. (Text self-consistency) input_batch_size=1, num_samples_per_input > 1:
           - Use `num_return_sequences` to generate N samples from a single input.
        2. (Noise self-consistency) input_batch_size=N, num_samples_per_input = 1:
           - Generate 1 sample for each of N inputs in parallel.
        """

        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}

        gen_params = sampling_params.copy()

        # Extract custom parameters
        window_size = gen_params.pop('confidence_window_size', 20)
        # 'seed' not handled here, randomness in batch is handled internally by generate
        gen_params.pop('seed', None)

        gen_params['output_scores'] = True
        gen_params['return_dict_in_generate'] = True

        # 1. Generation

        # Determine input batch size from 'input_ids'
        input_batch_size = inputs_on_device['input_ids'].shape[0]

        if input_batch_size == 1 and num_samples_per_input > 1:
            # Case 1: Text self-consistency (1 input -> N outputs)
            gen_params['num_return_sequences'] = num_samples_per_input
            gen_params['do_sample'] = True  # Must be True to take effect
            total_outputs = num_samples_per_input

        elif input_batch_size > 1 and num_samples_per_input == 1:
            # Case 2: Noise self-consistency (N inputs -> N outputs)
            # (already batched)
            total_outputs = input_batch_size

        elif input_batch_size == 1 and num_samples_per_input == 1:
            # Base case: (1 input -> 1 output)
            total_outputs = 1

        else:
            raise ValueError(
                f"Invalid batch combination: input_batch_size={input_batch_size} "
                f"and num_samples_per_input={num_samples_per_input}. "
                "This function only accepts (N, 1) or (1, N) combinations."
            )

        # Run model generation
        # outputs.sequences.shape will be [total_outputs, seq_len]
        # outputs.scores is T tuple, each element [total_outputs, vocab_size]
        outputs = self.model.generate(**inputs_on_device, **gen_params)

        # 2. Batch decode

        # 'outputs.sequences' shape: [total_outputs, seq_len]
        input_ids_len = inputs_on_device['input_ids'].shape[1]

        # [total_outputs, generated_seq_len]
        generated_token_ids_batch = outputs.sequences[:, input_ids_len:]

        # [total_outputs] (list of strings)
        full_texts_batch = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        generated_texts_batch = self.processor.batch_decode(generated_token_ids_batch, skip_special_tokens=True)

        all_traces = []

        # 3-5. Loop and build each trace

        for i in range(total_outputs):

            # 3. Compute confidence for sample i

            # Extract scores for the i-th sample
            # scores_tuple is [total_outputs, vocab_size]
            # scores_for_sample_i is T tuple, each element [vocab_size]
            scores_for_sample_i = tuple(scores_tuple[i].cpu() for scores_tuple in outputs.scores)
            
            token_confs = compute_token_confidence(
                scores_for_sample_i,
            )
            min_conf = calculate_bottom_window_confidence(token_confs, window_size)
            avg_conf = np.mean(token_confs) if token_confs else 0.0

            # 4. Extract answer for sample i
            generated_text = generated_texts_batch[i]
            answer = extract_answer(generated_text)

            # 5. Build trace for sample i
            generated_token_ids_sample = generated_token_ids_batch[i]

            trace = {
                "full_text": full_texts_batch[i],
                "generated_text": generated_text,
                "extracted_answer": answer,
                "token_ids": generated_token_ids_sample.cpu().tolist(),
                "num_tokens": len(generated_token_ids_sample),
                'confs': token_confs,
                "min_conf": min_conf,
                "avg_conf": avg_conf,

                # Store logits for the i-th sample (already .cpu() above)
                "logits_scores": scores_for_sample_i
            }
            all_traces.append(trace)

        return all_traces

    def _get_logits_from_trace(self, trace: Dict[str, Any]) -> torch.Tensor:
        """
        Internal helper function: extract and stack logits from trace.
        Returns (num_tokens, vocab_size) tensor, assuming batch_size=1.
        """
        if "logits_scores" not in trace or not trace["logits_scores"]:
            raise ValueError("Trace object does not contain 'logits_scores'.")

        # Stack tensors in tuple: (num_tokens, batch_size, vocab_size)
        try:
            logits_tensor = torch.stack(trace["logits_scores"], dim=0)
        except RuntimeError as e:
            # If generation is empty (0 tokens), scores might be empty
            logger.warning("Could not stack logits: %s. Returning empty tensor.", e)
            return torch.empty(0, 0, device='cpu')


        # Assuming batch_size=1, squeeze dimension
        if logits_tensor.shape[1] != 1:
             logger.warning("Batch size > 1 (%d).", logits_tensor.shape[1])

        # Return [num_tokens, vocab_size]
        return logits_tensor.squeeze(1).to(self.device)  # Move back to self.device for computation

    @torch.inference_mode()
    def self_check(
        self,
        image_path: str,
        prompt_text: str,
        sampling_params: Dict[str, Any],
        with_reference: bool = False,
        original_trace: Dict[str, Any] = None,
        contrastive_alpha: float = 0.5,
        attn_layer: int = 7
    ) -> Dict[str, Any]:
        """
        Perform visual contrastive decoding (self-check).
        """

        # 1. Get original logits
        if with_reference:
            if original_trace is None:
                raise ValueError("`original_trace` must be provided when `with_reference` is True.")
            logger.info("Using provided reference trace.")
        else:
            logger.info("Generating reference trace for original image...")
            original_inputs = prepare_inputs(self.processor, image_path, prompt_text)
            original_trace = self.generate_sample(original_inputs, sampling_params)

        try:
            original_logits = self._get_logits_from_trace(original_trace)
        except ValueError as e:
            logger.error("Error processing original trace: %s", e)
            return {"error": str(e), "original_trace": original_trace}

        # 2. Generate noisy image
        logger.info("Generating noisy image...")

        # Note: 'attn_model=self.model' is passed but will be ignored by create_noisy_image
        noisy_pil_image = create_noisy_image(
            image_path=image_path,
            attn_model=self.model,
            target_layer=attn_layer,
        )

        if noisy_pil_image is None:
             return {"error": "Failed to create noisy image.", "original_trace": original_trace}

        # 3. Get noisy logits
        logger.info("Generating trace for noisy image...")
        noisy_inputs = prepare_noise_inputs(self.processor, noisy_pil_image, prompt_text)
        time_before_trace = time.time()
        noisy_trace = self.generate_sample(noisy_inputs, sampling_params)
        time_after_trace = time.time()
        logger.debug("Noisy trace generation time: %.2f seconds", time_after_trace - time_before_trace)

        try:
            noisy_logits = self._get_logits_from_trace(noisy_trace)
        except ValueError as e:
            logger.error("Error processing noisy trace: %s", e)
            return {
                "error": str(e),
                "original_trace": original_trace,
                "noisy_trace": noisy_trace
            }

        # 4. Apply contrastive decoding
        logger.info("Applying contrastive decoding...")

        min_len = min(original_logits.shape[0], noisy_logits.shape[0])

        if min_len == 0:
            logger.error("One of the generations produced 0 tokens.")
            return {
                "error": "Zero-token generation.",
                "original_trace": original_trace,
                "noisy_trace": noisy_trace
            }

        original_logits_common = original_logits[:min_len, :]
        noisy_logits_common = noisy_logits[:min_len, :]

        # Apply "Adaptive Plausibility Constraints" (APC)
        # 1. Set beta (fixed at 0.1)
        cd_beta = 0.1

        # 2. Calculate cutoff (version 2, log-space)
        # 确保 beta 张量和 logits 在同一设备上
        beta_tensor = torch.tensor(cd_beta, device=original_logits_common.device)
        cutoff = torch.log(beta_tensor) + original_logits_common.max(dim=-1, keepdim=True).values

        # 3. Calculate contrastive differences (diffs)
        diffs = (1 + contrastive_alpha) * original_logits_common - (contrastive_alpha * noisy_logits_common)

        # 4. Apply mask, set tokens with original logit below cutoff to -inf
        contrastive_logits = diffs.masked_fill(original_logits_common < cutoff, -float("inf"))

        final_contrastive_token_logits = contrastive_logits[-1, :]
        final_contrastive_token_probs = torch.softmax(final_contrastive_token_logits, dim=-1)
        final_contrastive_token_id = torch.argmax(final_contrastive_token_probs).item()

        logger.info("Self-check complete.")

        return {
            "status": "success",
            "original_trace": original_trace,
            "noisy_trace": noisy_trace,
            "contrastive_alpha": contrastive_alpha,
            "common_length": min_len,
            "contrastive_logits_all_steps": contrastive_logits.cpu(),
            "final_contrastive_token_logits": final_contrastive_token_logits.cpu(),
            "final_contrastive_token_id": final_contrastive_token_id,
        }

    def generate_self_reflection(
        self,
        image_path: str,
        question: str,
        voted_answer: str,
        
        # --- 新增参数 ---
        expert_model_name: str,
        local_generator: Optional[Any],
        local_processor: Optional[Any],
        sampling_params: Optional[Dict[str, Any]]

    ) -> str:
        """
        Call the (modified) Verifier to generate a Self-Reflection critique for the given answer.
        This is now a wrapper that handles local or API verification.

        Args:
            image_path (str): Path to the image.
            question (str): Original question text submitted to Verifier (without prompt).
            voted_answer (str): The voted answer that needs to be critiqued.

            expert_model_name (str): Verifier name to use (e.g., 'gpt-5...', 'GRPO-...')
            local_generator (Optional[Any]):
            local_processor (Optional[Any]):
            sampling_params (Optional[Dict[str, Any]]):

        Returns:
            str: Critique text returned by Verifier, or error message.
        """
        logger.info("Generating Self-Reflection (Critique) via %s", expert_model_name)

        # 1. Load image
        try:
            image_pil = load_image(image_path)
        except Exception as e:
            return f"Error: Unable to load image {image_path}: {e}"

        # 2. Call (modified) verify function with all parameters
        try:
            critique = verify(
                image=image_pil,
                question=question,
                verification_mode=VerificationMode.SELF_REFLECTION,
                model_answer=voted_answer,

                expert_model_name=expert_model_name,
                local_generator=local_generator,
                local_processor=local_processor,
                image_path=image_path,  # verifier also needs image_path in local mode
                sampling_params=sampling_params
            )

            if not isinstance(critique, str):
                return f"Error: Verifier returned non-string type: {type(critique)}"

            return critique

        except Exception as e:
            return f"Error: Exception when calling verify: {e}"


class InternVLGenerator(BaseGenerator):
    """
    Generator wrapper for InternVL models (model.chat API from trust_remote_code).
    It mirrors the interface expected by the self-consistency pipeline.
    """

    def __init__(self, model: PreTrainedModel, tokenizer, device: str = 'cuda'):
        super().__init__(model, tokenizer, device)
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate_sample(self, inputs: Dict[str, Any], sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        pixel_values = inputs["pixel_values"].to(self.device)
        question = inputs["question"]
        num_patches_list = inputs.get("num_patches_list", None)
        tokenizer = inputs.get("tokenizer", self.tokenizer)

        gen_params = sampling_params.copy()
        window_size = gen_params.pop('confidence_window_size', 20)
        seed = gen_params.pop('seed', None)

        if seed is not None and gen_params.get('do_sample', False):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # To get logits, we need to call generate method directly instead of chat method
        # First prepare inputs, similar to chat method logic
        # Use model's existing conv_template
        template = self.model.conv_template
        template.system_message = self.model.system_message
        
        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id
        
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        
        # Prepare generation parameters
        # To get logits, we need to set return_dict_in_generate=True
        gen_params['eos_token_id'] = eos_token_id

        logger.debug("Generation params keys: %s", list(gen_params.keys()))
        logger.debug("max_new_tokens: %s", gen_params.get('max_new_tokens', 'NOT SET'))
        logger.debug("max_length: %s", gen_params.get('max_length', 'NOT SET'))
        logger.debug("do_sample: %s", gen_params.get('do_sample', 'NOT SET'))
        
        # Create GenerationConfig object if needed
        # InternVL's generate method accepts generation_config parameter
        generation_config = None
        if 'generation_config' in gen_params:
            generation_config = gen_params.pop('generation_config')
        elif any(k in gen_params for k in ['max_new_tokens', 'max_length', 'do_sample', 'temperature', 'top_p', 'top_k']):
            # If generation parameters exist, create GenerationConfig object
            gen_config_dict = {
                'eos_token_id': eos_token_id,
                'pad_token_id': eos_token_id,
            }
            # Add other generation parameters
            for key in ['max_new_tokens', 'max_length', 'do_sample', 'temperature', 'top_p', 'top_k']:
                if key in gen_params:
                    gen_config_dict[key] = gen_params[key]
            generation_config = GenerationConfig(**gen_config_dict)

        # Prepare parameters to pass to generate
        gen_params_for_logits = {}
        for key in ['output_scores', 'return_dict_in_generate']:
            gen_params_for_logits[key] = True

        # Pass other parameters as kwargs
        for key in ['max_new_tokens', 'max_length', 'do_sample', 'temperature', 'top_p', 'top_k', 'num_return_sequences']:
            if key in gen_params:
                gen_params_for_logits[key] = gen_params[key]

        # Call generate method to get logits
        try:
            generation_output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                **gen_params_for_logits,
            )
            
            # 处理返回值：可能是 GenerateDecoderOnlyOutput 对象（当 return_dict_in_generate=True）或 tensor
            if hasattr(generation_output, 'sequences'):
                # 返回的是 GenerateDecoderOnlyOutput 对象
                generated_ids = generation_output.sequences
                scores = generation_output.scores if hasattr(generation_output, 'scores') else None
            else:
                # 返回的是 tensor
                generated_ids = generation_output
                scores = None

            logger.debug("Input IDs length: %d", input_ids.shape[1])
            logger.debug("Generated IDs length: %d", generated_ids.shape[1])
            logger.debug("Generated IDs shape: %s", generated_ids.shape)
            logger.debug("Input IDs shape: %s", input_ids.shape)

            # Important: when return_dict_in_generate=True, sequences contain full sequence (input+generated)
            # But if generated sequence length < input length, there might be an issue
            input_ids_len = input_ids.shape[1]

            if generated_ids.shape[1] < input_ids_len:
                # This shouldn't happen, but for safety, assume generated_ids only contains generated part
                logger.warning("Generated IDs length (%d) < Input IDs length (%d)", generated_ids.shape[1], input_ids_len)
                logger.warning("This suggests generated_ids might only contain generated tokens, not the full sequence")
                # In this case, generated_ids is the generated part
                generated_token_ids = generated_ids
            else:
                # Normal case: generated_ids contains full sequence, extract generated part
                generated_token_ids = generated_ids[:, input_ids_len:]

            logger.debug("Generated token IDs shape: %s", generated_token_ids.shape)
            logger.debug("Number of new tokens: %d", generated_token_ids.shape[1])

            # Following original chat method logic: decode entire sequence, then extract via split
            full_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            sep_str = template.sep.strip()

            logger.debug("Full response length: %d", len(full_response))
            logger.debug("Full response (first 300 chars): %s", full_response[:300])
            logger.debug("Sep string: '%s'", sep_str)
            logger.debug("Generated token IDs shape: %s", generated_token_ids.shape)

            # Process response: if generated_ids only contains generated part, use directly
            if generated_ids.shape[1] < input_ids_len:
                # generated_ids only contains generated part, full_response is the generated content
                response = full_response.strip()
                logger.debug("Using full_response directly (generated_ids only contains generated tokens)")
            else:
                # generated_ids contains full sequence, need to extract generated part via separator or role marker
                if sep_str and sep_str in full_response:
                    # Split by separator, take last part (generated content)
                    parts = full_response.split(sep_str)
                    if len(parts) > 1:
                        response = parts[-1].strip()
                    else:
                        response = parts[0].strip()
                else:
                    # If no separator, try splitting by assistant role
                    assistant_role = template.roles[1] if len(template.roles) > 1 else None
                    if assistant_role and assistant_role in full_response:
                        response = full_response.split(assistant_role, 1)[-1].strip()
                    else:
                        # Remove input query
                        query_decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
                        if query_decoded in full_response and len(full_response) > len(query_decoded):
                            response = full_response.replace(query_decoded, "", 1).strip()
                        else:
                            # If nothing found, use full response (generated_ids might only contain generated part)
                            response = full_response.strip()

            logger.debug("Final response length: %d", len(response))
            logger.debug("Final response (first 500 chars): '%s'", response[:500])
            logger.debug("Final response (last 200 chars): '%s'", response[-200:] if len(response) > 200 else response)
            # Check if response contains answer format
            if '\\boxed' in response or 'boxed' in response:
                logger.debug("Found boxed format in response")
            else:
                logger.warning("No boxed format found in response!")

        except Exception as e:
            logger.error("Generation with logits failed: %s", e)
            logger.error("Falling back to original chat method...")
            # If getting logits fails, fall back to original method (without logits)
            generation_output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_params,
            )
            # Original method returns tensor
            generated_ids = generation_output
            scores = None

            # Following original chat method logic
            full_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            sep_str = template.sep.strip()

            # Use same processing logic
            if sep_str:
                parts = full_response.split(sep_str)
                if len(parts) > 1:
                    response = parts[0].strip()
                else:
                    assistant_role = template.roles[1] if len(template.roles) > 1 else None
                    if assistant_role and assistant_role in full_response:
                        response = full_response.split(assistant_role, 1)[-1].strip()
                    else:
                        input_ids_len = input_ids.shape[1]
                        generated_token_ids = generated_ids[:, input_ids_len:]
                        response = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0].strip()
            else:
                assistant_role = template.roles[1] if len(template.roles) > 1 else None
                if assistant_role and assistant_role in full_response:
                    response = full_response.split(assistant_role, 1)[-1].strip()
                else:
                    input_ids_len = input_ids.shape[1]
                    generated_token_ids = generated_ids[:, input_ids_len:]
                    response = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0].strip()

            # Calculate generated token IDs
            input_ids_len = input_ids.shape[1]
            generated_token_ids = generated_ids[:, input_ids_len:]

        # Calculate confidence
        if scores is not None and len(scores) > 0:
            token_confs = compute_token_confidence(scores)
            min_conf = calculate_bottom_window_confidence(token_confs, window_size)
            avg_conf = np.mean(token_confs) if token_confs else 0.0
            logits_scores = tuple(s.cpu() for s in scores)
        else:
            token_confs = []
            min_conf = 0.0
            avg_conf = 0.0
            logits_scores = tuple()

        # Check response
        if not response or len(response.strip()) == 0:
            logger.warning("Empty response after processing!")
            logger.warning("Generated token IDs shape: %s", generated_token_ids.shape)
            if generated_token_ids.shape[1] > 0:
                logger.warning("Generated token IDs (first 50): %s", generated_token_ids[0][:50].cpu().tolist())
                decoded_with_skip = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
                decoded_without_skip = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=False)[0]
                logger.warning("Decoded with skip_special_tokens=True: '%s'", decoded_with_skip[:200])
                logger.warning("Decoded with skip_special_tokens=False: '%s'", decoded_without_skip[:200])
                # If skip_special_tokens=True results in empty string, try False
                if decoded_without_skip and len(decoded_without_skip.strip()) > 0:
                    response = decoded_without_skip.strip()
                    logger.info("Using response decoded with skip_special_tokens=False")
        else:
            # Check if response contains obvious prompt content
            prompt_indicators = ["Please solve the problem", "What is the main focus", "<image>"]
            contains_prompt = any(indicator in response for indicator in prompt_indicators)
            if contains_prompt:
                logger.warning("Response appears to contain prompt!")
                logger.warning("Response (first 500 chars): %s", response[:500])
                # Try more aggressive approach: only use generated part
                response_from_generated_only = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0].strip()
                if response_from_generated_only and response_from_generated_only != response:
                    logger.info("Using response from generated tokens only: '%s'", response_from_generated_only[:200])
                    response = response_from_generated_only

        
        trace = {
            "full_text": response,
            "generated_text": response,
            "extracted_answer": extract_answer(response),
            "token_ids": generated_token_ids[0].cpu().tolist(),
            "num_tokens": int(len(generated_token_ids[0])),
            "confs": token_confs,
            "min_conf": min_conf,
            "avg_conf": avg_conf,
            "logits_scores": logits_scores,
        }
        return trace

    @torch.inference_mode()
    def generate_sample_batch(
        self,
        inputs: Dict[str, Any],
        sampling_params: Dict[str, Any],
        num_samples_per_input: int = 1
    ) -> List[Dict[str, Any]]:
        traces: List[Dict[str, Any]] = []
        base_seed = sampling_params.get('seed', None)
        for i in range(num_samples_per_input):
            sample_params = sampling_params.copy()
            if base_seed is not None:
                sample_params['seed'] = base_seed + i
            traces.append(self.generate_sample(inputs, sample_params))
        return traces
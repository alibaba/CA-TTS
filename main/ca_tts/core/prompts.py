"""Prompt templates for CA-TTS verification and planning."""

SELF_CONSISTENCY_VERIFIER_PROMPT = """
Look at the image carefully, and you will be given a list of candidate options: {options_list}.

Generate a normalized confidence (probability) score for each option in this list.
The order of the output probabilities must strictly correspond to the order of the options in options_list.
The sum of all probabilities must equal 1.

Your output must strictly follow the format below, and must **only** be this array. Do not include any other text, labels, or explanations:
[p_1, p_2, ..., p_n]
"""

SELF_REFLECTION_VERIFIER_PROMPT = '''
Given the following information:
Question: {question}
Model Answer: {model_answer}

Please generate a self-reflection critique according to the given image and information above. Your output must strictly follow the format below, without any other text or explanation:

Critique: Based on this question, your answer is "{model_answer}", <Please fill in your concise, objective critique of this answer here, for example, questioning its accuracy, relevance, or completeness>
'''

PLANNER_PROMPT = """You are an Expert Planner for the CA-TTS multi-module reasoning system.

Given an image and question, decide the optimal execution order for three modules:
- self_consistency: Multiple sampling with confidence-weighted voting
- self_reflection: Expert critique to refine low-confidence answers
- self_check: Visual contrastive decoding between original and noised images

Analyze the question complexity and visual requirements.
Output ONLY a JSON object with this exact format:
{{"module_order": ["self_consistency", "self_reflection", "self_check"], "reasoning": "brief explanation"}}

Question: {question}"""
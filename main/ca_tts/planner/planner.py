"""Expert Planner for CA-TTS module scheduling (Section 3.3.4 of paper)."""
import json
import enum
import logging
from dataclasses import dataclass
from typing import List, Optional, Any
from PIL import Image

logger = logging.getLogger(__name__)


class TTSModule(enum.Enum):
    """Three decoupled TTS modules as described in the paper."""
    SELF_CONSISTENCY = "self_consistency"
    SELF_REFLECTION = "self_reflection"
    SELF_CHECK = "self_check"


@dataclass
class PlannerDecision:
    """
    Planner output containing module execution order.

    Attributes:
        module_order: Scheduling permutation π of the three TTS modules
        reasoning: Expert's explanation for the chosen order
    """
    module_order: List[TTSModule]
    reasoning: str


class ExpertPlanner:
    """
    LLM-based Expert Planner (M^Planner_expert) as described in paper Section 3.3.4.

    The Expert Model functions as a Planner, responsible for module scheduling.
    Before inference, the planner analyzes the input (image, question) and outputs
    a scheduling order π, which is a permutation of the three modules:
    - Self-Consistency (M_sc)
    - Self-Reflection (M_sr)
    - Self-Check (M_sk)

    This adaptive scheduling is feasible because the three modules are fully
    decoupled and order-insensitive. Each module contributes to the shared
    voting dictionary V_final.
    """

    def __init__(
        self,
        expert_model_name: str = "gemini-2.5-pro-06-17",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the Expert Planner.

        Args:
            expert_model_name: Name of the expert model to use for planning
            api_key: API key for the expert model (if None, reads from env)
            base_url: Base URL for API (if None, uses default)
        """
        self.expert_model_name = expert_model_name
        self.api_key = api_key
        self.base_url = base_url

        # Import prompt
        from ..core.prompts import PLANNER_PROMPT
        self.planner_prompt = PLANNER_PROMPT

        logger.info(f"Initialized ExpertPlanner with model: {expert_model_name}")

    def plan(
        self,
        image: Image.Image,
        question: str,
        **kwargs
    ) -> PlannerDecision:
        """
        Call the expert model to determine the scheduling order π.

        Args:
            image: Input image
            question: Input question
            **kwargs: Additional arguments for the expert model

        Returns:
            PlannerDecision with the ordered module list and reasoning
        """
        logger.info("Planning module execution order...")

        # Format the prompt with the question
        prompt_text = self.planner_prompt.format(question=question)

        # Call the expert model
        response_text = self._call_expert_model(image, prompt_text)

        # Parse the response
        decision = self._parse_planner_response(response_text)

        logger.info(f"Planned order: {[m.value for m in decision.module_order]}")
        logger.debug(f"Reasoning: {decision.reasoning}")

        return decision

    def _call_expert_model(self, image: Image.Image, prompt_text: str) -> str:
        """
        Call the expert model API.

        Args:
            image: Input image
            prompt_text: Formatted prompt

        Returns:
            Raw response text from the expert model
        """
        import os
        import io
        import base64

        # Determine which API to use
        if self.expert_model_name.startswith("gpt-") or self.expert_model_name.startswith("72") or self.expert_model_name.startswith("max"):
            # Use OpenAI-compatible API
            from openai import OpenAI

            api_key = self.api_key or os.environ.get("EXPERT_API_KEY")
            base_url = self.base_url or os.environ.get("EXPERT_API_BASE_URL", "https://api.openai.com/v1")

            if not api_key:
                raise ValueError("EXPERT_API_KEY environment variable not set")

            client = OpenAI(api_key=api_key, base_url=base_url)

            # Convert image to base64
            buffered = io.BytesIO()
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Model name mapping
            model_name = self.expert_model_name
            if model_name.startswith("72"):
                model_name = "qwen2.5-vl-72b-instruct"
            elif model_name.startswith("max"):
                model_name = "qwen-vl-max-latest"

            # Build message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512,
                temperature=0.0  # Deterministic planning
            )

            return response.choices[0].message.content

        elif self.expert_model_name.startswith("gemini-"):
            # Use Google Gemini API
            import google.generativeai as genai

            api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.expert_model_name)

            response = model.generate_content([prompt_text, image])
            return response.text

        else:
            raise ValueError(f"Unsupported expert model: {self.expert_model_name}")

    def _parse_planner_response(self, response_text: str) -> PlannerDecision:
        """
        Parse the expert model's response to extract module order and reasoning.

        Args:
            response_text: Raw response from expert model

        Returns:
            PlannerDecision object
        """
        try:
            # Try to extract JSON from the response
            # The response might contain extra text, so we look for JSON block
            import re

            # Find JSON object in response
            json_match = re.search(r'\{[^{}]*"module_order"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Try the entire response
                json_str = response_text.strip()

            # Parse JSON
            data = json.loads(json_str)

            # Extract module order
            module_order_strs = data.get("module_order", [])
            module_order = []

            for module_str in module_order_strs:
                try:
                    module = TTSModule(module_str)
                    module_order.append(module)
                except ValueError:
                    logger.warning(f"Unknown module name: {module_str}, skipping")

            # Validate that we have all three modules
            if len(module_order) != 3 or set(module_order) != set(TTSModule):
                logger.warning(f"Invalid module order: {module_order}, using default order")
                module_order = [
                    TTSModule.SELF_CONSISTENCY,
                    TTSModule.SELF_REFLECTION,
                    TTSModule.SELF_CHECK
                ]

            reasoning = data.get("reasoning", "No reasoning provided")

            return PlannerDecision(
                module_order=module_order,
                reasoning=reasoning
            )

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.error(f"Failed to parse planner response: {e}")
            logger.debug(f"Response text: {response_text}")

            # Return default order on parse failure
            return PlannerDecision(
                module_order=[
                    TTSModule.SELF_CONSISTENCY,
                    TTSModule.SELF_REFLECTION,
                    TTSModule.SELF_CHECK
                ],
                reasoning="Failed to parse planner response, using default order"
            )

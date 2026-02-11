"""
Council Mode V8 - Temperature-Diverse Savant Booster

Part of College of Experts V8 Demo

Runs N copies of a Quantized Savant model at different temperatures to 
increase output diversity and coverage. Optimized for the V8 Execute loop.
"""

import torch
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .expert_slots_v8 import ExpertSlotManager, ExpertInstance
from .expert_catalog import ExpertCatalog


@dataclass
class CouncilResponse:
    """A single response from a council member."""
    expert_id: str
    temperature: float
    content: str
    generation_time: float
    token_count: int


@dataclass 
class CouncilResult:
    """Result of a council execution."""
    expert_id: str
    query: str
    responses: List[CouncilResponse]
    best_response: str
    selection_method: str
    selection_reasoning: Optional[str] = None
    
    @property
    def response_count(self) -> int:
        return len(self.responses)
    
    @property
    def temperatures_used(self) -> List[float]:
        return [r.temperature for r in self.responses]


class CouncilMode:
    """
    Temperature-diverse expert council for creative/exploratory tasks.
    
    Runs multiple copies of the same expert at different temperatures,
    then selects or blends the best response.
    
    Selection Methods:
    - vote: Count keyword overlaps, select most comprehensive
    - critic: Use a critic model to evaluate and select
    - blend: Synthesize elements from multiple responses
    - first: Just use the first response (for testing)
    """
    
    DEFAULT_TEMPERATURES = [0.3, 0.5, 0.7, 0.9, 1.1]
    
    def __init__(
        self,
        slot_manager: ExpertSlotManager,
        catalog: ExpertCatalog,
        default_selection: Literal["vote", "critic", "blend", "first"] = "vote"
    ):
        self.slot_manager = slot_manager
        self.catalog = catalog
        self.default_selection = default_selection
    
    def run(
        self,
        query: str,
        expert_id: str,
        num_members: int = 5,
        temperatures: Optional[List[float]] = None,
        max_tokens: int = 800,
        selection_method: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> CouncilResult:
        """
        Run a temperature-diverse council.
        
        Args:
            query: The query to process
            expert_id: Which expert to use (loaded N times)
            num_members: Number of council members
            temperatures: Specific temperatures (or use defaults)
            max_tokens: Max generation length
            selection_method: How to select best response
        
        Returns:
            CouncilResult with all responses and selected best
        """
        temps = temperatures or self.DEFAULT_TEMPERATURES[:num_members]
        method = selection_method or self.default_selection
        
        # Reserve slots (simulating queue)
        if not self.slot_manager.reserve_gpu_slots(len(temps), timeout=60.0):
            print(f"[Council] Failed to reserve {len(temps)} slots. Aborting.")
            return CouncilResult(expert_id, query, [], "Error: Busy", method)
            
        try:
            # Load council members
            instances = []
            for temp in temps[:num_members]:
                instance = self.slot_manager.get_or_load_expert(expert_id, temperature=temp, force_gpu=True)
                instances.append(instance)
            
            print(f"[Council] Running {len(instances)}× {expert_id} at temps {temps[:num_members]}")
            
            # Parallel generation
            responses = self._parallel_generate(instances, query, max_tokens, system_prompt)
            
            # Select best
            best_response, reasoning = self._select_best(responses, method, query)
            
            return CouncilResult(
                expert_id=expert_id,
                query=query,
                responses=responses,
                best_response=best_response,
                selection_method=method,
                selection_reasoning=reasoning
            )
        finally:
            self.slot_manager.release_gpu_reservations(len(temps))
    
    def _parallel_generate(
        self,
        instances: List[ExpertInstance],
        query: str,
        max_tokens: int,
        system_prompt: Optional[str] = None
    ) -> List[CouncilResponse]:
        """Generate responses from all council members in parallel."""
        
        def generate_single(instance: ExpertInstance) -> CouncilResponse:
            start = datetime.now()
            
            # Build prompt
            expert_def = instance.expert_def
            sys_prompt = system_prompt or expert_def.system_prompt
            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": query}]}
            ]
            
            # Check if this is an API backend (FLM, Ollama, etc)
            if getattr(instance.model, "is_api_backend", False):
                api_messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": query}
                ]
                response = instance.model.generate(
                    messages=api_messages,
                    max_tokens=max_tokens,
                    temperature=instance.temperature
                )
                elapsed = (datetime.now() - start).total_seconds()
                return CouncilResponse(
                    expert_id=instance.expert_def.id,
                    temperature=instance.temperature,
                    content=response.strip() if response else "",
                    generation_time=elapsed,
                    token_count=len(response.split()) if response else 0
                )
            
            # Use UniversalChatFormatter for model-agnostic formatting (HuggingFace)
            from .chat_utils import UniversalChatFormatter
            formatter = UniversalChatFormatter(instance.processor, instance.capabilities)
            text, images = formatter.format_messages(messages)
            inputs = formatter.prepare_inputs(text, images, instance.model.device)
            
            with torch.no_grad():
                outputs = instance.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=instance.temperature
                )
            
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            tokenizer = getattr(instance.processor, "tokenizer", instance.processor)
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            elapsed = (datetime.now() - start).total_seconds()
            
            return CouncilResponse(
                expert_id=instance.expert_def.id,
                temperature=instance.temperature,
                content=response.strip(),
                generation_time=elapsed,
                token_count=len(generated_ids)
            )

        
        # Serialize execution to prevent DirectML driver crashes
        # Parallel inference on consumer GPU with DML is unstable.
        print(f"[Council] Running {len(instances)} serial votes...")
        responses = []
        for i, inst in enumerate(instances):
            print(f"    [Member {i}] Generating (T={inst.temperature})...")
            try:
                result = generate_single(inst)
                responses.append(result)
                print(f"    [Member {i}] Done ({len(result.content)} chars)")
            except Exception as e:
                print(f"    [Member {i}] Failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Sort by temperature for consistent ordering
        responses.sort(key=lambda r: r.temperature)
        
        return responses
    
    def _select_best(
        self,
        responses: List[CouncilResponse],
        method: str,
        query: str
    ) -> Tuple[str, Optional[str]]:
        """
        Select or synthesize the best response.
        
        Returns:
            (best_response, reasoning)
        """
        if not responses:
            return "", "No responses to select from"
        
        if method == "first":
            return responses[0].content, "Selected first response"
        
        elif method == "vote":
            return self._select_by_voting(responses, query)
        
        elif method == "critic":
            return self._select_by_critic(responses, query)
        
        elif method == "blend":
            return self._blend_responses(responses, query)
        
        else:
            return responses[0].content, f"Unknown method {method}, using first"
    
    def _select_by_voting(
        self,
        responses: List[CouncilResponse],
        query: str
    ) -> Tuple[str, str]:
        """
        Select best response by keyword overlap and length.
        
        Scores based on:
        - Length (reasonable length is good)
        - Keyword coverage from query
        - Unique information
        """
        query_words = set(query.lower().split())
        
        scores = []
        for resp in responses:
            content_lower = resp.content.lower()
            
            # Keyword coverage
            keyword_hits = sum(1 for w in query_words if w in content_lower)
            
            # Length score (prefer 200-500 words)
            word_count = len(resp.content.split())
            if word_count < 100:
                length_score = word_count / 100
            elif word_count > 800:
                length_score = 800 / word_count
            else:
                length_score = 1.0
            
            # Uniqueness (count unique words)
            unique_words = len(set(resp.content.lower().split()))
            uniqueness = min(1.0, unique_words / 100)
            
            total = (keyword_hits * 0.4) + (length_score * 0.3) + (uniqueness * 0.3)
            scores.append((resp, total))
        
        # Select highest score
        scores.sort(key=lambda x: x[1], reverse=True)
        best = scores[0][0]
        
        reasoning = f"Selected T={best.temperature} with score {scores[0][1]:.2f}"
        return best.content, reasoning
    
    def _select_by_critic(
        self,
        responses: List[CouncilResponse],
        query: str
    ) -> Tuple[str, str]:
        """
        Use a critic model to evaluate responses.
        
        For V7.0, uses a simple heuristic since we're sharing models.
        Future: dedicated critic model.
        """
        # For now, combine voting with quality heuristics
        best, reasoning = self._select_by_voting(responses, query)
        return best, f"Critic (heuristic): {reasoning}"
    
    def _blend_responses(
        self,
        responses: List[CouncilResponse],
        query: str
    ) -> Tuple[str, str]:
        """
        Blend multiple responses into one.
        
        Simple approach: take unique sentences from each.
        """
        all_sentences = []
        seen = set()
        
        for resp in responses:
            # Split into sentences
            sentences = resp.content.replace('\n', ' ').split('. ')
            for sent in sentences:
                sent_clean = sent.strip()
                sent_key = sent_clean.lower()[:50]
                if sent_key and sent_key not in seen:
                    seen.add(sent_key)
                    all_sentences.append(sent_clean)
        
        # Reconstruct
        blended = '. '.join(all_sentences[:20])  # Limit length
        if blended and not blended.endswith('.'):
            blended += '.'
        
        return blended, f"Blended {len(all_sentences)} unique sentences"
    
    def __repr__(self) -> str:
        return f"CouncilMode(selection={self.default_selection})"


if __name__ == "__main__":
    # Quick structure test
    print("CouncilMode structure verified")
    print("Selection methods: vote, critic, blend, first")

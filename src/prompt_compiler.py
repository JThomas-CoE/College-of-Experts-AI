"""
Prompt Compiler - Two-Phase Architecture

Phase 1 (Preparation): While router is loaded
  - Decompose query into slots
  - Batch retrieve knowledge for all slots
  - Router generates optimized prompts with knowledge
  - Store compiled prompts in RAM

Phase 2 (Execution): After router unloaded
  - Load experts one at a time
  - Use pre-compiled prompts (just fill dependency placeholders)
  - Execute and collect outputs

This eliminates:
  - prompt_templates.py (530 lines)
  - persona_context.py (337 lines)
  - All persona YAML configs
  - Runtime template building
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompiledSlot:
    """A slot with its pre-compiled prompt ready for execution."""
    id: str
    title: str
    expert: str
    prompt: str  # Compiled prompt with knowledge, ready to use
    dependencies: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    knowledge_used: str = ""  # For debugging/logging


@dataclass
class CompiledPlan:
    """Complete execution plan with all prompts pre-compiled."""
    query: str
    slots: List[CompiledSlot]
    execution_order: List[str]  # Topologically sorted slot IDs
    
    # RAM-resident prompt cache
    _prompt_cache: Dict[str, str] = field(default_factory=dict)
    
    def get_prompt(self, slot_id: str, dependency_outputs: Dict[str, str] = None) -> str:
        """Get compiled prompt, filling in dependency placeholders."""
        slot = next((s for s in self.slots if s.id == slot_id), None)
        if not slot:
            raise ValueError(f"Unknown slot: {slot_id}")
        
        prompt = slot.prompt
        
        # Fill dependency placeholders: {{SLOT:slot_id}}
        if dependency_outputs:
            for dep_id, output in dependency_outputs.items():
                placeholder = f"{{{{SLOT:{dep_id}}}}}"
                if placeholder in prompt:
                    # Truncate long outputs to avoid context overflow
                    truncated = output[:2000] if len(output) > 2000 else output
                    prompt = prompt.replace(placeholder, truncated)
        
        return prompt


class PromptCompiler:
    """
    Compiles prompts during preparation phase.
    
    Builds clean, structured prompts with knowledge embedded.
    No LLM needed - just direct construction with proper structure.
    
    Usage:
        compiler = PromptCompiler(knowledge_retriever)
        plan = await compiler.compile(query, slots)
        
        # Later, during execution:
        prompt = plan.get_prompt("backend_api", dependency_outputs)
    """
    
    def __init__(
        self,
        knowledge_retriever: Any,
        embedding_fn: callable = None,
        router_generate_fn: callable = None  # Kept for API compatibility, not used
    ):
        """
        Initialize the prompt compiler.
        
        Args:
            knowledge_retriever: KnowledgeRetriever instance for fetching context
            embedding_fn: Optional - not currently used
            router_generate_fn: Optional - not used (prompts built directly)
        """
        self.knowledge_retriever = knowledge_retriever
    
    async def compile(
        self,
        query: str,
        slots: List[Any],  # List of FrameworkSlot
        max_knowledge_tokens: int = 1200
    ) -> CompiledPlan:
        """
        Compile all prompts for a query's slots.
        
        This runs during preparation phase while router is loaded.
        
        Args:
            query: Original user query
            slots: Decomposed slots from router
            max_knowledge_tokens: Max tokens for knowledge per slot
            
        Returns:
            CompiledPlan with all prompts ready for execution
        """
        logger.info(f"[PromptCompiler] Compiling {len(slots)} slots...")
        
        compiled_slots = []
        
        for slot in slots:
            # 1. Retrieve knowledge for this slot
            knowledge = await self._retrieve_knowledge(
                slot.description,
                slot.title,
                max_tokens=max_knowledge_tokens
            )
            
            # 2. Generate optimized prompt via router
            prompt = await self._generate_prompt(
                slot=slot,
                knowledge=knowledge,
                all_slots=slots,
                original_query=query  # Inject context
            )
            
            # 3. Create compiled slot
            compiled_slot = CompiledSlot(
                id=slot.id,
                title=slot.title,
                expert=slot.persona,  # Using persona as expert ID
                prompt=prompt,
                dependencies=slot.dependencies or [],
                expected_outputs=slot.expected_outputs or [],
                knowledge_used=knowledge[:200] if knowledge else ""
            )
            compiled_slots.append(compiled_slot)
            
            logger.debug(f"[PromptCompiler] Compiled '{slot.id}': {len(prompt)} chars")
        
        # Determine execution order (topological sort)
        execution_order = self._topological_sort(compiled_slots)
        
        plan = CompiledPlan(
            query=query,
            slots=compiled_slots,
            execution_order=execution_order
        )
        
        logger.info(f"[PromptCompiler] Compilation complete. Order: {execution_order}")
        return plan
    
    async def _retrieve_knowledge(
        self,
        task_description: str,
        task_title: str,
        max_tokens: int
    ) -> str:
        """Retrieve relevant knowledge for a task."""
        try:
            results = await self.knowledge_retriever.retrieve(
                query=f"{task_title} {task_description}",
                top_k=2,
                use_memory=True,
                use_local=True,
                min_score=0.55 # Filter weak relevance (e.g. 0.38 HTML matches for Math)
            )
            
            context = self.knowledge_retriever.format_context(
                results,
                max_tokens=max_tokens
            )
            return context
        except Exception as e:
            logger.warning(f"[PromptCompiler] Knowledge retrieval failed: {e}")
            return ""
    
    async def _generate_prompt(
        self,
        slot: Any,
        knowledge: str,
        all_slots: List[Any],
        original_query: str = ""
    ) -> str:
        """
        Build optimized prompt directly (no LLM overhead).
        
        Structure:
        1. Knowledge FIRST (primes context)
        2. Original intent (Context)
        3. Task description
        4. Output format
        5. Dependency placeholders (filled at runtime)
        """
        parts = []
        
        # 1. Knowledge FIRST (if available)
        if knowledge:
            # Clean up knowledge - remove redundant headers
            clean_knowledge = knowledge.strip()
            if clean_knowledge:
                parts.append(f"REFERENCE:\n{clean_knowledge}")
        
        # 2. Global Context
        if original_query:
            parts.append(f"USER QUERY: {original_query}")

        # 3. Clear task description
        parts.append(f"TASK: {slot.description}")
        
        # 3. Expected output format
        if slot.expected_outputs:
            outputs = ", ".join(slot.expected_outputs)
            parts.append(f"OUTPUT FORMAT: {outputs}")
        
        # 4. Dependencies with placeholders (filled at execution time)
        if slot.dependencies:
            dep_section = ["PRIOR WORK (use these as reference):"]
            for dep_id in slot.dependencies:
                dep_slot = next((s for s in all_slots if s.id == dep_id), None)
                if dep_slot:
                    dep_section.append(f"[{dep_slot.title}]:")
                    dep_section.append(f"{{{{SLOT:{dep_id}}}}}")
            parts.append("\n".join(dep_section))
        
        # 5. Conditional instruction based on persona
        non_coding_personas = {
            "math_expert", "legal_contracts", "legal_compliance", 
            "medical_clinical", "general_assistant"
        }
        
        # Check if persona is non-coding (or if 'code'/ 'script' is not in description)
        is_non_coding = slot.persona in non_coding_personas
        
        if is_non_coding:
            parts.append("Provide detailed, complete analysis and implementation details. Show your work.")
        else:
            parts.append("Provide complete, working code. No placeholders or TODOs.")
        
        return "\n\n".join(parts)
    
    def _topological_sort(self, slots: List[CompiledSlot]) -> List[str]:
        """Sort slots by dependency order."""
        # Build adjacency list
        graph = {s.id: set(s.dependencies) for s in slots}
        
        # Kahn's algorithm
        in_degree = {s.id: len(s.dependencies) for s in slots}
        queue = [s.id for s in slots if in_degree[s.id] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for slot_id, deps in graph.items():
                if node in deps:
                    in_degree[slot_id] -= 1
                    if in_degree[slot_id] == 0:
                        queue.append(slot_id)
        
        # If not all nodes processed, there's a cycle
        if len(result) != len(slots):
            logger.warning("[PromptCompiler] Dependency cycle detected, using original order")
            return [s.id for s in slots]
        
        return result


# Convenience function for integration
async def compile_execution_plan(
    query: str,
    slots: List[Any],
    router_generate_fn: callable,
    knowledge_retriever: Any,
    embedding_fn: callable
) -> CompiledPlan:
    """
    Compile a complete execution plan with all prompts.
    
    Call this during preparation phase while router is loaded.
    """
    compiler = PromptCompiler(
        router_generate_fn=router_generate_fn,
        knowledge_retriever=knowledge_retriever,
        embedding_fn=embedding_fn
    )
    return await compiler.compile(query, slots)

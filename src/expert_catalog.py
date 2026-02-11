"""
Expert Catalog - Loads and manages expert definitions from catalog.json

Part of College of Experts V7
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ExpertDefinition:
    """Definition of an expert from the catalog."""
    id: str
    name: str
    emoji: str
    capabilities: List[str]
    system_prompt: str
    recommended_temp: float = 0.5
    model_override: Optional[str] = None  # Use specific model instead of base

    
    def matches_capability(self, capability: str) -> bool:
        """Check if this expert has a given capability."""
        capability_lower = capability.lower()
        return any(cap.lower() == capability_lower or capability_lower in cap.lower() 
                   for cap in self.capabilities)
    
    def capability_score(self, required_capabilities: List[str]) -> float:
        """
        Score how well this expert matches required capabilities (0.0 - 1.1).
        
        Score components:
        1. Accuracy (0.0 - 1.0): Matches / Total Required
        2. Specificity (0.0 - 0.1): Matches / Total Expert Capabilities
        """
        if not required_capabilities:
            return 0.0
            
        matches = sum(1 for cap in required_capabilities if self.matches_capability(cap))
        if matches == 0:
            return 0.0
            
        accuracy = matches / len(required_capabilities)
        specificity = matches / len(self.capabilities) if self.capabilities else 0
        
        return accuracy + (specificity * 0.1)


class ExpertCatalog:
    """
    Manages the catalog of available experts.
    
    In V7.0, all experts use the same base model (Qwen3-VL:4B) with different
    harness prompts. The catalog defines their specializations.
    """
    
    def __init__(self, catalog_path: str = "config/expert_catalog.json"):
        self.catalog_path = Path(catalog_path)
        self.base_model: str = ""
        self.experts: Dict[str, ExpertDefinition] = {}
        self._load_catalog()
    
    def _load_catalog(self) -> None:
        """Load expert definitions from JSON catalog."""
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Expert catalog not found: {self.catalog_path}")
        
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.base_model = data.get("base_model", "models/Qwen3-VL-4B-Instruct")
        
        for expert_data in data.get("experts", []):
            expert = ExpertDefinition(
                id=expert_data["id"],
                name=expert_data["name"],
                emoji=expert_data.get("emoji", "🤖"),
                capabilities=expert_data.get("capabilities", []),
                system_prompt=expert_data["system_prompt"],
                recommended_temp=expert_data.get("recommended_temp", 0.5),
                model_override=expert_data.get("model_override", None)
            )
            self.experts[expert.id] = expert

    
    def get_expert(self, expert_id: str) -> Optional[ExpertDefinition]:
        """Get expert definition by ID."""
        return self.experts.get(expert_id)
    
    def list_experts(self) -> List[ExpertDefinition]:
        """List all available experts."""
        return list(self.experts.values())
    
    def find_experts_for_capabilities(
        self, 
        capabilities: List[str], 
        top_k: int = 3
    ) -> List[ExpertDefinition]:
        """
        Find the best experts for a set of required capabilities.
        
        Returns up to top_k experts sorted by capability match score.
        """
        scored = [
            (expert, expert.capability_score(capabilities))
            for expert in self.experts.values()
        ]
        # Sort by score descending, filter out zero scores
        scored = [(e, s) for e, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [expert for expert, score in scored[:top_k]]
    
    def get_expert_ids(self) -> List[str]:
        """Get list of all expert IDs."""
        return list(self.experts.keys())
    
    def __len__(self) -> int:
        return len(self.experts)
    
    def __repr__(self) -> str:
        return f"ExpertCatalog({len(self)} experts, base_model={self.base_model})"


# Convenience function for quick loading
def load_catalog(path: str = "config/expert_catalog.json") -> ExpertCatalog:
    """Load and return the expert catalog."""
    return ExpertCatalog(path)


if __name__ == "__main__":
    # Quick test
    catalog = load_catalog()
    print(f"Loaded: {catalog}")
    print(f"\nAvailable experts:")
    for expert in catalog.list_experts():
        print(f"  {expert.emoji} {expert.name} ({expert.id})")
        print(f"      Capabilities: {', '.join(expert.capabilities)}")
    
    print(f"\nBest experts for ['python', 'security']:")
    for expert in catalog.find_experts_for_capabilities(["python", "security"]):
        print(f"  {expert.emoji} {expert.name}")

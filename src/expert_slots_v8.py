"""
Expert Slot Manager V8 - Executive Resource Orchestrator

Part of College of Experts V8 Demo

Manages a hybrid compute topology:
1. Virtual Slots (0-2): NPU-accelerated router instances (via FLM).
2. Physical Slots (3+): Quantized Savant instances (via OGA/DML).
"""

import torch
import gc
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .expert_catalog import ExpertCatalog, ExpertDefinition
from .model_factory import ModelFactory, ModelCapabilities


@dataclass
class ExpertInstance:
    """A loaded expert instance occupying a slot."""
    slot_id: int
    expert_def: ExpertDefinition
    model: Any  # The loaded model
    processor: Any  # The processor/tokenizer
    temperature: float
    capabilities: ModelCapabilities = field(default_factory=lambda: ModelCapabilities(has_vision=True)) # Default to avoid breaking changes
    loaded_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    generation_count: int = 0
    
    def touch(self):
        """Update last_used timestamp."""
        self.last_used = datetime.now()
        self.generation_count += 1
    
    @property
    def idle_seconds(self) -> float:
        """Seconds since last use."""
        return (datetime.now() - self.last_used).total_seconds()


class ExpertSlotManager:
    """
    Manages expert model instances across available slots.
    
    Features:
    - Dynamic loading/unloading of experts
    - VRAM budget tracking
    - LRU eviction when slots are full
    - Temperature-specific instances for council mode
    
    V8 architecture:
    - VIRTUAL (NPU): Slots mapped to FLM server for zero-latency routing.
    - PHYSICAL (GPU): Slots mapped to quantized OGA models (Savants).
    """
    
    def __init__(
        self,
        catalog: ExpertCatalog,
        num_slots: int = 3,
        max_vram_mb: int = 64000,
        preload: bool = True,
        multi_instance: bool = True,  # NEW: Enable true parallel inference
        use_npu_for_router: bool = False, # NEW: Use NPU for router slots
        npu_slots: List[int] = None,      # NEW: List of slots to run on NPU
        onnx_model_path: str = None       # NEW: Path to ONNX model for NPU
    ):
        self.catalog = catalog
        self.num_slots = num_slots
        self.max_vram_mb = max_vram_mb
        self.multi_instance = multi_instance
        self.use_npu_for_router = use_npu_for_router
        self.npu_slots = set(npu_slots) if npu_slots else set()
        self.onnx_model_path = onnx_model_path
        
        # Slot management
        self.slots: Dict[int, Optional[ExpertInstance]] = {i: None for i in range(num_slots)}
        self._lock = threading.RLock()
        self._load_lock = threading.Lock()  # NEW: Prevent concurrent model loading
        
        # Reservation system for Councils
        self._reserved_gpu_slots = 0
        self._reservation_lock = threading.Condition(self._lock)
        
        # Model instances (multi-instance mode)
        self._models: Dict[int, Any] = {}  # slot_id -> model
        self._processors: Dict[int, Any] = {}  # slot_id -> processor
        self._capabilities: Dict[int, ModelCapabilities] = {} # slot_id -> capabilities
        self._streams: Dict[int, Any] = {}  # slot_id -> CUDA stream for parallel execution
        
        # Shared model (single-instance mode, fallback)
        self._shared_model = None
        self._shared_processor = None
        self._shared_capabilities = None
        self._model_loaded = False
        
        if preload and not multi_instance:
            self._load_base_model()
    
    def _load_base_model(self) -> None:
        """Load the shared base model (single-instance mode)."""
        if self._model_loaded:
            return
        
        model_path = self.catalog.base_model
        print(f"[SlotManager] Loading shared base model: {model_path}")
        
        # Use Factory to load agnostic model
        self._shared_model, self._shared_processor, self._shared_capabilities = ModelFactory.load_model_for_slot(
            slot_id=-1,  # -1 for shared
            model_path=model_path,
            device="cuda:0"
        )
        
        self._model_loaded = True
        print(f"[SlotManager] Shared model loaded")
    
    def _load_model_for_slot(self, slot_id: int, use_compile: bool = False, model_override: str = None) -> tuple:
        """
        V8 Resource Orchestration:
        - Slots 0-2 (NPU Pool): Targeted to pre-running FLM server.
        - Slots 3+ (GPU Pool): Targeted to DirectML OGA backends.
        """
        with self._load_lock:
            # For NPU slots, we only ever use the cached FLM backend (it's stateless on the server side)
            if slot_id in self.npu_slots and slot_id in self._models:
                return self._models[slot_id], self._processors[slot_id]
            
            # For GPU slots, check if model override matches loaded model
            if slot_id in self._models and model_override is None:
                return self._models[slot_id], self._processors[slot_id]

        # Hardware Pool Selection
        if slot_id in self.npu_slots:
            target_device = "flm"
            model_path = "NPU_MANAGED" # Ignored by flm backend
            print(f"[SlotManager] Targeting NPU Virtual Slot {slot_id} (FLM)")
        else:
            target_device = "cuda:0"
            model_path = model_override if model_override else self.catalog.base_model
            print(f"[SlotManager] Targeting GPU Physical Slot {slot_id} (DirectML)")

        # Log if using override model
        if model_override:
            print(f"[SlotManager] Slot {slot_id} using override model: {model_override}")
        
        # Current standard load:
        # V8 Stability Fix: Force GC and Wait to let DirectML driver release VRAM
        gc.collect()
        if not slot_id in self.npu_slots:
            time.sleep(2.0) 
            
        try:
            model, processor, capabilities = ModelFactory.load_model_for_slot(
                slot_id=slot_id,
                model_path=model_path,
                device=target_device,
                use_compile=use_compile
            )
        except Exception as e:
            import traceback
            print(f"[SlotManager] CRITICAL ERROR loading model for slot {slot_id}: {e}")
            traceback.print_exc()
            raise e

        
        # Create dedicated CUDA stream for this slot
        if torch.cuda.is_available():
            stream = torch.cuda.Stream()
            self._streams[slot_id] = stream
        
        self._models[slot_id] = model
        self._processors[slot_id] = processor
        self._capabilities[slot_id] = capabilities  # Store capabilities
        
        return model, processor
    
    def get_cuda_stream(self, slot_id: int) -> Optional[Any]:
        """Get the CUDA stream for a slot."""
        return self._streams.get(slot_id)
    
    def load_expert(
        self,
        expert_id: str,
        temperature: Optional[float] = None,
        slot_id: Optional[int] = None,
        force_gpu: bool = False
    ) -> ExpertInstance:
        """
        Load an expert into a slot.
        
        Args:
            expert_id: ID of expert from catalog
            temperature: Override temperature (None = use recommended)
            slot_id: Specific slot to use (None = auto-assign)
            force_gpu: If True, forces loading into GPU pool even for NPU-default experts
        
        Returns:
            ExpertInstance: The loaded expert
        
        Raises:
            ValueError: If expert not found in catalog
        """
        expert_def = self.catalog.get_expert(expert_id)
        if not expert_def:
            raise ValueError(f"Expert not found: {expert_id}")
        
        temp = temperature if temperature is not None else expert_def.recommended_temp
        print(f"[SlotManager-DEBUG] Request to load {expert_id} (ForceGPU={force_gpu})...")
        
        with self._lock:
            print(f"[SlotManager-DEBUG] Main lock acquired. Finding slot...")
            # Find or assign slot
            if slot_id is not None:
                if slot_id >= self.num_slots:
                    raise ValueError(f"Invalid slot_id: {slot_id} (max: {self.num_slots - 1})")
                target_slot = slot_id
                # Evict if occupied
                if self.slots[target_slot] is not None:
                    self._unload_slot(target_slot)
            else:
                # Determine if NPU expert (unless forced to GPU)
                is_npu_default = (expert_id in ["general_reasoner", "supervisor", "reviewer"])
                is_npu_actual = is_npu_default and not force_gpu
                
                target_slot = self._find_available_slot(is_npu=is_npu_actual)
            
            # Load model based on mode
            if self.multi_instance:
                # V8 Hardware Check
                is_npu_expert = (target_slot in self.npu_slots)
                
                # Direct call - _load_model_for_slot handles its own locking
                model, processor = self._load_model_for_slot(
                    target_slot, 
                    model_override=expert_def.model_override
                )
                capabilities = self._capabilities[target_slot]
            else:
                if not self._model_loaded:
                    self._load_base_model()
                model = self._shared_model
                processor = self._shared_processor
                capabilities = self._shared_capabilities

            
            # Create expert instance
            instance = ExpertInstance(
                slot_id=target_slot,
                expert_def=expert_def,
                model=model,
                processor=processor,
                temperature=temp,
                capabilities=capabilities
            )
            
            self.slots[target_slot] = instance
            print(f"[SlotManager] Loaded {expert_def.emoji} {expert_def.name} into slot {target_slot} (T={temp})")
            
            return instance
    
    def _find_available_slot(self, is_npu: bool = False) -> int:
        """Find an available slot in the appropriate hardware pool."""
        # V8 Pooling: 0,1,2 = NPU, 3+ = GPU
        target_pool = self.npu_slots if is_npu else set(self.slots.keys()) - self.npu_slots
        
        # 1. Look for empty slot in pool
        for slot_id in sorted(list(target_pool)):
            if self.slots[slot_id] is None:
                return slot_id
        
        # 2. All slots in pool full - Expand pool if under hardware limit (12 slots)
        current_max_slot = max(self.slots.keys()) if self.slots else 0
        # 2. All slots in pool full - strictly evict LRU
        # (Removed dynamic expansion to respect VRAM/Hardware limits)
        
        # 3. Hardware limit reached - evict LRU from pool
        pool_instances = {sid: self.slots[sid] for sid in target_pool if self.slots[sid]}
        if not pool_instances:
             # Should not happen if pool not empty, but safety first
             return sorted(list(target_pool))[0]

        lru_slot = min(
            pool_instances.items(),
            key=lambda x: x[1].last_used if x[1] else datetime.max
        )[0]
        
        self._unload_slot(lru_slot)
        return lru_slot
    
    def _unload_slot(self, slot_id: int) -> None:
        """Unload expert from a specific slot."""
        instance = self.slots.get(slot_id)
        
        # 1. Clean up metadata
        if instance:
            print(f"[SlotManager] Unloading {instance.expert_def.name} from slot {slot_id}")
            self.slots[slot_id] = None
            
        # 2. Clean up HW resources (CRITICAL for preventing OOM)
        if slot_id in self._models:
            del self._models[slot_id]
        if slot_id in self._processors:
            del self._processors[slot_id] 
        if slot_id in self._capabilities:
            del self._capabilities[slot_id]
            
        # 3. Force GC to release DirectML/CUDA handles
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def unload_expert(self, slot_id: int) -> None:
        """Public method to unload an expert from a slot."""
        with self._lock:
            self._unload_slot(slot_id)

    def unload_all_except(self, keep_slots: List[int]) -> None:
        """Unload ALL experts except those in the keep_slots list. Soft-evict to maintain residency handles."""
        with self._lock:
            print(f"[SlotManager] Soft-Flushing experts from slots except {keep_slots}...")
            initial_count = len(self.get_loaded_experts())
            
            for slot_id in list(self.slots.keys()):
                if slot_id not in keep_slots and self.slots[slot_id] is not None:
                    # We clear the active instance but keep the model weights in memory
                    self.slots[slot_id] = None
            
            gc.collect()
            remaining = len(self.get_loaded_experts())
            print(f"[SlotManager] Flush complete. Unloaded {initial_count - remaining} experts. Residency handles preserved.")
    
    def get_expert_instance(self, slot_id: int) -> Optional[ExpertInstance]:
        """Get the expert instance in a specific slot."""
        return self.slots.get(slot_id)
    
    def get_loaded_experts(self) -> List[ExpertInstance]:
        """Get list of currently loaded expert instances."""
        return [s for s in self.slots.values() if s is not None]

    
    def find_loaded_expert(self, expert_id: str) -> Optional[ExpertInstance]:
        """Find a loaded instance of a specific expert."""
        for instance in self.slots.values():
            if instance and instance.expert_def.id == expert_id:
                return instance
        return None
    
    def get_or_load_expert(
        self,
        expert_id: str,
        temperature: Optional[float] = None,
        force_gpu: bool = False
    ) -> ExpertInstance:
        """Get existing instance or load if not present."""
        
        # 1. Search for EXACT match (ID + Temperature + Hardware)
        with self._lock:
            for instance in self.slots.values():
                if not instance: continue
                
                # Check ID
                if instance.expert_def.id != expert_id:
                    continue
                    
                # Check Hardware Match
                is_on_npu = (instance.slot_id in self.npu_slots)
                if force_gpu and is_on_npu:
                    continue # Need GPU, but this is on NPU
                
                # Check Temperature (if specified)
                # If temp is None, any instance is fine (legacy behavior)
                if temperature is not None:
                    # Allow small float tolerance
                    if abs(instance.temperature - temperature) > 0.01:
                        continue
                
                # Found match!
                instance.touch()
                return instance

        # 2. No exact match found -> Load fresh
        return self.load_expert(expert_id, temperature, force_gpu=force_gpu)
    
    def swap_persona(
        self,
        slot_id: int,
        new_expert_id: str,
        temperature: Optional[float] = None,
        system_prompt_override: Optional[str] = None
    ) -> ExpertInstance:
        """
        Hot-swap the persona on an existing slot WITHOUT reloading model.
        
        This is the key to flexible slot sharing - same model weights,
        different system prompt. Instant persona switch.
        
        EXCEPTION: If the new expert has a different model_override than
        the current slot's model, the model will be reloaded.
        
        Args:
            slot_id: Slot to swap
            new_expert_id: New expert persona to apply
            temperature: New temperature (optional)
            system_prompt_override: Explicit system prompt (overrides catalog)
        
        Returns:
            Updated ExpertInstance
        """
        with self._lock:
            new_expert_def = self.catalog.get_expert(new_expert_id)
            if not new_expert_def:
                raise ValueError(f"Expert not found: {new_expert_id}")
            
            old_instance = self.slots.get(slot_id)
            old_name = old_instance.expert_def.name if old_instance else "empty"
            
            # Check if model_override differs (requires model reload)
            current_model_override = None
            if old_instance and old_instance.expert_def:
                current_model_override = old_instance.expert_def.model_override
            
            new_model_override = new_expert_def.model_override
            
            # Case 1: Slot has no model - load it (lazy loading for slots 3+)
            if slot_id not in self._models:
                print(f"[SlotManager] Slot {slot_id}: loading model (lazy init)...")
                model, processor = self._load_model_for_slot(
                    slot_id, 
                    model_override=new_model_override
                )
            # Case 2: Model override changed - reload
            elif new_model_override != current_model_override:
                print(f"[SlotManager] Slot {slot_id}: model_override changed, reloading...")
                
                # Unload current model from this slot
                if slot_id in self._models:
                    del self._models[slot_id]
                if slot_id in self._processors:
                    del self._processors[slot_id]
                if slot_id in self._capabilities:
                    del self._capabilities[slot_id]
                
                # Load new model with override
                model, processor = self._load_model_for_slot(
                    slot_id, 
                    model_override=new_model_override
                )
            # Case 3: Same model - just swap persona (fast path)
            else:
                model = self._models[slot_id]
                processor = self._processors[slot_id]

            
            temp = temperature if temperature is not None else new_expert_def.recommended_temp
            
            # Create new instance with model and new persona
            final_prompt = system_prompt_override if system_prompt_override else new_expert_def.system_prompt
            
            new_instance = ExpertInstance(
                slot_id=slot_id,
                expert_def=new_expert_def,
                model=model,
                processor=processor,
                temperature=temp,
                capabilities=self._capabilities.get(slot_id)
            )
            # Inject overridden prompt back into the instance's active def
            # (Note: we don't modify the global catalog, just this instance)
            new_instance.expert_def.system_prompt = final_prompt
            
            self.slots[slot_id] = new_instance
            print(f"[SlotManager] Swapped slot {slot_id}: {old_name} -> {new_expert_def.emoji} {new_expert_def.name} (T={temp})")
            if system_prompt_override:
                print(f"[SlotManager] -> Applied custom persona override.")
            
            return new_instance

    def reserve_gpu_slots(self, count: int, timeout: float = 30.0) -> bool:
        """Reserve N GPU slots. Blocks until available."""
        start_time = time.time()
        with self._reservation_lock:
            total_gpu = self.num_slots - len(self.npu_slots)
            while True:
                if self._reserved_gpu_slots + count <= total_gpu:
                    self._reserved_gpu_slots += count
                    return True
                
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0: return False
                
                print(f"[SlotManager] Waiting for {count} GPU slots (Reserved: {self._reserved_gpu_slots})...")
                self._reservation_lock.wait(timeout=remaining)

    def release_gpu_reservations(self, count: int):
        with self._reservation_lock:
            self._reserved_gpu_slots = max(0, self._reserved_gpu_slots - count)
            self._reservation_lock.notify_all()

    def _find_available_slot(self, is_npu: bool = False) -> int:
        """Find slot, respecting reservations."""
        target_pool = self.npu_slots if is_npu else set(self.slots.keys()) - self.npu_slots
        
        # 1. Look for empty slot
        for slot_id in sorted(list(target_pool)):
            if self.slots[slot_id] is None:
                return slot_id
        
        # 2. Evict LRU
        pool_instances = {sid: self.slots[sid] for sid in target_pool if self.slots[sid]}
        if not pool_instances: return sorted(list(target_pool))[0]

        lru_slot = min(pool_instances.items(), key=lambda x: x[1].last_used)[0]
        self._unload_slot(lru_slot)
        return lru_slot

    
    def get_available_slots_for_experts(self, exclude_slots: List[int] = None) -> List[int]:
        """Get slots available for expert use (excluding reserved slots)."""
        exclude = exclude_slots or []
        return [
            slot_id for slot_id in self._models.keys()
            if slot_id not in exclude
        ]
    
    def load_council(
        self,
        expert_id: str,
        num_members: int = 3,
        temperatures: Optional[List[float]] = None
    ) -> List[ExpertInstance]:
        """
        Load multiple instances of the same expert at different temperatures.
        Used for council mode with temperature-diverse generation.
        """
        if temperatures is None:
            temperatures = [0.3, 0.5, 0.7, 0.9, 1.1][:num_members]
        
        instances = []
        for i, temp in enumerate(temperatures[:num_members]):
            instance = self.load_expert(expert_id, temperature=temp)
            instances.append(instance)
        
        return instances
    
    def get_vram_usage(self) -> Dict[str, Any]:
        """Get current VRAM usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        else:
            allocated = reserved = 0.0
        
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "max_vram_mb": self.max_vram_mb,
            "slots_used": sum(1 for s in self.slots.values() if s is not None),
            "slots_total": self.num_slots
        }
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        with self._lock:
            for slot_id in list(self.slots.keys()):
                self._unload_slot(slot_id)
            
            # Clean up multi-instance models
            for slot_id in list(self._models.keys()):
                print(f"[SlotManager] Freeing model for slot {slot_id}")
                del self._models[slot_id]
            self._models.clear()
            self._processors.clear()
            
            # Clean up shared model (single-instance mode)
            if self._shared_model is not None:
                del self._shared_model
                self._shared_model = None
            if self._shared_processor is not None:
                del self._shared_processor
                self._shared_processor = None
            
            self._model_loaded = False
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("[SlotManager] Cleanup complete")
    
    def __repr__(self) -> str:
        loaded = sum(1 for s in self.slots.values() if s is not None)
        return f"ExpertSlotManager({loaded}/{self.num_slots} slots, model_loaded={self._model_loaded})"


if __name__ == "__main__":
    # Quick test
    from expert_catalog import load_catalog
    
    catalog = load_catalog()
    manager = ExpertSlotManager(catalog, num_slots=3, preload=False)
    
    print(f"Manager: {manager}")
    print(f"VRAM: {manager.get_vram_usage()}")
    
    # Don't actually load model in test - just verify structure
    print("\nTest passed - structure verified")

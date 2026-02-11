"""
VRAM Manager - V12.1 Memory-Aware Scheduling

Components:
- VRAMBudget: Tracks reservations and actual usage
- SavantPool: Loads/unloads models with reference counting
- VRAMAwareScheduler: Gates slot execution based on VRAM budget

Key Invariant:
  Once a slot starts, its VRAM is locked until output is in system RAM.
  Exception: KV cache explosion triggers emergency eviction of largest consumer.

Default: 32GB VRAM budget for accessibility
"""

import gc
import time
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class VRAMPressure(Enum):
    """VRAM pressure levels."""
    NORMAL = "normal"        # < 80% - business as usual
    ELEVATED = "elevated"    # 80-90% - pause preloading
    HIGH = "high"            # 90-95% - warn, no new slots
    CRITICAL = "critical"    # > 95% - emergency eviction


@dataclass
class VRAMReservation:
    """Tracks VRAM reserved for an active slot."""
    slot_id: str
    savant_id: str
    model_mb: int
    kv_cache_mb: int          # Initial estimate
    kv_cache_actual_mb: int   # Updated during inference
    activation_mb: int
    started_at: datetime
    tokens_generated: int = 0
    
    @property
    def total_mb(self) -> int:
        """Total current VRAM usage."""
        return self.model_mb + self.kv_cache_actual_mb + self.activation_mb
    
    @property
    def initial_estimate_mb(self) -> int:
        """Initial estimated VRAM."""
        return self.model_mb + self.kv_cache_mb + self.activation_mb


@dataclass
class SavantInfo:
    """Information about a loaded savant model."""
    savant_id: str
    model: Any  # The actual model object
    model_size_mb: int
    loaded_at: datetime
    last_used: datetime
    refcount: int = 0  # Number of active slots using this savant
    
    def touch(self):
        """Update last used timestamp."""
        self.last_used = datetime.now()


@dataclass 
class HardwareProfile:
    """Hardware configuration profile."""
    name: str
    vram_budget_mb: int
    max_concurrent_slots: int
    context_length: int
    preload_enabled: bool
    
    # Pressure thresholds (fraction of budget)
    elevated_threshold: float = 0.80
    high_threshold: float = 0.90
    critical_threshold: float = 0.95


# Predefined profiles
HARDWARE_PROFILES = {
    "minimal": HardwareProfile(
        name="minimal",
        vram_budget_mb=8000,
        max_concurrent_slots=1,
        context_length=4096,
        preload_enabled=False
    ),
    "compact": HardwareProfile(
        name="compact", 
        vram_budget_mb=16000,
        max_concurrent_slots=1,
        context_length=8192,
        preload_enabled=True
    ),
    "standard": HardwareProfile(
        name="standard",
        vram_budget_mb=32000,
        max_concurrent_slots=2,
        context_length=8192,
        preload_enabled=True
    ),
    "performance": HardwareProfile(
        name="performance",
        vram_budget_mb=48000,
        max_concurrent_slots=4,
        context_length=16384,
        preload_enabled=True
    ),
    "unlimited": HardwareProfile(
        name="unlimited",
        vram_budget_mb=64000,
        max_concurrent_slots=6,
        context_length=16384,
        preload_enabled=True
    )
}


class VRAMBudget:
    """
    Tracks VRAM reservations and actual usage.
    Thread-safe for concurrent slot execution.
    """
    
    # Fixed overhead (BGE-M3, system)
    FIXED_OVERHEAD_MB = 2000
    
    # Per-token KV cache growth estimate (bytes per token per layer)
    # This is now a default - use calculate_kv_cache_bytes for model-specific estimates
    DEFAULT_KV_BYTES_PER_TOKEN = 256 * 32
    
    # Model architecture defaults (7B model with 32 layers, 4096 hidden size)
    DEFAULT_NUM_LAYERS = 32
    DEFAULT_HIDDEN_SIZE = 4096
    
    def __init__(self, profile: HardwareProfile):
        self.profile = profile
        self.vram_budget_mb = profile.vram_budget_mb
        self.available_pool_mb = self.vram_budget_mb - self.FIXED_OVERHEAD_MB
        
        self._lock = threading.RLock()
        self.reservations: Dict[str, VRAMReservation] = {}
        
        # Callbacks for pressure events
        self._on_critical: Optional[Callable[[str], None]] = None
        self._on_alert: Optional[Callable[[VRAMPressure, int, int], None]] = None  # (pressure, usage_mb, budget_mb)
        self._last_reported_pressure = VRAMPressure.NORMAL
        
        logger.info(f"VRAMBudget initialized: {self.vram_budget_mb}MB total, "
                   f"{self.available_pool_mb}MB available pool")
    
    def set_critical_callback(self, callback: Callable[[str], None]):
        """Set callback for critical pressure (receives slot_id to evict)."""
        self._on_critical = callback
    
    def set_alert_callback(self, callback: Callable[[VRAMPressure, int, int], None]):
        """
        Set callback for VRAM alerts.
        
        Args:
            callback: Function called when VRAM pressure changes.
                     Receives (pressure, usage_mb, budget_mb)
        """
        self._on_alert = callback
        
    def _notify_alert(self, pressure: VRAMPressure):
        """Notify alert callback if pressure level changed."""
        if self._on_alert and pressure != self._last_reported_pressure:
            self._on_alert(pressure, self.current_usage_mb, self.vram_budget_mb)
            self._last_reported_pressure = pressure
    
    @staticmethod
    def calculate_kv_cache_bytes(num_layers: int, hidden_size: int, num_heads: int = None) -> int:
        """
        Calculate KV cache bytes per token based on model architecture.
        
        Each token stores: 2 (K+V) × layers × hidden_size × 2 bytes (FP16)
        
        Args:
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads (optional, not used in calculation)
            
        Returns:
            Bytes per token for KV cache
        """
        # Each token stores K and V vectors, each of size hidden_size
        # For FP16: 2 bytes per parameter
        return 2 * num_layers * hidden_size * 2
    
    def get_actual_gpu_memory_mb(self) -> int:
        """
        Get actual GPU memory usage from torch.cuda if available.
        
        Returns:
            Actual allocated GPU memory in MB, or 0 if unavailable
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() // (1024 * 1024)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not get GPU memory: {e}")
        return 0
    
    def get_actual_gpu_reserved_mb(self) -> int:
        """
        Get actual GPU reserved memory from torch.cuda if available.
        
        Returns:
            Actual reserved GPU memory in MB, or 0 if unavailable
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_reserved() // (1024 * 1024)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not get GPU reserved memory: {e}")
        return 0
    
    def get_dml_memory_estimate_mb(self) -> int:
        """
        Estimate DML (DirectML) memory usage for AMD GPUs.
        
        Since DML doesn't expose VRAM metrics, we estimate based on:
        - Loaded models in savant pool
        - Active reservations
        
        Returns:
            Estimated memory usage in MB, or 0 if not using DML
        """
        # Check if we're using DML by checking onnxruntime providers
        try:
            import onnxruntime as ort
            if 'DmlExecutionProvider' in ort.get_available_providers():
                # Estimate based on reservations
                # This is a rough estimate since DML doesn't expose actual metrics
                estimated = self.current_usage_mb + self.FIXED_OVERHEAD_MB
                logger.debug(f"DML memory estimate: {estimated}MB (based on reservations)")
                return estimated
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not check DML providers: {e}")
        return 0
    
    def get_full_memory_report(self) -> Dict:
        """
        Get comprehensive memory report including actual GPU metrics.
        
        Returns:
            Dict with estimated and actual memory metrics
        """
        report = {
            "estimated_reserved_mb": self.current_usage_mb,
            "budget_mb": self.vram_budget_mb,
            "available_pool_mb": self.available_pool_mb,
            "pressure": self.pressure.value,
        }
        
        # Add actual CUDA memory if available
        actual_allocated = self.get_actual_gpu_memory_mb()
        if actual_allocated > 0:
            report["cuda_allocated_mb"] = actual_allocated
            report["cuda_over_budget"] = actual_allocated > self.vram_budget_mb
        
        # Add DML estimate if using AMD
        dml_estimate = self.get_dml_memory_estimate_mb()
        if dml_estimate > 0:
            report["dml_estimated_mb"] = dml_estimate
            report["dml_over_budget"] = dml_estimate > self.vram_budget_mb
        
        return report
    
    @property
    def current_usage_mb(self) -> int:
        """Current total VRAM usage across all reservations."""
        with self._lock:
            return sum(r.total_mb for r in self.reservations.values())
    
    @property
    def pressure(self) -> VRAMPressure:
        """Current VRAM pressure level."""
        usage_fraction = self.current_usage_mb / self.available_pool_mb
        
        if usage_fraction >= self.profile.critical_threshold:
            return VRAMPressure.CRITICAL
        elif usage_fraction >= self.profile.high_threshold:
            return VRAMPressure.HIGH
        elif usage_fraction >= self.profile.elevated_threshold:
            return VRAMPressure.ELEVATED
        else:
            return VRAMPressure.NORMAL
    
    def _estimate_model_architecture(self, model_size_mb: int) -> Tuple[int, int]:
        """
        Estimate model architecture (layers, hidden_size) from model size.
        
        Args:
            model_size_mb: Model size in MB
            
        Returns:
            Tuple of (num_layers, hidden_size)
        """
        # Rough estimates based on common model sizes
        # These are approximations for transformer architectures
        if model_size_mb < 2000:  # < 2B params
            return 24, 2048
        elif model_size_mb < 5000:  # 3-7B params
            return 32, 4096
        elif model_size_mb < 10000:  # 8-14B params
            return 40, 5120
        elif model_size_mb < 20000:  # 15-30B params
            return 48, 6144
        else:  # 70B+ params
            return 80, 8192

    def estimate_slot_vram(
        self,
        savant_size_mb: int,
        context_length: int = None,
        num_layers: int = None,
        hidden_size: int = None,
        num_heads: int = None
    ) -> Tuple[int, int, int]:
        """
        Estimate VRAM needed for a slot using model-aware KV cache calculation.
        
        Args:
            savant_size_mb: Model size in MB
            context_length: Context length to estimate (defaults to profile)
            num_layers: Number of transformer layers (defaults to estimate from model size)
            hidden_size: Hidden dimension size (defaults to estimate from model size)
            num_heads: Number of attention heads (optional, not currently used)
            
        Returns:
            (model_mb, kv_cache_mb, activation_mb)
        """
        context_length = context_length or self.profile.context_length
        
        model_mb = savant_size_mb
        
        # Estimate model architecture from model size if not provided
        if num_layers is None or hidden_size is None:
            est_layers, est_hidden = self._estimate_model_architecture(savant_size_mb)
            num_layers = num_layers or est_layers
            hidden_size = hidden_size or est_hidden
        
        # Calculate KV cache size using model-aware formula
        # Assume 50% of max context used on average
        estimated_tokens = int(context_length * 0.5)
        kv_bytes_per_token = self.calculate_kv_cache_bytes(num_layers, hidden_size)
        kv_cache_mb = int((estimated_tokens * kv_bytes_per_token) / (1024 * 1024))
        
        activation_mb = 500  # Working memory during inference
        
        logger.debug(f"VRAM estimate for slot: model={model_mb}MB, kv_cache={kv_cache_mb}MB "
                    f"({num_layers}L x {hidden_size}H, ~{estimated_tokens} tokens), "
                    f"activation={activation_mb}MB")
        
        return model_mb, kv_cache_mb, activation_mb
    
    def can_reserve(self, model_mb: int, kv_cache_mb: int, activation_mb: int) -> bool:
        """Check if we have budget for this reservation."""
        needed = model_mb + kv_cache_mb + activation_mb
        
        with self._lock:
            return (self.current_usage_mb + needed) <= self.available_pool_mb
    
    def reserve(
        self,
        slot_id: str,
        savant_id: str,
        model_mb: int,
        kv_cache_mb: int,
        activation_mb: int
    ) -> bool:
        """
        Reserve VRAM for a slot. Returns False if insufficient budget.
        """
        with self._lock:
            if not self.can_reserve(model_mb, kv_cache_mb, activation_mb):
                return False
            
            self.reservations[slot_id] = VRAMReservation(
                slot_id=slot_id,
                savant_id=savant_id,
                model_mb=model_mb,
                kv_cache_mb=kv_cache_mb,
                kv_cache_actual_mb=kv_cache_mb,  # Start with estimate
                activation_mb=activation_mb,
                started_at=datetime.now()
            )
            
            logger.debug(f"Reserved {model_mb + kv_cache_mb + activation_mb}MB for slot {slot_id}")
            return True
    
    def release(self, slot_id: str):
        """Release VRAM reservation after output stored in system RAM."""
        with self._lock:
            if slot_id in self.reservations:
                res = self.reservations.pop(slot_id)
                logger.debug(f"Released {res.total_mb}MB from slot {slot_id}")
    
    def update_kv_cache(self, slot_id: str, tokens_generated: int, num_layers: int = None, hidden_size: int = None):
        """
        Update KV cache size based on actual tokens generated.
        
        Args:
            slot_id: The slot to update
            tokens_generated: Number of tokens generated so far
            num_layers: Override for number of layers (uses DEFAULT if None)
            hidden_size: Override for hidden size (uses DEFAULT if None)
        """
        pressure_to_report = None
        largest_slot = None
        
        with self._lock:
            if slot_id not in self.reservations:
                return
            
            res = self.reservations[slot_id]
            res.tokens_generated = tokens_generated
            
            # Calculate actual KV cache size using model-aware calculation
            layers = num_layers if num_layers is not None else self.DEFAULT_NUM_LAYERS
            hidden = hidden_size if hidden_size is not None else self.DEFAULT_HIDDEN_SIZE
            kv_bytes_per_token = self.calculate_kv_cache_bytes(layers, hidden)
            
            actual_bytes = tokens_generated * kv_bytes_per_token
            res.kv_cache_actual_mb = int(actual_bytes / (1024 * 1024))
            
            # Check for pressure
            current_pressure = self.pressure
            
            if current_pressure == VRAMPressure.CRITICAL:
                pressure_to_report = current_pressure
                largest_slot = self._find_largest_kv_holder()
                logger.warning(f"CRITICAL VRAM pressure! Usage: {self.current_usage_mb}MB / {self.available_pool_mb}MB")
            elif current_pressure == VRAMPressure.HIGH:
                logger.warning(f"HIGH VRAM pressure: {self.current_usage_mb}MB / {self.available_pool_mb}MB")
        
        # Notify alert callback (outside lock)
        self._notify_alert(current_pressure)
        
        # Call callback outside the lock to prevent deadlock (thread safety fix)
        if pressure_to_report == VRAMPressure.CRITICAL and largest_slot and self._on_critical:
            logger.warning(f"Triggering emergency eviction of slot {largest_slot}")
            self._on_critical(largest_slot)
    
    def _find_largest_kv_holder(self) -> Optional[str]:
        """Find the slot with the largest KV cache."""
        if not self.reservations:
            return None
        
        largest = max(
            self.reservations.values(),
            key=lambda r: r.kv_cache_actual_mb
        )
        return largest.slot_id
    
    def get_status(self) -> Dict:
        """Get current VRAM status with actual GPU metrics."""
        with self._lock:
            status = {
                "budget_mb": self.vram_budget_mb,
                "available_pool_mb": self.available_pool_mb,
                "current_usage_mb": self.current_usage_mb,
                "pressure": self.pressure.value,
                "active_slots": len(self.reservations),
                "reservations": {
                    slot_id: {
                        "savant_id": r.savant_id,
                        "total_mb": r.total_mb,
                        "kv_cache_mb": r.kv_cache_actual_mb,
                        "tokens": r.tokens_generated
                    }
                    for slot_id, r in self.reservations.items()
                }
            }
        
        # Add actual GPU memory metrics (outside lock)
        actual_allocated = self.get_actual_gpu_memory_mb()
        if actual_allocated > 0:
            status["actual_gpu_allocated_mb"] = actual_allocated
            status["actual_vs_estimated_diff_mb"] = actual_allocated - self.current_usage_mb
        
        # Add DML estimate if using AMD
        dml_estimate = self.get_dml_memory_estimate_mb()
        if dml_estimate > 0:
            status["dml_estimated_mb"] = dml_estimate
        
        return status


class SavantPool:
    """
    Manages loading/unloading of savant models with reference counting.
    
    Models are only evicted when:
    1. refcount == 0 (no active slots using it)
    2. VRAM pressure requires it
    3. Min resident time has passed (prevents thrashing)
    
    Supports callback for DeepSeek auto-reload:
    - set_all_unloaded_callback(): Called when all models are evicted
    """
    
    MIN_RESIDENT_SECONDS = 60  # Don't evict within 60s of loading
    
    def __init__(
        self,
        vram_budget: VRAMBudget,
        model_loader: Callable[[str], Tuple[Any, int]] = None  # Returns (model, size_mb)
    ):
        self.vram_budget = vram_budget
        self.model_loader = model_loader or self._default_loader
        
        self._lock = threading.RLock()
        self.loaded: Dict[str, SavantInfo] = {}
        self.preload_queue: List[str] = []
        
        # Track models pending eviction for retry
        self.pending_retry: Dict[str, dict] = {}  # slot_id -> retry_info
        
        # Callback for when all models are unloaded (for DeepSeek auto-reload)
        self._on_all_unloaded: Optional[Callable[[], None]] = None
    
    def _default_loader(self, savant_id: str) -> Tuple[Any, int]:
        """Default loader for DML quantized models."""
        logger.warning(f"Using placeholder loader for {savant_id}")
        return None, 5000  # DML models use ~5GB with INT4 quantization
    
    def acquire(self, savant_id: str, slot_id: str) -> Optional[Any]:
        """
        Acquire a savant model for a slot.
        Loads if not already loaded. Increments refcount.
        
        Returns:
            Model object, or None if cannot load
        """
        with self._lock:
            if savant_id in self.loaded:
                # Already loaded - just bump refcount
                info = self.loaded[savant_id]
                info.refcount += 1
                info.touch()
                logger.debug(f"Acquired {savant_id} for {slot_id} (refcount={info.refcount})")
                return info.model
            
            # Need to load - check if we need to evict first
            self._evict_if_needed()
            
            # Load the model
            try:
                model, size_mb = self.model_loader(savant_id)
                
                self.loaded[savant_id] = SavantInfo(
                    savant_id=savant_id,
                    model=model,
                    model_size_mb=size_mb,
                    loaded_at=datetime.now(),
                    last_used=datetime.now(),
                    refcount=1
                )
                
                logger.info(f"Loaded savant {savant_id} ({size_mb}MB) for {slot_id}")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load savant {savant_id}: {e}")
                return None
    
    def release(self, savant_id: str, slot_id: str):
        """
        Release a savant model from a slot.
        Decrements refcount. Model may be evicted later if refcount=0.
        """
        with self._lock:
            if savant_id not in self.loaded:
                return
            
            info = self.loaded[savant_id]
            info.refcount = max(0, info.refcount - 1)
            info.touch()
            
            logger.debug(f"Released {savant_id} from {slot_id} (refcount={info.refcount})")
    
    def preload(self, savant_id: str):
        """Queue a savant for background preloading."""
        if not self.vram_budget.profile.preload_enabled:
            return
        
        if self.vram_budget.pressure in (VRAMPressure.HIGH, VRAMPressure.CRITICAL):
            logger.debug(f"Skipping preload of {savant_id} due to VRAM pressure")
            return
        
        with self._lock:
            if savant_id not in self.loaded and savant_id not in self.preload_queue:
                self.preload_queue.append(savant_id)
                logger.debug(f"Queued {savant_id} for preload")
    
    async def process_preload_queue(self):
        """Process preload queue (call from background task)."""
        while self.preload_queue:
            if self.vram_budget.pressure != VRAMPressure.NORMAL:
                break  # Don't preload under pressure
            
            with self._lock:
                if not self.preload_queue:
                    break
                savant_id = self.preload_queue.pop(0)
            
            if savant_id not in self.loaded:
                # Preload with refcount=0 (not actively used yet)
                try:
                    model, size_mb = self.model_loader(savant_id)
                    
                    with self._lock:
                        self.loaded[savant_id] = SavantInfo(
                            savant_id=savant_id,
                            model=model,
                            model_size_mb=size_mb,
                            loaded_at=datetime.now(),
                            last_used=datetime.now(),
                            refcount=0  # Not actively used
                        )
                    
                    logger.info(f"Preloaded savant {savant_id} ({size_mb}MB)")
                    
                except Exception as e:
                    logger.warning(f"Preload failed for {savant_id}: {e}")
            
            await asyncio.sleep(0.1)  # Yield between loads
    
    def _evict_if_needed(self):
        """Evict idle models if approaching VRAM limits."""
        pressure = self.vram_budget.pressure
        
        if pressure == VRAMPressure.NORMAL:
            return
        
        # Find eviction candidates (refcount=0, past min resident time)
        now = datetime.now()
        candidates = [
            info for info in self.loaded.values()
            if info.refcount == 0 and 
               (now - info.loaded_at).total_seconds() > self.MIN_RESIDENT_SECONDS
        ]
        
        # Sort by last used (LRU)
        candidates.sort(key=lambda i: i.last_used)
        
        for info in candidates:
            if self.vram_budget.pressure == VRAMPressure.NORMAL:
                break  # Pressure relieved
            
            self._evict_model(info.savant_id)
    
    def _evict_model(self, savant_id: str) -> bool:
        """
        Evict a model from VRAM.
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if savant_id not in self.loaded:
                return False
            
            info = self.loaded.pop(savant_id)
            model_size_mb = info.model_size_mb
            
            # Get memory before cleanup for verification
            mem_before_mb = self.vram_budget.get_actual_gpu_memory_mb()
            
            # Release the model reference
            try:
                del info.model
            except Exception as e:
                logger.warning(f"Error deleting model object: {e}")
            
            # Force garbage collection
            gc.collect()
            
            # Try to clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                    # Verify VRAM was freed
                    mem_after_mb = self.vram_budget.get_actual_gpu_memory_mb()
                    freed_mb = mem_before_mb - mem_after_mb
                    if freed_mb > 0:
                        logger.info(f"Evicted savant {savant_id} ({model_size_mb}MB), VRAM freed: ~{freed_mb}MB")
                    else:
                        logger.info(f"Evicted savant {savant_id} ({model_size_mb}MB), VRAM change: {freed_mb}MB (may be cached)")
            except ImportError:
                logger.debug("torch not available for VRAM cleanup")
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache during eviction: {e}")
            
            # Check if all models are now unloaded and trigger callback
            if not self.loaded and self._on_all_unloaded:
                logger.info("All savant models evicted - triggering all_unloaded callback")
                # Call callback outside the lock to prevent deadlock
                callback = self._on_all_unloaded
                threading.Thread(target=callback, daemon=True).start()
            
            return True
    
    def force_evict_for_slot(self, slot_id: str) -> Optional[str]:
        """
        Emergency eviction: forcibly evict the model for a slot.
        Used when KV cache explosion triggers critical pressure.
        
        Returns:
            savant_id that was evicted, or None
        """
        with self._lock:
            # Find which savant this slot was using
            reservation = self.vram_budget.reservations.get(slot_id)
            if not reservation:
                return None
            
            savant_id = reservation.savant_id
            
            if savant_id in self.loaded:
                info = self.loaded[savant_id]
                info.refcount = max(0, info.refcount - 1)
                
                # Record for retry
                self.pending_retry[slot_id] = {
                    "savant_id": savant_id,
                    "evicted_at": datetime.now(),
                    "reason": "kv_cache_explosion",
                    "kv_cache_mb": reservation.kv_cache_actual_mb,
                    "tokens_generated": reservation.tokens_generated
                }
                
                # Evict if no other users
                if info.refcount == 0:
                    self._evict_model(savant_id)
                
                return savant_id
            
            return None
    
    def get_retry_slots(self) -> List[str]:
        """Get slot IDs that need to be retried after eviction."""
        return list(self.pending_retry.keys())
    
    def clear_retry(self, slot_id: str):
        """Clear retry status for a slot (after successful retry)."""
        self.pending_retry.pop(slot_id, None)
    
    def set_all_unloaded_callback(self, callback: Callable[[], None]):
        """
        Set callback for when all models have been unloaded.
        
        This is used to auto-reload DeepSeek for follow-up queries
        after all savant models have been evicted.
        
        Args:
            callback: Function called with no arguments when pool becomes empty
        """
        with self._lock:
            self._on_all_unloaded = callback
    
    def get_status(self) -> Dict:
        """Get pool status."""
        with self._lock:
            return {
                "loaded_count": len(self.loaded),
                "preload_queue": len(self.preload_queue),
                "pending_retry": len(self.pending_retry),
                "models": {
                    sid: {
                        "size_mb": info.model_size_mb,
                        "refcount": info.refcount,
                        "age_seconds": (datetime.now() - info.loaded_at).total_seconds()
                    }
                    for sid, info in self.loaded.items()
                }
            }


class VRAMAwareScheduler:
    """
    VRAM-aware slot scheduler with emergency eviction handling.
    
    Key behaviors:
    1. Only starts slots when VRAM reservation is possible
    2. Monitors KV cache growth during inference
    3. Triggers emergency eviction on critical pressure
    4. Reschedules evicted slots for retry
    """
    
    def __init__(
        self,
        profile: HardwareProfile = None,
        model_loader: Callable[[str], Tuple[Any, int]] = None
    ):
        self.profile = profile or HARDWARE_PROFILES["standard"]
        
        self.vram_budget = VRAMBudget(self.profile)
        self.savant_pool = SavantPool(self.vram_budget, model_loader)
        
        # Wire up emergency eviction callback
        self.vram_budget.set_critical_callback(self._handle_critical_pressure)
        
        # Wire up alert callback
        self.vram_budget.set_alert_callback(self._handle_alert)
        
        # Retry queue
        self._retry_queue: List[str] = []
        self._retry_lock = threading.Lock()
    
    def _handle_critical_pressure(self, slot_id: str):
        """Handle critical VRAM pressure by evicting and rescheduling."""
        logger.warning(f"Emergency eviction triggered for slot {slot_id}")
        
        # Force evict
        evicted_savant = self.savant_pool.force_evict_for_slot(slot_id)
        
        if evicted_savant:
            # Release the VRAM reservation
            self.vram_budget.release(slot_id)
            
            # Queue for retry
            with self._retry_lock:
                if slot_id not in self._retry_queue:
                    self._retry_queue.append(slot_id)
            
            logger.info(f"Slot {slot_id} queued for retry after eviction")
    
    def _handle_alert(self, pressure: VRAMPressure, usage_mb: int, budget_mb: int):
        """Handle VRAM alert callback."""
        logger.info(f"VRAM ALERT: {pressure.value} - Usage: {usage_mb}MB / {budget_mb}MB")
    
    def get_safe_context_length(self, input_tokens: int = 0, max_output: int = 1024) -> int:
        """
        Get safe context length based on current VRAM pressure.
        
        Args:
            input_tokens: Number of input tokens
            max_output: Maximum output tokens desired
            
        Returns:
            Safe total context length
        """
        pressure = self.vram_budget.pressure
        
        # Base limits per pressure level
        if pressure == VRAMPressure.CRITICAL:
            max_total = 1024
        elif pressure == VRAMPressure.HIGH:
            max_total = 2048
        elif pressure == VRAMPressure.ELEVATED:
            max_total = 2560
        else:  # NORMAL
            max_total = 3072
        
        # Apply input token offset
        safe_total = min(max_total, input_tokens + max_output)
        return safe_total
    
    def can_start_slot(self, savant_id: str, savant_size_mb: int) -> bool:
        """Check if we have VRAM budget to start a slot."""
        model_mb, kv_cache_mb, activation_mb = self.vram_budget.estimate_slot_vram(
            savant_size_mb
        )
        return self.vram_budget.can_reserve(model_mb, kv_cache_mb, activation_mb)
    
    def start_slot(
        self,
        slot_id: str,
        savant_id: str,
        savant_size_mb: int
    ) -> Optional[Any]:
        """
        Start a slot: reserve VRAM, acquire model.
        
        Returns:
            Model object, or None if cannot start
        """
        model_mb, kv_cache_mb, activation_mb = self.vram_budget.estimate_slot_vram(
            savant_size_mb
        )
        
        # Try to reserve VRAM
        if not self.vram_budget.reserve(slot_id, savant_id, model_mb, kv_cache_mb, activation_mb):
            logger.warning(f"Cannot reserve VRAM for slot {slot_id}: insufficient budget. "
                          f"Need {model_mb + kv_cache_mb + activation_mb}MB, "
                          f"have {self.vram_budget.available_pool_mb - self.vram_budget.current_usage_mb}MB available")
            return None
        
        # Check if model is already loaded (fast path)
        if savant_id in self.savant_pool.loaded:
            logger.debug(f"Model {savant_id} already loaded, acquiring reference")
            model = self.savant_pool.acquire(savant_id, slot_id)
            return model
        
        # Acquire model - this may trigger loading if not already loaded
        model = self.savant_pool.acquire(savant_id, slot_id)
        
        if model is None:
            # Failed to load - release reservation
            self.vram_budget.release(slot_id)
            logger.error(f"Failed to load model {savant_id} for slot {slot_id}")
            return None
        
        return model
    
    def update_slot_progress(self, slot_id: str, tokens_generated: int):
        """Update slot progress (call periodically during inference)."""
        self.vram_budget.update_kv_cache(slot_id, tokens_generated)
    
    def complete_slot(self, slot_id: str, savant_id: str):
        """
        Complete a slot: output is now in system RAM.
        Releases VRAM reservation and model reference.
        """
        # Release model reference
        self.savant_pool.release(savant_id, slot_id)
        
        # Release VRAM reservation
        self.vram_budget.release(slot_id)
        
        # Clear any retry status
        self.savant_pool.clear_retry(slot_id)
        
        logger.debug(f"Slot {slot_id} completed and VRAM released")
    
    def get_retry_queue(self) -> List[str]:
        """Get slots that need to be retried after eviction."""
        with self._retry_lock:
            return self._retry_queue.copy()
    
    def clear_retry(self, slot_id: str):
        """Clear a slot from retry queue after successful retry."""
        with self._retry_lock:
            if slot_id in self._retry_queue:
                self._retry_queue.remove(slot_id)
        self.savant_pool.clear_retry(slot_id)
    
    def preload_for_upcoming(self, savant_ids: List[str]):
        """Preload savants for upcoming slots (from DAG lookahead)."""
        for savant_id in savant_ids:
            self.savant_pool.preload(savant_id)
    
    async def run_preload_loop(self):
        """Background task to process preload queue."""
        while True:
            await self.savant_pool.process_preload_queue()
            await asyncio.sleep(1.0)
    
    def get_status(self) -> Dict:
        """Get comprehensive status."""
        return {
            "profile": self.profile.name,
            "vram": self.vram_budget.get_status(),
            "pool": self.savant_pool.get_status(),
            "retry_queue": self.get_retry_queue()
        }


def get_profile_for_vram(vram_mb: int) -> HardwareProfile:
    """Select appropriate profile based on available VRAM."""
    if vram_mb >= 64000:
        return HARDWARE_PROFILES["unlimited"]
    elif vram_mb >= 48000:
        return HARDWARE_PROFILES["performance"]
    elif vram_mb >= 32000:
        return HARDWARE_PROFILES["standard"]
    elif vram_mb >= 16000:
        return HARDWARE_PROFILES["compact"]
    else:
        return HARDWARE_PROFILES["minimal"]


def detect_vram() -> int:
    """Auto-detect available VRAM in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory // (1024 * 1024)
    except ImportError:
        pass
    
    # Try DirectML
    try:
        import onnxruntime as ort
        # DML doesn't expose VRAM directly, assume 32GB default
        if 'DmlExecutionProvider' in ort.get_available_providers():
            return 32000  # Conservative default
    except ImportError:
        pass
    
    # Fallback to standard profile
    logger.warning("Could not detect VRAM, using 32GB default")
    return 32000


def create_scheduler(
    vram_limit_mb: Optional[int] = None,
    profile_name: Optional[str] = None,
    model_loader: Callable[[str], Tuple[Any, int]] = None
) -> VRAMAwareScheduler:
    """
    Factory function to create a VRAM-aware scheduler.
    
    Args:
        vram_limit_mb: Override VRAM limit (auto-detect if None)
        profile_name: Use specific profile (auto-select if None)
        model_loader: Custom model loading function
        
    Returns:
        Configured VRAMAwareScheduler
    """
    if profile_name and profile_name in HARDWARE_PROFILES:
        profile = HARDWARE_PROFILES[profile_name]
    elif vram_limit_mb:
        profile = get_profile_for_vram(vram_limit_mb)
        # Override with actual limit
        profile = HardwareProfile(
            name=f"custom_{vram_limit_mb}",
            vram_budget_mb=vram_limit_mb,
            max_concurrent_slots=profile.max_concurrent_slots,
            context_length=profile.context_length,
            preload_enabled=profile.preload_enabled
        )
    else:
        detected = detect_vram()
        profile = get_profile_for_vram(detected)
    
    logger.info(f"Creating scheduler with profile: {profile.name} ({profile.vram_budget_mb}MB)")
    
    return VRAMAwareScheduler(profile, model_loader)


if __name__ == "__main__":
    # Test the VRAM manager
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 60)
    print("VRAM Manager Test")
    print("=" * 60)
    
    # Create scheduler with standard profile
    scheduler = create_scheduler(profile_name="standard")
    
    print(f"\nProfile: {scheduler.profile.name}")
    print(f"VRAM Budget: {scheduler.profile.vram_budget_mb}MB")
    print(f"Max Concurrent Slots: {scheduler.profile.max_concurrent_slots}")
    
    # Simulate starting some slots
    print("\n--- Simulating slot execution ---")
    
    # Slot 1
    model1 = scheduler.start_slot("slot_1", "qwen_coder", 7000)
    print(f"Slot 1 started: {model1 is not None}")
    print(f"Status: {scheduler.vram_budget.get_status()}")
    
    # Slot 2
    model2 = scheduler.start_slot("slot_2", "law_llm", 7000)
    print(f"Slot 2 started: {model2 is not None}")
    print(f"Status: {scheduler.vram_budget.get_status()}")
    
    # Slot 3 - might fail depending on estimates
    model3 = scheduler.start_slot("slot_3", "bio_mistral", 7000)
    print(f"Slot 3 started: {model3 is not None}")
    print(f"Status: {scheduler.vram_budget.get_status()}")
    
    # Simulate KV cache growth
    print("\n--- Simulating KV cache growth ---")
    for tokens in [1000, 2000, 4000, 8000]:
        scheduler.update_slot_progress("slot_1", tokens)
        print(f"Tokens: {tokens}, Pressure: {scheduler.vram_budget.pressure.value}")
    
    # Complete slots
    print("\n--- Completing slots ---")
    scheduler.complete_slot("slot_1", "qwen_coder")
    scheduler.complete_slot("slot_2", "law_llm")
    
    print(f"Final status: {scheduler.get_status()}")

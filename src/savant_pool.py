"""
Savant Pool - OGA Model Management with VRAM-Aware Scheduling

Manages model loading/unloading and integrates with VRAMAwareScheduler.
"""

import json
import gc
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable

import onnxruntime_genai as og


class SavantPool:
    """
    Extended SavantPool that integrates with VRAMAwareScheduler.
    Wraps the scheduler's pool and adds OGA-specific model loading.
    """
    
    def __init__(self, scheduler_pool=None):
        self.scheduler_pool = scheduler_pool
        self._tokenizers: Dict[str, Any] = {}       # savant_id -> tokenizer
        self._context_lengths: Dict[str, int] = {}  # savant_id -> context limit
        self._model_paths: Dict[str, str] = {}      # savant_id -> model_path
        self._path_to_savant: Dict[str, str] = {}   # model_path -> first savant_id
        self._lock = threading.RLock()
        
        # Callback for when all models are unloaded (for DeepSeek auto-reload)
        self._on_all_unloaded: Optional[Callable[[], None]] = None
    
    @property
    def loaded(self):
        """Access scheduler's loaded models."""
        if self.scheduler_pool:
            return {sid: info.model for sid, info in self.scheduler_pool.loaded.items()}
        return {}
    
    @property
    def loaded_info(self):
        """Access scheduler's loaded SavantInfo objects (for refcount, etc.)."""
        if self.scheduler_pool:
            return self.scheduler_pool.loaded
        return {}
    
    def is_loaded(self, savant_id: str) -> bool:
        """Check if model is loaded in scheduler's pool."""
        if self.scheduler_pool:
            return savant_id in self.scheduler_pool.loaded
        return savant_id in self._tokenizers
    
    def get_savant_for_model(self, model_path: str) -> Optional[str]:
        """Check if this model path is already loaded."""
        return self._path_to_savant.get(model_path)
    
    def load(self, savant_id: str, model_path: str, model_size_mb: int = 5000):
        """Load model via OGA and register with scheduler."""
        with self._lock:
            # Check if already loaded
            if self.is_loaded(savant_id):
                return
            
            # CHECK IF THIS MODEL IS ALREADY LOADED UNDER A DIFFERENT SAVANT
            existing_savant = self.get_savant_for_model(model_path)
            if existing_savant and self.is_loaded(existing_savant):
                # REUSE the existing model!
                print(f"    [SavantPool] REUSING {model_path} (already loaded as {existing_savant}) for {savant_id}")
                self._tokenizers[savant_id] = self._tokenizers[existing_savant]
                self._context_lengths[savant_id] = self._context_lengths[existing_savant]
                self._model_paths[savant_id] = model_path
                
                # Register with scheduler's pool as alias
                if self.scheduler_pool and existing_savant in self.scheduler_pool.loaded:
                    from .vram_manager import SavantInfo
                    existing_info = self.scheduler_pool.loaded[existing_savant]
                    self.scheduler_pool.loaded[savant_id] = SavantInfo(
                        savant_id=savant_id,
                        model=existing_info.model,
                        model_size_mb=existing_info.model_size_mb,
                        loaded_at=existing_info.loaded_at,
                        last_used=datetime.now(),
                        refcount=0
                    )
                return
            
            # Load new model
            print(f"    [SavantPool] Loading {savant_id} from {model_path}")
            model = og.Model(model_path)
            tokenizer = og.Tokenizer(model)
            
            # Read context_length from genai_config.json
            config_path = Path(model_path) / "genai_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    self._context_lengths[savant_id] = config.get("model", {}).get("context_length", 8192)
                    print(f"    [SavantPool] Context length: {self._context_lengths[savant_id]}")
            else:
                self._context_lengths[savant_id] = 8192
            
            # Track mappings
            self._tokenizers[savant_id] = tokenizer
            self._model_paths[savant_id] = model_path
            self._path_to_savant[model_path] = savant_id
            
            # Register with scheduler's pool
            if self.scheduler_pool:
                from .vram_manager import SavantInfo
                self.scheduler_pool.loaded[savant_id] = SavantInfo(
                    savant_id=savant_id,
                    model=model,
                    model_size_mb=model_size_mb,
                    loaded_at=datetime.now(),
                    last_used=datetime.now(),
                    refcount=0
                )
    
    def unload(self, savant_id: str):
        """Unload model from pool."""
        with self._lock:
            if savant_id not in self._tokenizers:
                return
            
            model_path = self._model_paths.get(savant_id)
            is_owner = self._path_to_savant.get(model_path) == savant_id
            
            # Find other savants using this same model path
            other_users = [sid for sid, path in self._model_paths.items() 
                           if path == model_path and sid != savant_id and sid in self._tokenizers]
            
            if is_owner and other_users:
                # Transfer ownership
                new_owner = other_users[0]
                self._path_to_savant[model_path] = new_owner
                print(f"    [SavantPool] Releasing {savant_id} (ownership transferred to {new_owner})")
            
            # Remove references
            del self._tokenizers[savant_id]
            if savant_id in self._context_lengths:
                del self._context_lengths[savant_id]
            if savant_id in self._model_paths:
                del self._model_paths[savant_id]
            
            # Remove from scheduler pool
            if self.scheduler_pool and savant_id in self.scheduler_pool.loaded:
                del self.scheduler_pool.loaded[savant_id]
            
            # Cleanup if owner
            if is_owner and not other_users:
                print(f"    [SavantPool] Unloading {savant_id} from {model_path}")
                if model_path in self._path_to_savant:
                    del self._path_to_savant[model_path]
                gc.collect()
            elif not is_owner:
                print(f"    [SavantPool] Releasing alias {savant_id}")
            
            # Check if all models are now unloaded and trigger callback
            if not self._tokenizers and self._on_all_unloaded:
                print("    [SavantPool] All models unloaded - triggering callback")
                # Call callback in a separate thread to avoid blocking
                threading.Thread(target=self._on_all_unloaded, daemon=True).start()
    
    def get_lru(self) -> Optional[str]:
        """Get least recently used savant."""
        if not self.scheduler_pool or not self.scheduler_pool.loaded:
            return None
        return min(
            self.scheduler_pool.loaded.keys(),
            key=lambda s: self.scheduler_pool.loaded[s].last_used
        )
    
    def evict_lru(self):
        """Evict least recently used model."""
        lru = self.get_lru()
        if lru:
            self.unload(lru)
    
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
    
    def get_tokenizer(self, savant_id: str) -> Optional[Any]:
        """Get tokenizer for a savant."""
        return self._tokenizers.get(savant_id)
    
    def get_context_length(self, savant_id: str) -> int:
        """Get context length for a savant."""
        return self._context_lengths.get(savant_id, 8192)
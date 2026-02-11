"""
Crosstalk Bus V8 - A2A-Aligned Expert-to-Expert Communication Protocol

Part of College of Experts V8 Demo

Implements Google A2A-style structured messaging for zero-copy, efficient
communication between the NPU-resident Executive and GPU-resident Savants.
"""

import threading
import queue
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Union, Any
from datetime import datetime
from enum import Enum


class MessageType(Enum):
    REQUEST = "request"      # Expert requests input from another
    RESPONSE = "response"    # Expert responds to a request
    NOTIFY = "notify"        # Expert broadcasts information
    CRITIQUE = "critique"    # Expert critiques another's work


class PartType(Enum):
    """A2A-style message part types."""
    TEXT = "text"            # Natural language (requires tokenization for LLM)
    DATA = "data"            # Structured data (NO tokenization - direct Python access)
    ARTIFACT = "artifact"    # Reference to memory item (NO tokenization - just pointer)
    CODE = "code"            # Code snippet with language tag
    FILE = "file"            # File reference


@dataclass
class MessagePart:
    """
    A2A-style message part.
    
    Text parts are for human/LLM readable content (requires tokenization).
    Data parts are structured and bypass tokenization entirely.
    Artifact parts are references to episodic memory items.
    """
    type: PartType
    content: Union[str, dict, list]  # Text string, structured data, or artifact ID
    metadata: Optional[dict] = None  # Optional metadata (language for code, etc.)
    
    def is_tokenization_required(self) -> bool:
        """Check if this part requires tokenization for LLM processing."""
        return self.type == PartType.TEXT
    
    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MessagePart":
        """Create from dict."""
        return cls(
            type=PartType(data["type"]),
            content=data["content"],
            metadata=data.get("metadata")
        )
    
    @classmethod
    def text(cls, content: str) -> "MessagePart":
        """Create a text part."""
        return cls(type=PartType.TEXT, content=content)
    
    @classmethod
    def data(cls, content: Union[dict, list], label: str = None) -> "MessagePart":
        """Create a structured data part (NO tokenization!)."""
        return cls(type=PartType.DATA, content=content, metadata={"label": label} if label else None)
    
    @classmethod
    def artifact(cls, artifact_id: str) -> "MessagePart":
        """Create an artifact reference part."""
        return cls(type=PartType.ARTIFACT, content=artifact_id)
    
    @classmethod
    def code(cls, content: str, language: str = "python") -> "MessagePart":
        """Create a code part."""
        return cls(type=PartType.CODE, content=content, metadata={"language": language})


@dataclass
class CrosstalkMessage:
    """
    A2A-aligned message between experts.
    
    Uses multi-part structure to minimize tokenization:
    - text parts: Require tokenization (keep minimal)
    - data parts: Direct Python access (NO tokenization)
    - artifact parts: Memory references (NO tokenization)
    """
    id: str
    from_expert: str
    to_expert: str  # "broadcast" for all experts
    msg_type: MessageType
    parts: List[MessagePart]  # A2A-style multi-part content
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None  # ID of message being replied to
    context_ref: Optional[str] = None  # Backward compatibility
    
    # Backward compatibility: content property for text-only access
    @property
    def content(self) -> str:
        """Get text content (for backward compatibility)."""
        text_parts = [p.content for p in self.parts if p.type == PartType.TEXT]
        return " ".join(text_parts) if text_parts else ""
    
    def get_text_parts(self) -> List[str]:
        """Get all text parts (requires tokenization)."""
        return [p.content for p in self.parts if p.type == PartType.TEXT]
    
    def get_data_parts(self) -> List[dict]:
        """Get all structured data parts (NO tokenization!)."""
        return [p.content for p in self.parts if p.type == PartType.DATA]
    
    def get_artifact_refs(self) -> List[str]:
        """Get all artifact references (NO tokenization!)."""
        return [p.content for p in self.parts if p.type == PartType.ARTIFACT]
    
    def get_total_text_tokens_estimate(self) -> int:
        """Estimate tokens needed (only text parts)."""
        text = self.content
        # Rough estimate: 1 token per 4 chars
        return len(text) // 4
    
    def tokenization_fraction(self) -> float:
        """Fraction of content that requires tokenization."""
        if not self.parts:
            return 0.0
        text_parts = sum(1 for p in self.parts if p.type == PartType.TEXT)
        return text_parts / len(self.parts)
    
    def is_broadcast(self) -> bool:
        return self.to_expert == "broadcast"
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": self.id,
            "from_expert": self.from_expert,
            "to_expert": self.to_expert,
            "msg_type": self.msg_type.value,
            "parts": [p.to_dict() for p in self.parts],
            "timestamp": self.timestamp.isoformat(),
            "reply_to": self.reply_to,
            "context_ref": self.context_ref
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CrosstalkMessage":
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            from_expert=data["from_expert"],
            to_expert=data["to_expert"],
            msg_type=MessageType(data["msg_type"]),
            parts=[MessagePart.from_dict(p) for p in data["parts"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            reply_to=data.get("reply_to"),
            context_ref=data.get("context_ref")
        )
    
    # --- Factory methods for common patterns ---
    
    @classmethod
    def create_text_only(
        cls,
        msg_id: str,
        from_expert: str,
        to_expert: str,
        msg_type: MessageType,
        text: str
    ) -> "CrosstalkMessage":
        """Create a simple text-only message (backward compatible)."""
        return cls(
            id=msg_id,
            from_expert=from_expert,
            to_expert=to_expert,
            msg_type=msg_type,
            parts=[MessagePart.text(text)]
        )
    
    @classmethod
    def create_structured(
        cls,
        msg_id: str,
        from_expert: str,
        to_expert: str,
        msg_type: MessageType,
        summary: str,  # Short text (tokenized)
        data: dict     # Structured data (NOT tokenized)
    ) -> "CrosstalkMessage":
        """
        Create a message with minimal text and structured data.
        
        This is the preferred format for efficiency:
        - summary: Brief text for LLM context (~10-20 tokens)
        - data: Detailed structured info (NO tokenization!)
        """
        return cls(
            id=msg_id,
            from_expert=from_expert,
            to_expert=to_expert,
            msg_type=msg_type,
            parts=[
                MessagePart.text(summary),
                MessagePart.data(data)
            ]
        )


class CrosstalkBus:
    """
    A2A-aligned message bus for expert-to-expert communication.
    
    Features:
    - Multi-part messages (text, data, artifacts)
    - Structured data bypasses tokenization
    - Async message queuing per expert
    - Broadcast support
    - Request/response correlation
    - Statistics tracking
    """
    
    def __init__(self, max_queue_size: int = 100):
        self._queues: Dict[str, queue.Queue] = {}
        self._history: List[CrosstalkMessage] = []
        self._lock = threading.RLock()
        self._max_queue_size = max_queue_size
        self._message_counter = 0
        
        # Stats for optimization tuning
        self._stats = {
            "messages_sent": 0,
            "text_parts": 0,
            "data_parts": 0,
            "artifact_parts": 0,
            "estimated_tokens_saved": 0
        }
    
    def register_expert(self, expert_id: str) -> None:
        """Register an expert to receive messages."""
        with self._lock:
            if expert_id not in self._queues:
                self._queues[expert_id] = queue.Queue(maxsize=self._max_queue_size)
    
    def unregister_expert(self, expert_id: str) -> None:
        """Unregister an expert (clears their queue)."""
        with self._lock:
            if expert_id in self._queues:
                del self._queues[expert_id]
    
    def _generate_id(self) -> str:
        """Generate unique message ID."""
        with self._lock:
            self._message_counter += 1
            return f"msg_{self._message_counter:06d}"
    
    def _update_stats(self, message: CrosstalkMessage) -> None:
        """Update statistics for optimization tracking."""
        self._stats["messages_sent"] += 1
        for part in message.parts:
            if part.type == PartType.TEXT:
                self._stats["text_parts"] += 1
            elif part.type == PartType.DATA:
                self._stats["data_parts"] += 1
                # Estimate tokens saved by not tokenizing data
                data_str = json.dumps(part.content)
                self._stats["estimated_tokens_saved"] += len(data_str) // 4
            elif part.type == PartType.ARTIFACT:
                self._stats["artifact_parts"] += 1
                self._stats["estimated_tokens_saved"] += 50  # Average artifact tokens
    
    def send(self, message: CrosstalkMessage) -> str:
        """
        Send a message to another expert.
        
        Returns:
            Message ID for tracking
        """
        if not message.id:
            message.id = self._generate_id()
        
        with self._lock:
            self._history.append(message)
            self._update_stats(message)
            
            if message.is_broadcast():
                # Send to all registered experts except sender
                for expert_id, q in self._queues.items():
                    if expert_id != message.from_expert:
                        try:
                            q.put_nowait(message)
                        except queue.Full:
                            pass  # Drop if queue full
            else:
                # Send to specific expert
                if message.to_expert in self._queues:
                    try:
                        self._queues[message.to_expert].put_nowait(message)
                    except queue.Full:
                        pass  # Drop if queue full
        
        return message.id
    
    def receive(
        self, 
        expert_id: str, 
        timeout: float = 0.0,
        msg_type: Optional[MessageType] = None
    ) -> List[CrosstalkMessage]:
        """
        Receive pending messages for an expert.
        
        Args:
            expert_id: The receiving expert
            timeout: Seconds to wait (0 = non-blocking)
            msg_type: Filter by message type (None = all)
        
        Returns:
            List of messages (may be empty)
        """
        messages = []
        
        if expert_id not in self._queues:
            return messages
        
        q = self._queues[expert_id]
        deadline = time.time() + timeout
        
        while True:
            remaining = max(0, deadline - time.time()) if timeout > 0 else 0
            
            try:
                msg = q.get(timeout=remaining if remaining > 0 else 0.001)
                if msg_type is None or msg.msg_type == msg_type:
                    messages.append(msg)
                elif msg_type is not None and msg.msg_type != msg_type:
                    try:
                        q.put_nowait(msg)
                    except queue.Full:
                        pass
            except queue.Empty:
                break
            
            if timeout == 0:
                continue
            elif time.time() >= deadline:
                break
        
        return messages
    
    # --- A2A-style convenience methods ---
    
    def send_structured(
        self,
        from_expert: str,
        to_expert: str,
        summary: str,
        data: dict,
        msg_type: MessageType = MessageType.NOTIFY
    ) -> str:
        """
        Send a structured message (minimal tokenization).
        
        Args:
            summary: Short text description (~10-20 tokens)
            data: Structured data (NOT tokenized!)
        
        Returns:
            Message ID
        """
        message = CrosstalkMessage.create_structured(
            msg_id=self._generate_id(),
            from_expert=from_expert,
            to_expert=to_expert,
            msg_type=msg_type,
            summary=summary,
            data=data
        )
        return self.send(message)
    
    def broadcast(
        self, 
        from_expert: str, 
        content: str,
        msg_type: MessageType = MessageType.NOTIFY,
        data: Optional[dict] = None
    ) -> str:
        """
        Broadcast a message to all experts.
        
        Args:
            content: Text summary
            data: Optional structured data
        
        Returns:
            Message ID
        """
        parts = [MessagePart.text(content)]
        if data:
            parts.append(MessagePart.data(data))
        
        message = CrosstalkMessage(
            id=self._generate_id(),
            from_expert=from_expert,
            to_expert="broadcast",
            msg_type=msg_type,
            parts=parts
        )
        return self.send(message)
    
    def request(
        self,
        from_expert: str,
        to_expert: str,
        content: str,
        data: Optional[dict] = None,
        artifact_ref: Optional[str] = None
    ) -> str:
        """
        Send a request to another expert.
        
        Args:
            content: Text request
            data: Optional structured context
            artifact_ref: Optional memory item reference
        
        Returns:
            Message ID (use for correlating response)
        """
        parts = [MessagePart.text(content)]
        if data:
            parts.append(MessagePart.data(data))
        if artifact_ref:
            parts.append(MessagePart.artifact(artifact_ref))
        
        message = CrosstalkMessage(
            id=self._generate_id(),
            from_expert=from_expert,
            to_expert=to_expert,
            msg_type=MessageType.REQUEST,
            parts=parts
        )
        return self.send(message)
    
    def respond(
        self,
        from_expert: str,
        to_expert: str,
        content: str,
        reply_to: str,
        data: Optional[dict] = None,
        artifact_ref: Optional[str] = None
    ) -> str:
        """
        Send a response to a previous request.
        
        Returns:
            Message ID
        """
        parts = [MessagePart.text(content)]
        if data:
            parts.append(MessagePart.data(data))
        if artifact_ref:
            parts.append(MessagePart.artifact(artifact_ref))
        
        message = CrosstalkMessage(
            id=self._generate_id(),
            from_expert=from_expert,
            to_expert=to_expert,
            msg_type=MessageType.RESPONSE,
            parts=parts,
            reply_to=reply_to
        )
        return self.send(message)
    
    def get_history(
        self, 
        limit: int = 100,
        expert_id: Optional[str] = None
    ) -> List[CrosstalkMessage]:
        """Get message history for debugging."""
        with self._lock:
            history = self._history[-limit:]
            
            if expert_id:
                history = [
                    m for m in history 
                    if m.from_expert == expert_id or m.to_expert == expert_id
                ]
            
            return history
    
    def clear_history(self) -> None:
        """Clear message history."""
        with self._lock:
            self._history.clear()
    
    def get_pending_count(self, expert_id: str) -> int:
        """Get number of pending messages for an expert."""
        if expert_id in self._queues:
            return self._queues[expert_id].qsize()
        return 0
    
    def get_stats(self) -> dict:
        """Get statistics about message types and savings."""
        with self._lock:
            return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0
    
    def __repr__(self) -> str:
        registered = len(self._queues)
        history_size = len(self._history)
        tokens_saved = self._stats["estimated_tokens_saved"]
        return f"CrosstalkBus({registered} experts, {history_size} msgs, ~{tokens_saved} tokens saved)"


# Singleton instance for global access
_bus_instance: Optional[CrosstalkBus] = None

def get_crosstalk_bus() -> CrosstalkBus:
    """Get or create the global crosstalk bus instance."""
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = CrosstalkBus()
    return _bus_instance


if __name__ == "__main__":
    # Test A2A-style messaging
    bus = CrosstalkBus()
    
    # Register experts
    bus.register_expert("python_expert")
    bus.register_expert("security_expert")
    
    # Send a structured message (minimal tokenization)
    msg_id = bus.send_structured(
        from_expert="security_expert",
        to_expert="python_expert",
        summary="Found 3 security issues",  # Small text (tokenized)
        data={  # Detailed data (NOT tokenized!)
            "vulnerabilities": [
                {"type": "sql_injection", "severity": "high", "line": 42},
                {"type": "xss", "severity": "medium", "line": 87},
                {"type": "csrf", "severity": "low", "line": 123}
            ],
            "files_scanned": ["app.py", "routes.py", "auth.py"],
            "scan_time_ms": 150
        }
    )
    print(f"Sent structured message: {msg_id}")
    
    # Receive on python side
    messages = bus.receive("python_expert")
    for msg in messages:
        print(f"\nReceived from {msg.from_expert}:")
        print(f"  Text (tokenized): {msg.get_text_parts()}")
        print(f"  Data (NOT tokenized): {msg.get_data_parts()}")
        print(f"  Tokenization fraction: {msg.tokenization_fraction():.1%}")
    
    # Show stats
    print(f"\n{bus}")
    stats = bus.get_stats()
    print(f"Stats: {stats}")

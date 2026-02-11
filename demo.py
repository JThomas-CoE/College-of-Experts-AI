#!/usr/bin/env python3
"""
College of Experts - Interactive Demo

This demo showcases the core functionality:
1. Router classifies queries and selects experts
2. Experts are dynamically loaded via pluggable backends
3. Shared Memvid memory enables continuity
4. Multi-expert coordination for complex queries

Prerequisites:
- Python 3.10+
- For Transformers backend: torch, transformers
- For Ollama backend: ollama installed and running
- Optional: memvid for video-encoded memory
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt

try:
    from src.harness import Harness, HarnessConfig
    from src.router import RouterConfig
    from src.expert_loader import ExpertLoaderConfig
    from src.backends import BackendType
    
    # Try Memvid first, fall back to SQLite
    try:
        from src.memvid_memory import MemvidConfig
        MEMVID_AVAILABLE = True
    except ImportError:
        from src.memory_backbone import MemoryConfig
        MEMVID_AVAILABLE = False
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory.")
    print("Install dependencies: pip install -r requirements.txt")
    sys.exit(1)


console = Console()


def print_welcome(backend_type: BackendType, using_memvid: bool):
    """Print welcome banner."""
    memory_type = "Memvid" if using_memvid else "SQLite"
    console.print(Panel.fit(
        "[bold blue]College of Experts[/bold blue]\n"
        f"[dim]A disk-resident sparse mixture architecture[/dim]\n"
        f"[dim]Backend: {backend_type.value} | Memory: {memory_type}[/dim]",
        border_style="blue"
    ))
    console.print()
    console.print("[dim]Commands:[/dim]")
    console.print("  [cyan]/status[/cyan]  - Show system status")
    console.print("  [cyan]/experts[/cyan] - List available experts")
    console.print("  [cyan]/loaded[/cyan]  - Show loaded experts")
    console.print("  [cyan]/memory[/cyan]  - Show memory stats")
    console.print("  [cyan]/backend[/cyan] - Show backend info")
    console.print("  [cyan]/clear[/cyan]   - Clear conversation")
    console.print("  [cyan]/quit[/cyan]    - Exit")
    console.print()


def print_status(harness: Harness):
    """Print system status."""
    status = harness.get_status()
    
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Memory Backend", status.get("memory_backend", "unknown"))
    table.add_row("Model Backend", status["cache_stats"].get("backend_type", "unknown"))
    table.add_row("Active Sessions", str(status["active_sessions"]))
    table.add_row("Hot Experts", str(status["cache_stats"]["hot_count"]))
    table.add_row("Warm Experts", str(status["cache_stats"]["warm_count"]))
    
    # Handle different memory stat formats
    mem_stats = status["memory_stats"]
    if "working_count" in mem_stats:
        # Memvid format
        table.add_row("Working Memory", str(mem_stats["working_count"]))
        table.add_row("Pending Episodic", str(mem_stats.get("pending_episodic", 0)))
    else:
        # SQLite format
        table.add_row("Working Memory", str(mem_stats.get("working", 0)))
        table.add_row("Episodic Memory", str(mem_stats.get("episodic", 0)))
    
    console.print(table)


def print_backend_info(harness: Harness):
    """Print backend information."""
    cache_stats = harness.loader.get_cache_stats()
    backend = harness.loader.backend
    
    table = Table(title="Backend Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Backend Type", cache_stats.get("backend_type", "unknown"))
    table.add_row("Device", getattr(backend, 'default_device', 'N/A'))
    table.add_row("Hot Capacity", str(cache_stats["hot_capacity"]))
    table.add_row("Warm Capacity", str(cache_stats["warm_capacity"]))
    
    # Show loaded models
    loaded = backend.list_loaded()
    table.add_row("Models Loaded", str(len(loaded)))
    
    console.print(table)


def print_experts(harness: Harness):
    """Print available experts."""
    experts = harness.loader.list_available()
    
    table = Table(title="Available Experts")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Domains", style="dim")
    
    for expert_id in experts:
        info = harness.loader.get_expert_info(expert_id)
        if info:
            # Show appropriate model path based on backend
            if harness.loader.config.backend_type == BackendType.OLLAMA:
                model = info.get("ollama_name", info.get("model", ""))
            else:
                model = info.get("model", "")
            # Truncate long model names
            if len(model) > 35:
                model = model[:32] + "..."
            table.add_row(
                expert_id,
                info.get("display_name", ""),
                model,
                ", ".join(info.get("domains", []))
            )
    
    console.print(table)


def print_loaded(harness: Harness):
    """Print currently loaded experts."""
    stats = harness.loader.get_cache_stats()
    
    console.print("\n[bold]Hot (Active) Experts:[/bold]")
    for exp in stats["hot_experts"]:
        console.print(f"  • {exp}")
    if not stats["hot_experts"]:
        console.print("  [dim]None[/dim]")
    
    console.print("\n[bold]Warm (Cached) Experts:[/bold]")
    for exp in stats["warm_experts"]:
        console.print(f"  • {exp}")
    if not stats["warm_experts"]:
        console.print("  [dim]None[/dim]")
    console.print()


def print_memory(harness: Harness):
    """Print memory statistics."""
    if harness._using_memvid:
        stats = harness.memory.get_stats()
        table = Table(title="Memory Backbone (Memvid)")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Working Entries", str(stats["working_count"]))
        table.add_row("Pending Episodic", str(stats["pending_episodic"]))
        table.add_row("Pending Semantic", str(stats["pending_semantic"]))
        table.add_row("Expert Memories", ", ".join(stats["expert_memories_loaded"]) or "None")
    else:
        stats = harness.memory.get_tier_stats()
        table = Table(title="Memory Backbone (SQLite)")
        table.add_column("Tier", style="cyan")
        table.add_column("Entries", style="green")
        table.add_column("Description", style="dim")
        
        table.add_row("Working", str(stats["working"]), "Current task context")
        table.add_row("Episodic", str(stats["episodic"]), "Past sessions")
        table.add_row("Semantic", str(stats["semantic"]), "Learned facts")
    
    console.print(table)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="College of Experts - Interactive Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backend Types:
  transformers  Pure Python with HuggingFace (DirectML/CUDA/CPU)
  ollama        Ollama API with quantized models (fastest)
  hybrid        Ollama for hot model + Transformers for warm

Examples:
  python demo.py                          # Default: Transformers backend
  python demo.py --backend transformers   # Explicit Transformers
  python demo.py --backend ollama         # Use Ollama
  python demo.py --backend hybrid         # Hybrid mode
  python demo.py --no-memvid              # Disable Memvid memory
        """
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["transformers", "ollama", "hybrid"],
        default="transformers",
        help="Model backend to use (default: transformers)"
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        help="Device for transformers backend (auto, cuda, cpu, directml)"
    )
    parser.add_argument(
        "--no-memvid",
        action="store_true",
        help="Disable Memvid memory, use SQLite instead"
    )
    parser.add_argument(
        "--router-model",
        default="qwen3:4b",
        help="Model name for the router (default: qwen3:4b)"
    )
    return parser.parse_args()


def main():
    """Run the interactive demo."""
    args = parse_args()
    
    # Determine backend type
    backend_map = {
        "transformers": BackendType.TRANSFORMERS,
        "ollama": BackendType.OLLAMA,
        "hybrid": BackendType.HYBRID
    }
    backend_type = backend_map[args.backend]
    
    # Determine memory type
    use_memvid = MEMVID_AVAILABLE and not args.no_memvid
    
    print_welcome(backend_type, use_memvid)
    
    # Check backend availability
    if backend_type in (BackendType.OLLAMA, BackendType.HYBRID):
        try:
            import ollama
            ollama.list()
            console.print("[green]✓[/green] Ollama connected")
        except Exception as e:
            console.print(f"[yellow]![/yellow] Ollama not available: {e}")
            if backend_type == BackendType.OLLAMA:
                console.print("[dim]Falling back to Transformers backend[/dim]")
                backend_type = BackendType.TRANSFORMERS
    
    if backend_type == BackendType.TRANSFORMERS:
        try:
            import torch
            console.print(f"[green]✓[/green] PyTorch available (device: {args.device})")
        except ImportError:
            console.print("[red]✗[/red] PyTorch not available")
            console.print("[dim]Install with: pip install torch[/dim]")
            return
    
    # Memory status
    if use_memvid:
        console.print("[green]✓[/green] Memvid memory enabled")
    else:
        console.print("[yellow]![/yellow] Using SQLite memory")
    
    console.print()
    console.print("[dim]Initializing College of Experts...[/dim]")
    
    try:
        # Configure memory
        if use_memvid:
            memory_config = MemvidConfig(memory_dir=Path("data/memory"))
        else:
            memory_config = MemoryConfig(db_path=Path("data/memory.db"))
        
        # Configure expert loader with backend
        loader_config = ExpertLoaderConfig(
            backend_type=backend_type,
            device=args.device,
            max_hot_experts=3,
            max_warm_experts=8
        )
        
        # Configure harness
        config = HarnessConfig(
            router_config=RouterConfig(model_name=args.router_model),
            loader_config=loader_config,
            memory_config=memory_config,
            use_memvid=use_memvid
        )
        
        harness = Harness(config)
        session = harness.create_session()
        console.print(f"[green]✓[/green] Session created: {session.session_id}")
        console.print(f"[dim]Backend: {backend_type.value}[/dim]\n")
        
    except Exception as e:
        console.print(f"[red]Error initializing:[/red] {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Main loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            
            if not user_input.strip():
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()
                
                if cmd == "/quit" or cmd == "/exit":
                    console.print("[dim]Goodbye![/dim]")
                    harness.end_session(session.session_id)
                    break
                
                elif cmd == "/status":
                    print_status(harness)
                    continue
                
                elif cmd == "/experts":
                    print_experts(harness)
                    continue
                
                elif cmd == "/loaded":
                    print_loaded(harness)
                    continue
                
                elif cmd == "/memory":
                    print_memory(harness)
                    continue
                
                elif cmd == "/backend":
                    print_backend_info(harness)
                    continue
                
                elif cmd == "/clear":
                    harness.end_session(session.session_id)
                    session = harness.create_session()
                    console.print("[dim]Conversation cleared.[/dim]")
                    continue
                
                else:
                    console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
                    continue
            
            # Process through harness
            with console.status("[dim]Thinking...[/dim]"):
                response = harness.process(session.session_id, user_input)
            
            # Display response
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold green]Assistant[/bold green]",
                border_style="green"
            ))
            
            # Show which experts were used
            if session.active_experts:
                experts_str = ", ".join(session.active_experts)
                console.print(f"[dim]Experts: {experts_str}[/dim]")
        
        except KeyboardInterrupt:
            console.print("\n[dim]Use /quit to exit[/dim]")
        
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


if __name__ == "__main__":
    main()

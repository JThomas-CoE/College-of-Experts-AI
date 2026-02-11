"""
College of Experts GUI Components

Extracted and modularized from demo_v12_full.py for reusability.
"""

import tkinter as tk
from tkinter import ttk
import queue
import threading
import tempfile
import webbrowser
from datetime import datetime
from typing import List, Optional, Callable, Tuple


class QuerySubwindow:
    """A subwindow for a single query's progress and output."""
    
    def __init__(self, parent_frame: tk.Frame, query_text: str, query_num: int):
        self.parent = parent_frame
        self.query_text = query_text
        self.query_num = query_num
        self.progress_lines: List[str] = []
        self.output_sections: List[tuple] = []  # (title, content, expert_id)
        self.is_frozen = False
        self.html_content: Optional[str] = None
        
        # Main container frame with border
        self.frame = tk.Frame(parent_frame, bg="#2d2d3d", relief="ridge", bd=1)
        self.frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Query header
        header_frame = tk.Frame(self.frame, bg="#3d3d5c")
        header_frame.pack(fill=tk.X)
        
        query_label = tk.Label(
            header_frame,
            text=f"Query #{query_num}",
            font=("Consolas", 10, "bold"),
            fg="#89b4fa",
            bg="#3d3d5c",
            anchor="w",
            padx=5, pady=2
        )
        query_label.pack(side=tk.LEFT)
        
        time_label = tk.Label(
            header_frame,
            text=datetime.now().strftime("%H:%M:%S"),
            font=("Consolas", 9),
            fg="#6c7086",
            bg="#3d3d5c",
            padx=5
        )
        time_label.pack(side=tk.RIGHT)
        
        # Query text display - height based on content
        query_lines = query_text.count('\n') + 1
        query_height = min(10, max(4, query_lines + 1))  # 4-10 lines
        query_display = tk.Text(
            self.frame,
            height=query_height,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#1e1e2e",
            fg="#cdd6f4",
            relief="flat",
            padx=5, pady=5
        )
        query_display.pack(fill=tk.X, padx=5, pady=(5, 0))
        query_display.insert("1.0", query_text)
        query_display.config(state=tk.DISABLED)
        
        # Progress box (5 lines, scrollable)
        progress_frame = tk.Frame(self.frame, bg="#252535")
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        progress_label = tk.Label(
            progress_frame,
            text="Progress:",
            font=("Consolas", 9, "italic"),
            fg="#6c7086",
            bg="#252535",
            anchor="w"
        )
        progress_label.pack(anchor="w")
        
        self.progress_text = tk.Text(
            progress_frame,
            height=5,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#252535",
            fg="#a6e3a1",
            relief="flat",
            padx=5, pady=2
        )
        self.progress_text.pack(fill=tk.X)
        self.progress_text.config(state=tk.DISABLED)
        
        # Output sections container (initially empty)
        self.output_frame = tk.Frame(self.frame, bg="#2d2d3d")
        self.output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Button frame (initially hidden, shown after completion)
        self.button_frame = tk.Frame(self.frame, bg="#2d2d3d")
        
    def add_progress(self, text: str):
        """Add a progress line to the 5-line progress box."""
        if self.is_frozen:
            return
            
        self.progress_lines.append(text)
        display_lines = self.progress_lines[-5:]
        
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.delete("1.0", tk.END)
        self.progress_text.insert("1.0", "\n".join(display_lines))
        self.progress_text.see(tk.END)
        self.progress_text.config(state=tk.DISABLED)
        
    def add_section(self, title: str, content: str, expert_id: str = ""):
        """Add an output section."""
        self.output_sections.append((title, content, expert_id))
        
        # Check for HTML content
        if "<html" in content.lower() or "<!doctype html" in content.lower():
            self.html_content = content
        
        # Create section frame
        section_frame = tk.Frame(self.output_frame, bg="#1e1e2e", relief="groove", bd=1)
        section_frame.pack(fill=tk.X, pady=2)
        
        # Section header with copy button
        sec_header = tk.Frame(section_frame, bg="#3d3d5c")
        sec_header.pack(fill=tk.X)
        
        title_text = f"{title}"
        if expert_id:
            title_text += f" (by {expert_id})"
        
        title_label = tk.Label(
            sec_header,
            text=title_text,
            font=("Consolas", 10, "bold"),
            fg="#f9e2af",
            bg="#3d3d5c",
            anchor="w",
            padx=5, pady=2
        )
        title_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        copy_btn = tk.Button(
            sec_header,
            text="Copy",
            font=("Consolas", 8),
            bg="#45475a",
            fg="#cdd6f4",
            relief="flat",
            padx=5,
            command=lambda c=content: self._copy_to_clipboard(c)
        )
        copy_btn.pack(side=tk.RIGHT, padx=2, pady=2)
        
        content_text = tk.Text(
            section_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#1e1e2e",
            fg="#cdd6f4",
            relief="flat",
            padx=5, pady=5,
            height=min(20, max(5, content.count('\n') + 1))
        )
        content_text.pack(fill=tk.X)
        content_text.insert("1.0", content)
        content_text.config(state=tk.DISABLED)
        
    def freeze_progress(self):
        """Freeze the progress box and show action buttons."""
        self.is_frozen = True
        self.progress_text.config(bg="#1e1e2e")
        
        self.button_frame.pack(fill=tk.X, pady=5)
        
        copy_all_btn = tk.Button(
            self.button_frame,
            text="Copy All Sections",
            font=("Consolas", 9),
            bg="#89b4fa",
            fg="#1e1e2e",
            relief="flat",
            padx=10, pady=3,
            command=self._copy_all_sections
        )
        copy_all_btn.pack(side=tk.LEFT, padx=5)
        
        if self.html_content:
            preview_btn = tk.Button(
                self.button_frame,
                text="Preview HTML in Browser",
                font=("Consolas", 9),
                bg="#a6e3a1",
                fg="#1e1e2e",
                relief="flat",
                padx=10, pady=3,
                command=self._preview_html
            )
            preview_btn.pack(side=tk.LEFT, padx=5)
    
    def _copy_to_clipboard(self, text: str):
        self.parent.clipboard_clear()
        self.parent.clipboard_append(text)
        
    def _copy_all_sections(self):
        all_text = []
        for title, content, expert in self.output_sections:
            all_text.append(f"=== {title} ===")
            if expert:
                all_text.append(f"(Generated by: {expert})")
            all_text.append(content)
            all_text.append("")
        self._copy_to_clipboard("\n".join(all_text))
        
    def _preview_html(self):
        if not self.html_content:
            return
        content = self.html_content
        if "```html" in content:
            start = content.find("```html") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        webbrowser.open(f'file://{temp_path}')


class CoEDemoGUI:
    """Main GUI window for College of Experts Demo."""
    
    def __init__(self, on_query_callback: Optional[Callable] = None, demo_mode: bool = False):
        self.on_query = on_query_callback
        self.demo_mode = demo_mode
        self.is_running = True
        self.is_generating = False
        self.message_queue = queue.Queue()
        self.current_subwindow: Optional[QuerySubwindow] = None
        self.query_count = 0
        
        self.root = tk.Tk()
        self.root.title("College of Experts V13" + (" - Demo Mode" if demo_mode else ""))
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e2e")
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2e")
        
        self._create_widgets()
        self._bind_events()
        self._show_welcome()
        
    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(main_frame, bg="#1e1e2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#1e1e2e")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        input_frame = tk.Frame(self.root, bg="#313244", pady=10, padx=10)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(
            input_frame,
            textvariable=self.input_var,
            font=("Consolas", 11),
            bg="#45475a",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            relief="flat"
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=5)
        
        self.send_btn = tk.Button(
            input_frame,
            text="Send",
            font=("Consolas", 10, "bold"),
            bg="#89b4fa",
            fg="#1e1e2e",
            relief="flat",
            padx=15, pady=5,
            command=self._on_send
        )
        self.send_btn.pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Consolas", 9),
            bg="#313244",
            fg="#6c7086",
            anchor="w",
            padx=10, pady=3
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def _bind_events(self):
        self.input_entry.bind("<Return>", lambda e: self._on_send())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
    def _show_welcome(self):
        welcome_frame = tk.Frame(self.scrollable_frame, bg="#2d2d3d", relief="ridge", bd=1)
        welcome_frame.pack(fill=tk.X, padx=5, pady=5)
        
        msg = ("Welcome to College of Experts V13!\n\n"
               "Enter a query below to generate code with multiple expert models.\n"
               "Example: 'Build a task manager with HTML frontend, SQLite database, and Flask backend'\n\n"
               "Features:\n"
               "  - Dual-embedding semantic routing to specialist models\n"
               "  - Quality gate with completeness/accuracy scoring\n"
               "  - Automatic retry with prompt refinement\n"
               "  - DeepSeek R1 fallback for complex tasks\n\n"
               "Commands: /exit to quit")
        
        if self.demo_mode:
            msg += "\n\n[Demo Mode] Will auto-run preset query..."
            
        welcome_text = tk.Label(
            welcome_frame,
            text=msg,
            font=("Consolas", 10),
            bg="#2d2d3d",
            fg="#cdd6f4",
            justify=tk.LEFT,
            padx=15, pady=15
        )
        welcome_text.pack()
            
    def _on_send(self):
        if self.is_generating:
            return
        text = self.input_var.get().strip()
        if not text:
            return
        if text.lower() in ['/exit', '/quit', 'exit', 'quit']:
            self._on_close()
            return
        self.input_var.set("")
        self._submit_query(text)
        
    def _submit_query(self, query_text: str):
        self.is_generating = True
        self.send_btn.config(state=tk.DISABLED, bg="#45475a")
        self.status_var.set("Generating...")
        
        self.query_count += 1
        self.current_subwindow = QuerySubwindow(
            self.scrollable_frame, 
            query_text, 
            self.query_count
        )
        
        self.root.update_idletasks()
        self.canvas.yview_moveto(1.0)
        
        if self.on_query:
            thread = threading.Thread(
                target=self._process_query_thread,
                args=(query_text,),
                daemon=True
            )
            thread.start()
            
    def _process_query_thread(self, query_text: str):
        try:
            if self.on_query:
                self.on_query(query_text, self)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.message_queue.put(("error", str(e)))
        finally:
            self.message_queue.put(("done", None))
            
    def show_progress(self, text: str):
        """Thread-safe: show progress in current subwindow."""
        self.message_queue.put(("progress", text))
        
    def show_section(self, title: str, content: str, expert_id: str = ""):
        """Thread-safe: show a generated section."""
        self.message_queue.put(("section", (title, content, expert_id)))
        
    def set_status(self, text: str):
        """Thread-safe: update status bar."""
        self.message_queue.put(("status", text))
        
    def generation_complete(self):
        """Thread-safe: signal generation finished."""
        self.message_queue.put(("complete", None))
        
    def _check_queue(self):
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                
                if msg_type == "progress" and self.current_subwindow:
                    self.current_subwindow.add_progress(data)
                    self.root.update_idletasks()
                elif msg_type == "section" and self.current_subwindow:
                    title, content, expert = data
                    self.current_subwindow.add_section(title, content, expert)
                    self.root.update_idletasks()
                    self.canvas.yview_moveto(1.0)
                elif msg_type == "status":
                    self.status_var.set(data)
                    self.root.update_idletasks()
                elif msg_type == "complete":
                    if self.current_subwindow:
                        self.current_subwindow.freeze_progress()
                    self.is_generating = False
                    self.send_btn.config(state=tk.NORMAL, bg="#89b4fa")
                    self.status_var.set("Ready - Enter another query or /exit")
                    self.root.update_idletasks()
                elif msg_type == "done":
                    self.is_generating = False
                    self.send_btn.config(state=tk.NORMAL, bg="#89b4fa")
                    self.root.update_idletasks()
                elif msg_type == "error":
                    if self.current_subwindow:
                        self.current_subwindow.add_progress(f"[ERROR] {data}")
                        self.current_subwindow.freeze_progress()
                    self.status_var.set(f"Error: {data}")
                    self.root.update_idletasks()
        except queue.Empty:
            pass
            
        if self.is_running:
            self.root.after(16, self._check_queue)
            
    def _on_close(self):
        self.is_running = False
        self.root.destroy()
        
    def run(self):
        self._check_queue()
        self.input_entry.focus()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n[GUI interrupted]")
        except Exception as e:
            print(f"\n[GUI exception]: {e}")
            import traceback
            traceback.print_exc()
        
    def submit_demo_query(self, query_text: str):
        """Auto-submit a query after a delay (for demo mode)."""
        self.root.after(1000, lambda: self._submit_query(query_text))
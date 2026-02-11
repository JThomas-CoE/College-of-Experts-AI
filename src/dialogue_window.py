"""
Dialogue Window - Clean GUI for College of Experts V7
Separates conversation from loading/system messages.
"""
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import threading
import queue
from datetime import datetime


class DialogueWindow:
    """Simple GUI window for conversation - separate from terminal."""
    
    def __init__(self, on_submit_callback=None):
        self.on_submit = on_submit_callback
        self.response_queue = queue.Queue()
        self.is_running = True
        
        # Create window
        self.root = tk.Tk()
        self.root.title("College of Experts V7 - Dialogue")
        self.root.geometry("800x600")
        self.root.configure(bg="#1e1e2e")
        
        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2e")
        style.configure("TButton", padding=6, relief="flat", background="#89b4fa")
        style.configure("TEntry", padding=6)
        
        self._create_widgets()
        self._bind_events()
        
        # Show welcome message
        self.add_system_message("Welcome to College of Experts V7!")
        self.add_system_message("Your AI assistant council is loading...")
        self.add_system_message("You can start typing - responses will appear once ready.\n")
    
    def _create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Conversation display
        self.conversation = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg="#1e1e2e",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            selectbackground="#45475a",
            relief="flat",
            padx=10,
            pady=10
        )
        self.conversation.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.conversation.config(state=tk.DISABLED)
        
        # Configure text tags for styling
        self.conversation.tag_configure("user", foreground="#89b4fa", font=("Consolas", 11, "bold"))
        self.conversation.tag_configure("assistant", foreground="#a6e3a1")
        self.conversation.tag_configure("system", foreground="#6c7086", font=("Consolas", 10, "italic"))
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X)
        
        # Input entry
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(
            input_frame,
            textvariable=self.input_var,
            font=("Consolas", 11)
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Send button
        self.send_btn = ttk.Button(
            input_frame,
            text="Send",
            command=self._on_send
        )
        self.send_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Copy button
        self.copy_btn = ttk.Button(
            input_frame,
            text="Copy Last",
            command=self._copy_last_response
        )
        self.copy_btn.pack(side=tk.LEFT)
        
        # Status bar
        self.status_var = tk.StringVar(value="Loading models...")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            background="#313244",
            foreground="#6c7086",
            padding=5
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self._last_response = ""
    
    def _bind_events(self):
        self.input_entry.bind("<Return>", lambda e: self._on_send())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _on_send(self):
        text = self.input_var.get().strip()
        if not text:
            return
        
        self.input_var.set("")
        self.add_user_message(text)
        
        if self.on_submit:
            # Call in thread to not block UI
            threading.Thread(target=self._process_input, args=(text,), daemon=True).start()
    
    def _process_input(self, text):
        try:
            if self.on_submit:
                response = self.on_submit(text)
                self.response_queue.put(("assistant", response))
        except Exception as e:
            import traceback
            traceback.print_exc()  # Print full traceback to terminal
            self.response_queue.put(("system", f"[Error] {e}"))
    
    def _copy_last_response(self):
        if self._last_response:
            self.root.clipboard_clear()
            self.root.clipboard_append(self._last_response)
            self.set_status("Copied to clipboard!")
    
    def _on_close(self):
        self.is_running = False
        self.root.destroy()
    
    def add_user_message(self, text):
        self._append_text(f"\nYou: {text}\n", "user")
    
    def add_assistant_message(self, text):
        self._append_text(f"\nAssistant: {text}\n", "assistant")
        self._last_response = text
    
    def add_system_message(self, text):
        self._append_text(f"{text}\n", "system")
    
    def _append_text(self, text, tag=None):
        self.conversation.config(state=tk.NORMAL)
        if tag:
            self.conversation.insert(tk.END, text, tag)
        else:
            self.conversation.insert(tk.END, text)
        self.conversation.see(tk.END)
        self.conversation.config(state=tk.DISABLED)
    
    def set_status(self, text):
        self.status_var.set(text)
    
    def check_queue(self):
        """Check for responses from background threads."""
        try:
            while True:
                msg_type, msg = self.response_queue.get_nowait()
                if msg_type == "assistant":
                    self.add_assistant_message(msg)
                elif msg_type == "system":
                    self.add_system_message(msg)
        except queue.Empty:
            pass
        
        if self.is_running:
            self.root.after(100, self.check_queue)
    
    def run(self):
        """Start the GUI event loop."""
        self.check_queue()
        self.input_entry.focus()
        self.root.mainloop()


# Test
if __name__ == "__main__":
    def mock_callback(text):
        import time
        time.sleep(1)
        return f"You said: {text}"
    
    window = DialogueWindow(on_submit_callback=mock_callback)
    window.run()

import os
import argparse
import subprocess
import psutil
import time
import threading
import shutil

def memory_monitor(stop_event, max_ram_gb=61.5):
    print(f"[Monitor] Starting memory monitor. Base limit: {max_ram_gb} GB")
    peak_ram = 0
    while not stop_event.is_set():
        virtual_mem = psutil.virtual_memory()
        used_gb = virtual_mem.used / (1024**3)
        peak_ram = max(peak_ram, used_gb)
        
        if used_gb > max_ram_gb - 2:
            print(f"[Monitor] ALERT: System RAM at {used_gb:.2f} GB", flush=True)
        
        time.sleep(3)
    print(f"[Monitor] Peak System RAM: {peak_ram:.2f} GB")

def convert_to_oga_4bit(model_path, output_path, ep="cpu"):
    print(f"--- Disk-Offloaded Build ({ep}) ---")
    print(f"Target: {model_path}")
    
    # Create offload folder
    offload_dir = os.path.join(os.getcwd(), "offload_temp")
    if os.path.exists(offload_dir):
        shutil.rmtree(offload_dir)
    os.makedirs(offload_dir)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    builder_script = r"C:\RyzenAI\envs\zimage\Lib\site-packages\onnxruntime_genai\models\builder.py"
    
    # Passing options to Transformers via Builder
    # We use 'device_map=auto' and 'offload_folder' to essentially use the SSD as RAM extension
    model_path = os.path.abspath(model_path)
    output_path = os.path.abspath(output_path)
    
    cmd = [
        "python", builder_script,
        "-i", model_path,
        "-o", output_path,
        "-p", "int4",
        "-e", ep, 
        "--extra_options", 
        "low_cpu_mem_usage=True", 
        "torch_dtype=float16",
        "device_map=cpu",
        f"offload_folder={offload_dir}"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=memory_monitor, args=(stop_event, 62.0))
    monitor_thread.start()
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            # Tripwire
            if (psutil.virtual_memory().used / (1024**3)) > 63.5:
                print("\n[CRITICAL] SYSTEM RAM LIMIT REACHED. ABORTING.")
                process.kill()
                break
        process.wait()
    finally:
        stop_event.set()
        monitor_thread.join()
        if os.path.exists(offload_dir):
            shutil.rmtree(offload_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("output_path")
    parser.add_argument("--ep", default="cpu", help="Execution provider for build (cpu or dml)")
    args = parser.parse_args()
    convert_to_oga_4bit(args.model_path, args.output_path, args.ep)

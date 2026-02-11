from llama_cpp import llama_cpp
import os

def check_backend():
    print(f"llama-cpp-python version: {os.path.dirname(llama_cpp.__file__)}")
    
    # Check for library features
    # llama_cpp.llama_supports_gpu_offload() might exist in some versions
    # We can check the system info string
    try:
        sys_info = llama_cpp.llama_print_system_info().decode('utf-8')
        print(f"System Info: {sys_info}")
    except Exception as e:
        print(f"Could not get system info: {e}")

if __name__ == "__main__":
    check_backend()

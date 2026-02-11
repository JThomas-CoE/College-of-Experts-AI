"""Quick test to capture demo output to file for comparison."""
import sys
import io

# Capture all output
output = io.StringIO()
sys.stdout = output
sys.stderr = output

try:
    from demo_multi_expert_v6 import CollegeOfExperts, DemoConfig
    
    config = DemoConfig()
    coe = CollegeOfExperts(config)
    
    query = "How do I implement JWT authentication securely in FastAPI?"
    result = coe.compare(query)
    
    # Reset stdout for final output
    sys.stdout = sys.__stdout__
    
    # Write clean output to file
    with open("output_comparison.txt", "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("CoE SYNTHESIZED ANSWER\n")
        f.write("="*60 + "\n\n")
        f.write(result["coe"]["synthesis"])
        f.write("\n\n")
        f.write("="*60 + "\n")
        f.write("DIRECT ROUTER ANSWER\n")
        f.write("="*60 + "\n\n")
        f.write(result["direct"]["response"])
        f.write("\n\n")
        f.write("="*60 + "\n")
        f.write("TIMING COMPARISON\n")
        f.write("="*60 + "\n")
        f.write(f"CoE total time: {result['coe'].get('parallel_time', 'N/A')}s\n")
        f.write(f"Direct answer time: {result['direct']['time']:.1f}s\n")
    
    print("Output written to output_comparison.txt")
    coe.cleanup()
    
except Exception as e:
    sys.stdout = sys.__stdout__
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

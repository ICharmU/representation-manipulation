import subprocess
import sys


def generate_causation_correlation_pairs():
    script_fp = "generate_causation_correlation.py"

    try:
        print("Generating causation-correlation pairs...")
        subprocess.run(["python", script_fp])
    except Exception as e:
        print("Failed to generate causation-correlation pairs")
        raise e
    
    print("Finished generating causation-correlation pairs")

def generate_mechanism_temporal_pairs():
    script_fp = "generate_mechanism_temporal.py"

    try:
        print("Generating mechanism-temporal pairs...")
        subprocess.run(["python", script_fp])
    except Exception as e:
        print("Failed to generate mechanism-temporal pairs")
        raise e
    
    print("Finished generating mechanism-temporal pairs")

def generate_synthetic_scms():
    script_fp = "generate_synthetic_scm.py"

    try:
        print("Generating synthetic structural causal models...")
        subprocess.run(["python", script_fp])
    except Exception as e:
        print("Failed to generate synthetic structural causal models")
        raise e
    
    print("Finished generating synthetic structural causal models")

# expects rm_p1 environment to be active and script to be run from /part_1 directory:
# conda activate rm_p1
# python runner.py
if __name__ == "__main__":

    generate_causation_correlation_pairs()
    generate_mechanism_temporal_pairs()
    generate_synthetic_scms()
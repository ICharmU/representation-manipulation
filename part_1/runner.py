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

# expects rm_p1 environment to be active and script to be run from /part_1 directory:
# conda activate rm_p1
# python runner.py
if __name__ == "__main__":

    generate_causation_correlation_pairs()
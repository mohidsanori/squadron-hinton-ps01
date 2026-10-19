"""
Mission 02 entry point under the file name used in the instructor repo README.
The simulator lives in 04_fst_noisy_simulator.py (the name on the course site);
this file runs it with the same arguments.
"""

import os
import runpy

if __name__ == "__main__":
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_fst_noisy_simulator.py"), run_name="__main__")

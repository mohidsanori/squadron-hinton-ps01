"""
Mission 03 entry point under the file name used in the instructor repo README.
The code lives in 01_MultiHead_Attention_Core.py (the name in the briefing and on the course site);
this file runs it with the same arguments.
"""

import os
import runpy

if __name__ == "__main__":
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_MultiHead_Attention_Core.py"), run_name="__main__")

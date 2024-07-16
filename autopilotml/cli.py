import os
import subprocess
from .streamlit_ui import main

def main():
    # Get the directory of the current module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "streamlit_ui.py")
    subprocess.call(["streamlit", "run", script_path])

if __name__ == "__main__":
    main()
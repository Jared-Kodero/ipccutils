import stat
import subprocess
import warnings
from pathlib import Path
import os
import time
warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
HOME = Path.home()

def which(cmd):
    try:
        path = (
            subprocess.check_output(["which", cmd], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

        return True if path else False
    except subprocess.CalledProcessError:
        return None


def eval_pkg_latex():
    """
    Check if LaTeX is installed; if not, begin asynchronous installation using a shell script.
    Returns:
        bool: True if LaTeX is installed, False otherwise (installation started or in progress).
    """

    latex_path = which("latex")
    if latex_path:
        return True

    # Define paths
    latex_sh = SCRIPT_DIR / "src" / "latex.install"
    state_file = SCRIPT_DIR / "src" / "installing"
    pid_file = SCRIPT_DIR / "src" / "PID"

    # Check if installation is already in progress
    if state_file.exists():
        print("LaTeX installation in progress...")
        return False

    # Make the install script executable


    # copy the install script to ~/
    os.system(f"cp -rf {latex_sh} {HOME}/latex.install")
    os.system(f"chmod +x {HOME}/latex.install")



    # Construct and run the command

    cmd = f"nohup bash -c {latex_sh} > {state_file} 2>&1 & echo $! > {pid_file}"
    os.system(cmd)
    
    time.sleep(5)  # Wait a moment to ensure the PID file is created

    with open(pid_file, "r") as f:
        pid_code = f.read().strip()
        print(f"Installing LaTeX... PID: {pid_code}"
              f" | Estimated time remaining: ~57 minutes. You can continue using the package.")

    return False

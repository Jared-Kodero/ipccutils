import stat
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent


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

    mode = latex_sh.stat().st_mode
    latex_sh.chmod(mode | stat.S_IXUSR)

    # Construct and run the command
    try:
        with open(pid_file, "w") as f_pid, open(state_file, "w") as f_state:
            cmd = f"nohup bash -c '{latex_sh}' > '{state_file}' 2>&1 & echo $! > '{pid_file}'"
            subprocess.Popen(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except Exception as e:
        print(f"Error launching LaTeX installation: {e}")
        return False

    try:
        with open(pid_file, "r") as f:
            pid_code = f.read().strip()
        print(
            f"Installing LaTeX... PID: {pid_code} | Estimated time remaining: ~57 minutes. You can continue using the package."
        )
    except FileNotFoundError:
        print("PID file not created. Installation may not have started correctly.")
        return False

    return False

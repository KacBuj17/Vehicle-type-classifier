import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.abspath(os.path.join(ROOT, "source"))
SCRIPTS = os.path.join(ROOT, "scripts")

env = os.environ.copy()
env["PYTHONPATH"] = SOURCE

def run_script(name):
    script_path = os.path.join(SCRIPTS, name)
    print(f"\nRunning {script_path}")
    subprocess.run(
        ["python", script_path],
        check=True,
        env=env
    )

if __name__ == "__main__":
    run_script("train.py")
    run_script("prune.py")
    run_script("quantize.py")
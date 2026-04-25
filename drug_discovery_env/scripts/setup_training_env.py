from __future__ import annotations

import argparse
import platform
import subprocess
import sys


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install training dependencies for current platform")
    parser.add_argument("--cuda", action="store_true", help="Install CUDA torch wheel set")
    args = parser.parse_args()

    py = sys.executable

    run([py, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])

    if args.cuda:
        run([
            py,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ])
    else:
        # macOS (MPS) and CPU default path.
        run([py, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])

    # OpenEnv is required by core models/environment classes.
    run([py, "-m", "pip", "install", "-U", "openenv"])

    run([py, "-m", "pip", "install", "-e", ".[training,test]"])

    system = platform.system().lower()
    print("Setup complete on", system)
    print("Try: python -m drug_discovery_env.scripts.run_grpo --episodes 1 --data-mode live_only --device auto")


if __name__ == "__main__":
    main()

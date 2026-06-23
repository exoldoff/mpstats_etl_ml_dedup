from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import os
import platform
import shutil
import subprocess
import sys


PACKAGES = [
    "torch",
    "transformers",
    "sentence-transformers",
    "datasets",
    "accelerate",
    "peft",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check runtime readiness for dedup fine-tuning.")
    parser.add_argument("--allow-cpu", action="store_true", help="Do not fail when CUDA is unavailable.")
    return parser


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _print_nvidia_smi() -> None:
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi: not found")
        return
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
    except Exception as exc:
        print(f"nvidia-smi: failed: {exc}")
        return
    print("nvidia-smi:")
    for line in output.strip().splitlines():
        print(f"  {line}")


def _torch_report() -> tuple[bool, list[str]]:
    import torch

    errors: list[str] = []
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    cuda_available = bool(torch.cuda.is_available())
    print(f"torch.cuda.is_available: {cuda_available}")
    if not cuda_available:
        errors.append("CUDA is not available to PyTorch")
        return False, errors

    device_count = torch.cuda.device_count()
    print(f"torch.cuda.device_count: {device_count}")
    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / 1024**3
        bf16_supported = torch.cuda.is_bf16_supported()
        print(
            "cuda_device:"
            f" index={index}"
            f" name={props.name!r}"
            f" capability={props.major}.{props.minor}"
            f" memory_gb={total_gb:.1f}"
            f" bf16_supported={bf16_supported}"
        )
    return True, errors


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"DEDUP_TRAINING_PRECISION: {os.environ.get('DEDUP_TRAINING_PRECISION', 'fp16')}")
    for package in PACKAGES:
        print(f"{package}: {_package_version(package)}")
    _print_nvidia_smi()

    try:
        cuda_ok, errors = _torch_report()
    except Exception as exc:
        cuda_ok = False
        errors = [f"torch check failed: {exc}"]

    if errors:
        print("runtime_doctor_errors:")
        for error in errors:
            print(f"  - {error}")
    if not cuda_ok and not args.allow_cpu:
        return 2
    print("runtime_doctor: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

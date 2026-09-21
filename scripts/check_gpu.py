"""Verify that the local environment can see the NVIDIA GPU."""

import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Install a CUDA-enabled PyTorch build before training."
        )

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()

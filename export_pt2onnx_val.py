#!/usr/bin/env python3
"""
Export a YOLO .pt checkpoint to ONNX compatible with val.py.

This wrapper uses the project's official export pipeline (export.run),
which configures detection heads for export and writes YOLO metadata.
"""

import argparse
import datetime
from pathlib import Path

from export import run as export_run


def parse_args():
    parser = argparse.ArgumentParser(description="Export .pt to ONNX for val.py compatibility")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[640, 640],
        help="Image size as: H W (or single value for square)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Export batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Export device, e.g. cpu or 0")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX with onnx-simplifier")
    parser.add_argument(
        "--out-root",
        type=str,
        default="onnx_exports",
        help="Root directory where timestamped export folders are created",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional explicit output .onnx path (overrides timestamped folder output)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    imgsz = args.imgsz if len(args.imgsz) > 1 else [args.imgsz[0], args.imgsz[0]]
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    def run_export(device_value):
        return export_run(
            weights=str(weights_path),
            imgsz=imgsz,
            batch_size=args.batch_size,
            device=device_value,
            include=("onnx",),
            dynamic=args.dynamic,
            simplify=args.simplify,
            opset=args.opset,
            half=False,
            inplace=False,
        )

    try:
        exported = run_export(args.device)
    except Exception as e:
        # Some checkpoints/modules can trigger mixed CPU/CUDA tensors during ONNX export.
        # Fallback to CPU export for reliability with val.py.
        if "Expected all tensors to be on the same device" in str(e) and args.device != "cpu":
            print("[WARN] CUDA export hit mixed-device tensors; retrying ONNX export on CPU...")
            exported = run_export("cpu")
        else:
            raise

    onnx_files = [Path(x) for x in exported if str(x).endswith(".onnx")]
    if not onnx_files:
        raise RuntimeError("ONNX export did not produce an .onnx file.")

    onnx_path = onnx_files[0].resolve()
    if args.output:
        target = Path(args.output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        onnx_path.replace(target)
        onnx_path = target
    else:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.out_root).resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / onnx_path.name
        onnx_path.replace(target)
        onnx_path = target

    print(f"[OK] ONNX saved: {onnx_path}")
    print(
        "[TIP] Validate with: "
        f"python val.py --data <data.yaml> --weights {onnx_path} --task val"
    )


if __name__ == "__main__":
    main()

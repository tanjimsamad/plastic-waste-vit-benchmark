#!/usr/bin/env python3
"""Plastic / Non_Plastic / Background inference with the fine-tuned DeiT-Small checkpoint.

Examples
--------
Classify images (files and/or folders):
    python scripts/inference.py --image photo.jpg samples/

Save predictions to CSV:
    python scripts/inference.py --image samples/ --csv predictions.csv

Live webcam (press Q to quit, S to save the current frame):
    python scripts/inference.py --webcam
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

MODEL_NAME = "deit_small_patch16_224"
DEFAULT_CLASSES = ["Background", "Non_Plastic", "Plastic"]  # ImageFolder (alphabetical) order
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "deit_small_model" / "best_model.pth",
                   help="Path to best_model.pth saved by the notebook (default: %(default)s)")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", nargs="+", type=Path, help="Image file(s) and/or folder(s)")
    mode.add_argument("--webcam", action="store_true", help="Run live webcam inference")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    p.add_argument("--save-dir", type=Path, default=REPO_ROOT / "webcam_test",
                   help="Where pressing S saves frames (default: %(default)s)")
    p.add_argument("--csv", type=Path, help="Write image predictions to this CSV file")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        sys.exit(f"Checkpoint not found: {checkpoint_path}\n"
                 "Train it with the notebook (section 5) or download it from the Releases page.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_names = list(checkpoint.get("class_names", DEFAULT_CLASSES))
    else:
        state_dict, class_names = checkpoint, DEFAULT_CLASSES

    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    return model, transform, class_names


@torch.inference_mode()
def predict(model, transform, image: Image.Image, device: torch.device) -> torch.Tensor:
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    return torch.softmax(model(tensor), dim=1)[0].cpu()


def collect_images(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files += sorted(f for f in path.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        elif path.suffix.lower() in IMAGE_EXTENSIONS and path.exists():
            files.append(path)
        else:
            print(f"Skipping (not an image or not found): {path}", file=sys.stderr)
    return files


def run_images(model, transform, class_names, device, paths, csv_path):
    files = collect_images(paths)
    if not files:
        sys.exit("No images found.")

    rows = []
    print(f"{'file':<50} {'prediction':<14} confidence")
    for file in files:
        try:
            with Image.open(file) as img:
                probs = predict(model, transform, img, device)
        except Exception as exc:  # corrupt / unreadable file
            print(f"{str(file):<50} ERROR: {exc}", file=sys.stderr)
            continue
        idx = int(probs.argmax())
        rows.append({"file": str(file), "prediction": class_names[idx],
                     "confidence": round(float(probs[idx]) * 100, 2),
                     **{f"p_{n}": round(float(p) * 100, 2) for n, p in zip(class_names, probs)}})
        print(f"{str(file)[-50:]:<50} {class_names[idx]:<14} {probs[idx] * 100:6.1f}%")

    if csv_path and rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} predictions to {csv_path}")


def run_webcam(model, transform, class_names, device, camera_index, save_dir):
    import cv2  # imported here so image mode works without a display stack

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        sys.exit(f"Could not open webcam {camera_index}. Try --camera 1.")

    save_dir.mkdir(parents=True, exist_ok=True)
    print("Press Q to quit | Press S to save the current frame")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a webcam frame.", file=sys.stderr)
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probs = predict(model, transform, image, device)
            idx = int(probs.argmax())

            cv2.putText(frame, f"Prediction: {class_names[idx]}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {probs[idx] * 100:.1f}%", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for i, (name, prob) in enumerate(zip(class_names, probs)):
                cv2.putText(frame, f"{name}: {prob * 100:.1f}%", (15, 110 + 27 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("DeiT-Small | Plastic Classification", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out = save_dir / f"webcam_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(out), frame)
                print("Saved:", out)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, transform, class_names = load_model(args.checkpoint, device)
    print(f"Loaded {MODEL_NAME} on {device} | classes: {class_names}")

    if args.webcam:
        run_webcam(model, transform, class_names, device, args.camera, args.save_dir)
    else:
        run_images(model, transform, class_names, device, args.image, args.csv)


if __name__ == "__main__":
    main()

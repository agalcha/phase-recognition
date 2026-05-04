#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import cv2
from unet_model import MobileUNet

def draw_label(img, pct):
    h, w = img.shape[:2]
    text = f"Target phase: {pct:.2f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, w / 1200)
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 12
    box_x2 = pad + tw + pad
    box_y2 = pad + th + baseline + pad
    cv2.rectangle(img, (pad, pad), (box_x2, box_y2), (20, 20, 20), -1)
    cv2.rectangle(img, (pad, pad), (box_x2, box_y2), (0, 212, 200), 1)
    text_x = pad + pad
    text_y = pad + th + (baseline // 2)
    cv2.putText(img, text, (text_x, text_y), font, scale, (0, 212, 200), thickness, cv2.LINE_AA)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to any image to run inference on")
    args = parser.parse_args()

    HERE       = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(HERE, "models", "model.pth")
    test_image = args.image if args.image else os.path.join(HERE, "images", "test.png")

    img_orig = cv2.imread(test_image)
    if img_orig is None:
        raise FileNotFoundError(f"Not found: {test_image}")
    h, w = img_orig.shape[:2]

    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    t   = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    model = MobileUNet(n_classes=2, freeze_encoder=True)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        mask_small = torch.softmax(model(t), dim=1)[0, 1].numpy() > 0.5

    mask = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    pct = 100.0 * mask.sum() / mask.size
    print(f"Target phase: {pct:.2f}%")
    print(f"Background:   {100.0 - pct:.2f}%")

    overlay = img_orig.copy()
    overlay[mask == 1] = (220, 220, 0)   # cyan in BGR
    blend = cv2.addWeighted(img_orig, 0.55, overlay, 0.45, 0)
    blend = draw_label(blend, pct)

    name = os.path.splitext(os.path.basename(test_image))[0]
    overlay_path = os.path.join(HERE, "models", name + "_overlay.png")
    mask_path    = os.path.join(HERE, "models", name + "_mask.png")
    cv2.imwrite(overlay_path, blend)
    cv2.imwrite(mask_path, mask * 255)
    print(f"Saved overlay → {overlay_path}")
    print(f"Saved mask    → {mask_path}")

if __name__ == "__main__":
    main()

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PhaseDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=512, train=True, augment=False):
        all_images = sorted(glob.glob(os.path.join(image_dir, "*.png")) +
                            glob.glob(os.path.join(image_dir, "*.jpg")) +
                            glob.glob(os.path.join(image_dir, "*.jpeg")))

        # only keep images that have a matching mask filename
        mask_names = {os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(mask_dir, "*.png"))}

        paired = [(p, os.path.join(mask_dir, os.path.splitext(os.path.basename(p))[0] + ".png"))
                  for p in all_images
                  if os.path.splitext(os.path.basename(p))[0] in mask_names]

        if train:
            paired = [(img, mask) for img, mask in paired
                      if "test" not in os.path.basename(img).lower()]

        if not paired:
            raise RuntimeError("No matched image/mask pairs found.")

        self.image_paths = [p[0] for p in paired]
        self.mask_paths  = [p[1] for p in paired]
        self.img_size    = img_size
        self.augment     = augment


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img  = cv2.imread(self.image_paths[idx])
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        img  = cv2.resize(img,  (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        if self.augment:
            if np.random.rand() < 0.5:
                img  = np.flip(img,  axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() < 0.5:
                img  = np.flip(img,  axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            k = np.random.randint(4)
            img  = np.rot90(img,  k, (0, 1)).copy()
            mask = np.rot90(mask, k, (0, 1)).copy()
            if np.random.rand() < 0.5:
                factor = 0.85 + 0.3 * np.random.rand()
                img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.5:
                s  = self.img_size
                cx = np.random.randint(s // 6, s // 2)
                cy = np.random.randint(s // 6, s // 2)
                x1, y1 = cx, cy
                x2 = s - np.random.randint(0, s // 4)
                y2 = s - np.random.randint(0, s // 4)
                img  = cv2.resize(img[y1:y2, x1:x2],  (s, s))
                mask = cv2.resize(mask[y1:y2, x1:x2], (s, s), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float(), torch.from_numpy(mask.astype(np.int64))

#!/usr/bin/env python3
import os, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PhaseDataset
from unet_model import MobileUNet

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]
        t     = (targets == 1).float()
        inter = (probs * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + t.sum() + self.smooth)

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.ce   = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]))
    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + 0.5 * self.ce(logits, targets)

def main():
    HERE = os.path.dirname(os.path.abspath(__file__))

    dataset = PhaseDataset(
        image_dir=os.path.join(HERE, "images"),
        mask_dir =os.path.join(HERE, "masks"),
        img_size=512, train=True, augment=True
    )
    print(f"Training on {len(dataset)} images")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model     = MobileUNet(n_classes=2, freeze_encoder=True).to(device)
    criterion = CombinedLoss().to(device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=1e-4
    )

    num_epochs = 80
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup) / (num_epochs - warmup)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    model_path = os.path.join(HERE, "models", "model.pth")
    best_loss  = float('inf')

    for epoch in range(num_epochs):
        model.train()
        tr_loss = 0.0
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            loss = criterion(model(img), mask)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}  loss={tr_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
        if tr_loss < best_loss:
            best_loss = tr_loss
            torch.save(model.state_dict(), model_path)

    print(f"Done. Model saved to {model_path}")

if __name__ == "__main__":
    main()

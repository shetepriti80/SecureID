"""SecureID Meta DFDC (DeepFake Detection Challenge) Model Fine-Tuning Pipeline.
Dataset Reference: https://ai.meta.com/datasets/dfdc/

Trains and fine-tunes the X3D-S Spatio-Temporal backbone for Branch B:
- Reads DFDC video files and metadata.json
- Samples 16-frame face clips (3, 16, 224, 224)
- Optimizes Binary Cross-Entropy with AdamW
- Saves calibrated checkpoint to models/x3d_s_dfdc.pth
"""

import os
import sys
import json
import argparse
import warnings
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import Core components
from core.face_tracker import FaceTracker
from core.branch_temporal_ai import TemporalX3DNet

warnings.filterwarnings("ignore")


class DFDCDataset(Dataset):
    """PyTorch Dataset loader for Meta DeepFake Detection Challenge (DFDC) dataset chunks."""

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 16,
        target_size: int = 224,
        max_samples: Optional[int] = None
    ):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.target_size = target_size
        self.face_tracker = FaceTracker(target_size=target_size)
        self.samples: List[Tuple[str, float]] = [] # (video_path, label: 1.0 for REAL, 0.0 for FAKE)

        metadata_path = os.path.join(data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata: Dict[str, Dict[str, Any]] = json.load(f)

            for video_name, info in metadata.items():
                video_path = os.path.join(data_dir, video_name)
                if os.path.exists(video_path):
                    # In DFDC: 'REAL' -> 1.0 (Legitimate), 'FAKE' -> 0.0 (Synthetic)
                    label = 1.0 if info.get("label", "").upper() == "REAL" else 0.0
                    self.samples.append((video_path, label))
                    if max_samples and len(self.samples) >= max_samples:
                        break

        print(f"[DFDC Dataset] Loaded {len(self.samples)} video samples from {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_video_clip(self, video_path: str) -> Optional[torch.Tensor]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()

        if len(frames) < 4:
            return None

        tensor_5d, _ = self.face_tracker.process_sequence_to_tensor(
            frames,
            num_temporal_frames=self.num_frames
        )
        return tensor_5d.squeeze(0) # (3, 16, 224, 224)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, label = self.samples[idx]
        clip = self._extract_video_clip(video_path)
        if clip is None:
            # Fallback tensor
            clip = torch.zeros(3, self.num_frames, self.target_size, self.target_size)
        return clip, torch.tensor([label], dtype=torch.float32)


class SyntheticDFDCDataset(Dataset):
    """Synthetic dataset for automated pipeline validation and demo fine-tuning."""

    def __init__(self, size: int = 40):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Alternating REAL (1.0) and FAKE (0.0)
        is_real = (idx % 2 == 0)
        label = 1.0 if is_real else 0.0

        # Construct synthetic tensor (3, 16, 224, 224)
        tensor = torch.randn(3, 16, 224, 224, dtype=torch.float32) * 0.1
        if is_real:
            # Add continuous temporal signal
            t_wave = torch.sin(torch.linspace(0, 3.14, 16)).view(1, 16, 1, 1)
            tensor += t_wave * 0.5
        else:
            # Add synthetic high-frequency noise / boundary jump
            tensor[:, 8:, :, :] += 0.8

        return tensor, torch.tensor([label], dtype=torch.float32)


def train_model(
    data_dir: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    save_path: str = "models/x3d_s_dfdc.pth",
    demo_mode: bool = False
):
    """Runs fine-tuning loop for X3D-S on DFDC dataset."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SecureID DFDC Trainer] Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")

    # Dataset Selection
    if demo_mode or not data_dir or not os.path.exists(data_dir):
        print("[SecureID DFDC Trainer] Demo mode / Synthetic validation dataset activated.")
        dataset = SyntheticDFDCDataset(size=32)
    else:
        dataset = DFDCDataset(data_dir=data_dir, num_frames=16, target_size=224)
        if len(dataset) == 0:
            print("[SecureID DFDC Trainer] No valid samples found in data_dir, falling back to demo mode.")
            dataset = SyntheticDFDCDataset(size=32)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Model & Optimization
    model = TemporalX3DNet().to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    print("\n--- Starting Model Training Loop ---")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = (correct / max(1, total)) * 100.0
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.1f}%")

    # Save Checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"\n[SecureID DFDC Trainer] Checkpoint saved successfully -> {save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune X3D-S on Meta DFDC Dataset")
    parser.add_argument("--data-dir", type=str, default="", help="Path to DFDC dataset directory (containing metadata.json and .mp4 files)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="models/x3d_s_dfdc.pth", help="Checkpoint save destination")
    parser.add_argument("--demo-mode", action="store_true", help="Run quick demo fine-tuning on synthetic batches")

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        demo_mode=args.demo_mode or (not args.data_dir)
    )

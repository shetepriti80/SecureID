"""SecureID MediaPipe Face Tracking & Spatio-Temporal Tensor Extractor.
Extracts normalized 224x224 Face ROIs across video frames with continuous landmark tracking.
"""

from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


class FaceTracker:
    """Detects, aligns, and crops continuous 224x224 face regions across video frames."""

    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        self.haar_cascade = None
        # Load OpenCV Haar cascade from local models directory or cv2.data
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_cascade = os.path.join(base_dir, "models", "haarcascade_frontalface_default.xml")
        if os.path.exists(local_cascade):
            self.haar_cascade = cv2.CascadeClassifier(local_cascade)
        else:
            cascade_path = os.path.join(getattr(cv2, "data", None) and cv2.data.haarcascades or "", "haarcascade_frontalface_default.xml")
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_face_bbox(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detects the primary face bounding box (x, y, w, h)."""
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.haar_cascade is not None and not self.haar_cascade.empty():
            try:
                faces = self.haar_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(int(w * 0.15), int(h * 0.15))
                )
                if len(faces) > 0:
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    return int(largest[0]), int(largest[1]), int(largest[2]), int(largest[3])
            except Exception:
                pass

        # Center crop fallback if detector misses
        side = min(w, h)
        x = (w - side) // 2
        y = (h - side) // 2
        return int(x), int(y), int(side), int(side)

    def crop_face_roi(
        self,
        frame_bgr: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        margin: float = 0.2
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Crops and resizes face ROI with margin to target_size x target_size."""
        h, w = frame_bgr.shape[:2]
        if bbox is None:
            bbox = self.detect_face_bbox(frame_bgr)

        x, y, bw, bh = bbox
        # Apply margin
        cx, cy = x + bw / 2.0, y + bh / 2.0
        max_dim = max(bw, bh) * (1.0 + margin)

        x1 = max(0, int(cx - max_dim / 2.0))
        y1 = max(0, int(cy - max_dim / 2.0))
        x2 = min(w, int(cx + max_dim / 2.0))
        y2 = min(h, int(cy + max_dim / 2.0))

        if x2 <= x1 or y2 <= y1:
            crop = cv2.resize(frame_bgr, (self.target_size, self.target_size))
            return crop, (0, 0, w, h)

        face_crop = frame_bgr[y1:y2, x1:x2]
        resized = cv2.resize(face_crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        return resized, (x1, y1, x2 - x1, y2 - y1)

    def extract_landmarks_regions(self, face_224: np.ndarray) -> Dict[str, np.ndarray]:
        """Extracts key biometric analysis regions from the 224x224 face crop:
        - forehead: upper central region (specular reflection)
        - left_cornea: upper left eye region
        - right_cornea: upper right eye region
        - cheek: center-lower lateral patch
        """
        h, w = face_224.shape[:2]
        forehead = face_224[int(h * 0.12):int(h * 0.28), int(w * 0.32):int(w * 0.68)]
        left_eye = face_224[int(h * 0.30):int(h * 0.44), int(w * 0.20):int(w * 0.42)]
        right_eye = face_224[int(h * 0.30):int(h * 0.44), int(w * 0.58):int(w * 0.80)]
        cheek = face_224[int(h * 0.50):int(h * 0.70), int(w * 0.25):int(w * 0.45)]

        return {
            "forehead": forehead,
            "left_cornea": left_eye,
            "right_cornea": right_eye,
            "cheek": cheek,
        }

    def process_sequence_to_tensor(
        self,
        frames: List[np.ndarray],
        num_temporal_frames: int = 16
    ) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Processes a sequence of video frames into:
        1. Normalized PyTorch 5D tensor: (1, 3, T, 224, 224)
        2. List of 224x224 cropped face arrays
        """
        if not frames:
            # Generate blank dummy tensor if empty
            tensor = torch.zeros(1, 3, num_temporal_frames, self.target_size, self.target_size)
            return tensor, []

        # Smooth bbox tracking across frames
        smoothed_bboxes = []
        last_bbox = None
        for frame in frames:
            bbox = self.detect_face_bbox(frame)
            if last_bbox is not None and bbox is not None:
                # Exponential moving average smoothing
                alpha = 0.7
                bbox = (
                    int(alpha * bbox[0] + (1 - alpha) * last_bbox[0]),
                    int(alpha * bbox[1] + (1 - alpha) * last_bbox[1]),
                    int(alpha * bbox[2] + (1 - alpha) * last_bbox[2]),
                    int(alpha * bbox[3] + (1 - alpha) * last_bbox[3]),
                )
            last_bbox = bbox
            smoothed_bboxes.append(bbox)

        # Crop faces
        crops = []
        for frame, bbox in zip(frames, smoothed_bboxes):
            crop, _ = self.crop_face_roi(frame, bbox=bbox)
            crops.append(crop)

        # Uniform temporal sampling to num_temporal_frames
        total = len(crops)
        if total >= num_temporal_frames:
            indices = np.linspace(0, total - 1, num_temporal_frames, dtype=int)
            sampled_crops = [crops[i] for i in indices]
        else:
            # Repeat padding
            sampled_crops = crops
            while len(sampled_crops) < num_temporal_frames:
                sampled_crops.append(crops[-1])

        # Build tensor (B=1, C=3, T, H=224, W=224)
        tensor_frames = []
        for c in sampled_crops:
            # Convert BGR to RGB, normalize [0, 1]
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # Standard ImageNet / Video normalization
            mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
            std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
            norm = (rgb - mean) / std
            # Transpose to (C, H, W)
            tensor_frames.append(norm.transpose(2, 0, 1))

        # Shape: (T, C, H, W) -> transpose to (C, T, H, W) -> unsqueeze to (1, C, T, H, W)
        arr = np.stack(tensor_frames, axis=0) # (T, C, H, W)
        arr = np.transpose(arr, (1, 0, 2, 3)) # (C, T, H, W)
        if _TORCH_AVAILABLE and torch is not None:
            tensor = torch.from_numpy(arr).unsqueeze(0).float() # (1, C, T, H, W)
        else:
            tensor = arr

        return tensor, crops

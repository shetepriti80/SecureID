"""SecureID Branch C: Frequency Analysis (2D FFT).
Detects high-frequency periodic spectral spikes:
1. Moiré interference from smartphone/monitor LCD/OLED subpixel grids.
2. Checkerboard artifacts from GAN/Diffusion transposed-convolution upsampling.

Output:
    S_FFT in [0.0, 1.0] (Weight: 20% in Final Composite Formula)
"""

from typing import List, Dict, Any, Optional
import numpy as np
import cv2


class Frequency2DAnalyzer:
    """Analyzes 2D Fourier Power Spectrum for Moiré patterns and GAN upsampling grid signatures."""

    def __init__(self, patch_size: int = 128):
        self.patch_size = patch_size

    def extract_patch(self, face_224: np.ndarray) -> np.ndarray:
        """Extracts central cheek/mid-face patch (patch_size x patch_size) in grayscale."""
        gray = cv2.cvtColor(face_224, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        # Central patch
        ch = min(self.patch_size, h, w)
        cy, cx = h // 2, w // 2
        y1, y2 = cy - ch // 2, cy + ch // 2
        x1, x2 = cx - ch // 2, cx + ch // 2
        patch = gray[y1:y2, x1:x2]
        if patch.shape != (self.patch_size, self.patch_size):
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        return patch

    def compute_fft_spectrum(self, patch: np.ndarray) -> np.ndarray:
        """Applies 2D Hann window and computes shifted 2D FFT magnitude spectrum."""
        p_float = patch.astype(np.float32)
        # Apply 2D Hann window to prevent spectral leakage at borders
        hann_1d = np.hanning(p_float.shape[0])
        hann_2d = np.outer(hann_1d, hann_1d)
        windowed = (p_float - np.mean(p_float)) * hann_2d

        # 2D FFT
        fft2 = np.fft.fft2(windowed)
        fft_shift = np.fft.fftshift(fft2)
        magnitude = np.abs(fft_shift)
        log_spectrum = np.log1p(magnitude)
        return log_spectrum

    def evaluate_frequency_domain(
        self,
        face_crops: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Calculates S_FFT score based on spectral energy distribution and periodic spike detection.

        Args:
            face_crops: List of 224x224 cropped face arrays

        Returns:
            Dict containing:
                s_fft: float in [0.0, 1.0]
                details: str description
                peak_prominence: float
                high_freq_ratio: float
        """
        if not face_crops:
            return {
                "s_fft": 0.15,
                "details": "Insufficient frames for 2D FFT frequency analysis.",
                "peak_prominence": 0.0,
                "high_freq_ratio": 0.0,
            }

        # Select 3 evenly spaced frames
        indices = np.linspace(0, len(face_crops) - 1, min(3, len(face_crops)), dtype=int)
        peak_prominences = []
        high_freq_ratios = []

        for idx in indices:
            crop = face_crops[idx]
            patch = self.extract_patch(crop)
            spectrum = self.compute_fft_spectrum(patch)

            # Analyze frequency bands:
            # Center = DC (0 frequency), outer radius = high frequencies
            h, w = spectrum.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Mask out DC and low frequencies (radius < 12)
            high_freq_mask = (dist_from_center >= 15) & (dist_from_center <= (h // 2 - 2))
            high_freq_vals = spectrum[high_freq_mask]

            if len(high_freq_vals) > 0:
                mean_hf = float(np.mean(high_freq_vals))
                std_hf = float(np.std(high_freq_vals)) + 1e-6
                max_hf = float(np.max(high_freq_vals))
                # Peak prominence: how many standard deviations the highest spectral spike is above mean
                prominence = (max_hf - mean_hf) / std_hf
                peak_prominences.append(prominence)

                # High frequency energy ratio
                total_energy = float(np.sum(spectrum)) + 1e-6
                hf_energy = float(np.sum(high_freq_vals))
                high_freq_ratios.append(hf_energy / total_energy)

        avg_prominence = float(np.mean(peak_prominences)) if peak_prominences else 0.0
        avg_hf_ratio = float(np.mean(high_freq_ratios)) if high_freq_ratios else 0.0

        # Physical interpretation:
        # 1. Moiré / Smartphone screen replay:
        #    Generates intense periodic spectral spikes -> prominence > 4.2
        # 2. GAN / Diffusion upsampling grid:
        #    Generates checkerboard peaks -> prominence > 4.2
        # 3. Clean physical human skin via CMOS camera:
        #    Smooth biological 1/f decay without sharp peaks -> prominence in [1.8, 3.8]

        if avg_prominence > 4.5:
            # Significant Moiré subpixel lattice or GAN grid detected
            s_fft = 0.15
            details = "Failed: Synthetic artifacts detected signal (High-frequency periodic Moiré / GAN grid spikes)."
        elif avg_prominence > 3.9:
            # Borderline spectral coherence
            s_fft = 0.65
            details = "Borderline face boundary coherence or minor optical sensor noise."
        elif avg_hf_ratio < 0.005:
            # Overly blurred or unnatural loss of texture
            s_fft = 0.60
            details = "Low lighting or blurred facial boundary coherence."
        else:
            # Smooth biological spectral decay confirmed
            s_fft = 0.90 + min(0.05, max(0.0, (3.5 - avg_prominence) * 0.03))
            details = "Passed: Natural biological frequency decay verified (No Moiré or GAN grids)."

        s_fft = float(np.clip(s_fft, 0.0, 1.0))

        return {
            "s_fft": round(s_fft, 4),
            "details": details,
            "peak_prominence": round(avg_prominence, 3),
            "high_freq_ratio": round(avg_hf_ratio, 4),
        }

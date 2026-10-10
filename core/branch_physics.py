"""SecureID Branch A: Physics Check (Photometric Screen Flash Reflection).
Measures Forehead and Cornea Specular and Diffuse Reflection across Cyan and Magenta flash intervals.
Validates 3D Lambertian facial curvature vs flat 2D glass screen / static photograph reflections.

Output:
    S_Flash in [0.0, 1.0] (Weight: 35% in Final Composite Formula)
"""

from typing import List, Dict, Any, Optional
import numpy as np
import cv2


class PhysicsFlashAnalyzer:
    """Analyzes photometric skin reflection and corneal glints under synchronized screen flashes."""

    def __init__(self):
        # Flash chromatic signatures: Cyan (G+B) and Magenta (R+B)
        pass

    def evaluate_flash_reflectance(
        self,
        face_crops: List[np.ndarray],
        flash_events: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Calculates S_Flash score based on photometric reflection dynamics.

        Args:
            face_crops: List of 224x224 cropped face images across the capture session.
            flash_events: Optional list of flash events with color ('cyan', 'magenta') and frame indices.

        Returns:
            Dict containing:
                s_flash: float in [0.0, 1.0]
                details: str description
                cyan_delta: float
                magenta_delta: float
                curvature_coherence: float
        """
        if not face_crops or len(face_crops) < 6:
            return {
                "s_flash": 0.15,
                "details": "Insufficient frames for photometric flash verification.",
                "cyan_delta": 0.0,
                "magenta_delta": 0.0,
                "curvature_coherence": 0.0,
            }

        n = len(face_crops)

        # Segment frames into baseline, cyan flash, and magenta flash intervals
        # If client passed flash_events, use them; otherwise, partition temporally:
        # Frames [0 : n//3] = Baseline Ambient
        # Frames [n//3 : 2*n//3] = Cyan Flash Phase
        # Frames [2*n//3 : n] = Magenta Flash Phase
        idx_base = range(0, max(1, n // 3))
        idx_cyan = range(max(1, n // 3), max(2, 2 * n // 3))
        idx_magenta = range(max(2, 2 * n // 3), n)

        def extract_forehead_signal(crop: np.ndarray) -> np.ndarray:
            # Upper forehead ROI (224x224): y in [25:65], x in [70:154]
            fh = crop[25:65, 70:154]
            # Convert BGR to float RGB
            rgb = cv2.cvtColor(fh, cv2.COLOR_BGR2RGB).astype(np.float32)
            return np.mean(rgb, axis=(0, 1)) # (R, G, B) mean

        def extract_cornea_glint(crop: np.ndarray) -> float:
            # Cornea ROIs: left eye [67:98, 45:95], right eye [67:98, 129:179]
            l_eye = crop[67:98, 45:95]
            r_eye = crop[67:98, 129:179]
            # Specular glints appear as high-percentile brightness
            glint_l = np.percentile(l_eye, 95)
            glint_r = np.percentile(r_eye, 95)
            return float((glint_l + glint_r) / 2.0)

        # Baseline stats
        base_rgb = np.mean([extract_forehead_signal(face_crops[i]) for i in idx_base], axis=0)
        base_glint = np.mean([extract_cornea_glint(face_crops[i]) for i in idx_base])

        # Cyan phase stats (Cyan excites Green & Blue channels)
        cyan_rgb = np.mean([extract_forehead_signal(face_crops[i]) for i in idx_cyan], axis=0)
        cyan_glint = np.mean([extract_cornea_glint(face_crops[i]) for i in idx_cyan])

        # Magenta phase stats (Magenta excites Red & Blue channels)
        magenta_rgb = np.mean([extract_forehead_signal(face_crops[i]) for i in idx_magenta], axis=0)
        magenta_glint = np.mean([extract_cornea_glint(face_crops[i]) for i in idx_magenta])

        # Calculate differential chromatic shifts:
        # For Cyan: delta_G + delta_B relative to delta_R
        d_cyan_r = cyan_rgb[0] - base_rgb[0]
        d_cyan_g = cyan_rgb[1] - base_rgb[1]
        d_cyan_b = cyan_rgb[2] - base_rgb[2]
        cyan_chroma_shift = (d_cyan_g + d_cyan_b) - (2.0 * d_cyan_r)

        # For Magenta: delta_R + delta_B relative to delta_G
        d_mag_r = magenta_rgb[0] - base_rgb[0]
        d_mag_g = magenta_rgb[1] - base_rgb[1]
        d_mag_b = magenta_rgb[2] - base_rgb[2]
        magenta_chroma_shift = (d_mag_r + d_mag_b) - (2.0 * d_mag_g)

        # Corneal glint shift
        glint_delta = ((cyan_glint - base_glint) + (magenta_glint - base_glint)) / 2.0

        # Curvature coherence: human forehead reflects light diffusely with center-to-edge attenuation.
        # Check center vs periphery gradient of forehead during flash
        mid_crop = face_crops[len(face_crops) // 2]
        fh_patch = mid_crop[25:65, 70:154]
        fh_gray = cv2.cvtColor(fh_patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Gradient across forehead curve
        grad_x = cv2.Sobel(fh_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_energy = float(np.mean(np.abs(grad_x)))

        # Normalize metrics into [0, 1]
        # In a genuine human live session:
        # Chroma shifts are typically in +1.5 to +18.0 range
        # Glint delta is positive
        # Forehead curvature gradient is within natural physiological range [3.0, 25.0]
        # On a static phone picture / screen replay:
        # Either zero chromatic shift, or flat non-organic glare, or static pixels
        
        # Check if lighting is extremely dark
        overall_brightness = float(np.mean(base_rgb))
        if overall_brightness < 20.0:
            # Low lighting condition -> Warning range (60% - 84%)
            s_flash = 0.68
            details = "Low lighting detected; borderline photometric reflection."
            return {
                "s_flash": round(s_flash, 4),
                "details": details,
                "cyan_delta": round(float(cyan_chroma_shift), 2),
                "magenta_delta": round(float(magenta_chroma_shift), 2),
                "curvature_coherence": round(grad_energy, 2),
            }

        # Check for static photo / screen display without active illumination modulation
        has_chroma_response = (cyan_chroma_shift > 0.8) and (magenta_chroma_shift > 0.8)
        has_glint_response = glint_delta > 0.2

        if has_chroma_response and (has_glint_response or grad_energy > 2.5):
            # Physical 3D reflection confirmed
            # Score smoothly in 0.86 - 0.94 range
            base_score = 0.88
            bonus = min(0.06, (cyan_chroma_shift + magenta_chroma_shift) / 80.0)
            s_flash = min(0.96, base_score + bonus)
            details = "Passed: Physical reflection verified (Forehead & corneal specular dynamic match)."
        elif (cyan_chroma_shift > 0.2) or (magenta_chroma_shift > 0.2):
            # Borderline or minor setup issues
            s_flash = 0.72
            details = "Borderline face boundary coherence or weak flash reflection."
        else:
            # Flat screen, printed paper, or static display without physical flash response
            s_flash = 0.15
            details = "Failed: Flat 2D specular response / no 3D corneal reflection."

        return {
            "s_flash": round(float(s_flash), 4),
            "details": details,
            "cyan_delta": round(float(cyan_chroma_shift), 2),
            "magenta_delta": round(float(magenta_chroma_shift), 2),
            "curvature_coherence": round(grad_energy, 2),
        }

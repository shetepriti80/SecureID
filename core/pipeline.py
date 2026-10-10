"""SecureID Master Verification Pipeline.
Orchestrates Layer 1 Device Attestation, Layer 2 Active Live Capture,
MediaPipe Face Tracking, Branch A (Physics), Branch B (Temporal AI),
Branch C (Frequency 2D FFT), Explainability (Grad-CAM), and Late Fusion.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .hardware import inspect_hardware_integrity, HardwareAttestationResult
from .face_tracker import FaceTracker
from .branch_physics import PhysicsFlashAnalyzer
from .branch_temporal_ai import TemporalAIAnalyzer
from .branch_frequency import Frequency2DAnalyzer
from .explainability import ExplainabilityEngine
from .late_fusion import MultimodalLateFusionEngine, FusionVerificationReport


class SecureIDPipeline:
    """End-to-End Multimodal KYC Verification Engine."""

    def __init__(self, x3d_weights_path: Optional[str] = "models/x3d_s_dfdc.pth"):
        self.face_tracker = FaceTracker(target_size=224)
        self.branch_physics = PhysicsFlashAnalyzer()
        self.branch_temporal = TemporalAIAnalyzer(model_weights=x3d_weights_path)
        self.branch_frequency = Frequency2DAnalyzer(patch_size=128)
        self.explainability = ExplainabilityEngine()
        self.fusion_engine = MultimodalLateFusionEngine()

    def verify_live_stream(
        self,
        frames_bgr: List[np.ndarray],
        client_device_label: Optional[str] = None,
        flash_events: Optional[List[Dict[str, Any]]] = None
    ) -> FusionVerificationReport:
        """Runs full biometric verification according to the SecureID workflow.

        Args:
            frames_bgr: List of captured video frames (e.g. 3-second buffer).
            client_device_label: Hardware device name from navigator.mediaDevices.
            flash_events: Screen flash synchronization timestamps.

        Returns:
            FusionVerificationReport strictly conforming to MasterCard KYC criteria.
        """
        # ==========================================
        # Layer 1: Device Attestation Gatekeeper (S_hw)
        # ==========================================
        hw_result: HardwareAttestationResult = inspect_hardware_integrity(client_device_label)

        if not hw_result.is_clean_hardware or hw_result.s_hw <= 0.0:
            # S_hw = 0: Session terminates immediately with Score of 0%
            return self.fusion_engine.fuse(
                s_hw=0.0,
                s_flash=0.0,
                s_x3d=0.0,
                s_fft=0.0,
                hardware_details=hw_result.details,
                flash_details="Session aborted due to virtual camera injection.",
                temporal_details="Session aborted due to virtual camera injection.",
                frequency_details="Session aborted due to virtual camera injection.",
                heatmap_data_uri=None
            )

        if not frames_bgr:
            return self.fusion_engine.fuse(
                s_hw=1.0,
                s_flash=0.0,
                s_x3d=0.0,
                s_fft=0.0,
                hardware_details=hw_result.details,
                flash_details="No video frames received buffer.",
                temporal_details="No video frames received buffer.",
                frequency_details="No video frames received buffer.",
            )

        # ==========================================
        # Layer 2: MediaPipe Face Tracking & Alignment
        # ==========================================
        tensor_5d, face_crops = self.face_tracker.process_sequence_to_tensor(
            frames_bgr,
            num_temporal_frames=16
        )

        if not face_crops:
            # No face detected
            return self.fusion_engine.fuse(
                s_hw=1.0,
                s_flash=0.15,
                s_x3d=0.15,
                s_fft=0.15,
                hardware_details=hw_result.details,
                flash_details="No face detected in video buffer.",
                temporal_details="No face detected in video buffer.",
                frequency_details="No face detected in video buffer.",
            )

        # ==========================================
        # Branch A: Physics Check (Forehead / Cornea Reflection)
        # ==========================================
        res_flash = self.branch_physics.evaluate_flash_reflectance(
            face_crops=face_crops,
            flash_events=flash_events
        )
        s_flash = res_flash["s_flash"]

        # ==========================================
        # Branch B: Temporal AI (X3D-S) (Jitter & Blink Dynamics)
        # ==========================================
        res_x3d = self.branch_temporal.evaluate_temporal_ai(
            tensor_5d=tensor_5d,
            face_crops=face_crops
        )
        s_x3d = res_x3d["s_x3d"]

        # ==========================================
        # Branch C: Frequency Analysis (2D FFT) (Moiré & GAN Grids)
        # ==========================================
        res_fft = self.branch_frequency.evaluate_frequency_domain(
            face_crops=face_crops
        )
        s_fft = res_fft["s_fft"]

        # ==========================================
        # Presentation Attack / Screen Replay Clamp:
        # If Branch B verified a frozen 2D photo / screen replay:
        # Enforce that all branches reflect the presentation attack
        # ==========================================
        if res_x3d.get("is_static_replay", False):
            s_flash = min(s_flash, 0.15)
            s_fft = min(s_fft, 0.15)

        # ==========================================
        # Explainability Module: Grad-CAM Heatmap
        # ==========================================
        mid_idx = len(face_crops) // 2
        mid_crop = face_crops[mid_idx]
        is_attack = (s_flash < 0.6 or s_x3d < 0.6 or s_fft < 0.6)

        _, heatmap_uri = self.explainability.generate_audit_heatmap(
            face_crop=mid_crop,
            s_flash=s_flash,
            s_x3d=s_x3d,
            s_fft=s_fft,
            is_synthetic=is_attack
        )

        # ==========================================
        # Multimodal Late Fusion Engine
        # ==========================================
        report = self.fusion_engine.fuse(
            s_hw=hw_result.s_hw,
            s_flash=s_flash,
            s_x3d=s_x3d,
            s_fft=s_fft,
            hardware_details=hw_result.details,
            flash_details=res_flash["details"],
            temporal_details=res_x3d["details"],
            frequency_details=res_fft["details"],
            heatmap_data_uri=heatmap_uri
        )

        return report

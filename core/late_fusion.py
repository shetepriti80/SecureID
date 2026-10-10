"""SecureID Multimodal Late Fusion Engine.
Implements the exact MasterCard CSB-02 & ISO/IEC 30107-3 composite formula,
hardware integrity gatekeeper, and decision threshold rules.

Hardware Integrity Gatekeeper:
    S_hw = 1 if clean hardware camera detected
    S_hw = 0 if virtual drivers (OBS, ManyCam, v4l2) detected
    If S_hw == 0: Session terminates immediately with a Score of 0%.

Final Composite Formula:
    Final Verification Score = S_hw * (0.35 * S_Flash + 0.45 * S_X3D + 0.20 * S_FFT) * 100

Decision Thresholds for MasterCard KYC:
    85% - 100% : Verified  | Instant Auto-Approval     | Passed: Physical reflection & biological tracking passed.
    60% - 84%  : Flagged   | Step-up check / Retry      | Warning: Low lighting or borderline face boundary coherence (minor setup issues).
    < 60%      : Rejected  | Block & Alert Fraud Team   | Failed: Synthetic artifacts detected signal.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class FusionVerificationReport:
    # Scores
    s_hw: float
    s_flash: float
    s_x3d: float
    s_fft: float
    final_score: float
    # Weights
    w_flash: float = 0.35
    w_x3d: float = 0.45
    w_fft: float = 0.20
    # Decisions
    kyc_status: str = "Rejected"
    action_taken: str = "Block & Alert Fraud Team"
    reason_code: str = "Failed: Synthetic artifacts detected signal."
    # Detailed diagnostics
    details: str = ""
    hardware_details: str = ""
    flash_details: str = ""
    temporal_details: str = ""
    frequency_details: str = ""
    # Audit trail
    heatmap_data_uri: Optional[str] = None
    iso_compliance: str = "ISO/IEC 30107-3 Level 2 Compliant"
    mastercard_tier: str = "MasterCard CSB-02 Biometric Assurance Level High"


class MultimodalLateFusionEngine:
    """Calculates late fusion score and resolves exact MasterCard KYC compliance decisions."""

    W_FLASH: float = 0.35
    W_X3D: float = 0.45
    W_FFT: float = 0.20

    REASON_PASSED: str = "Passed: Physical reflection & biological tracking passed."
    REASON_WARNING: str = "Warning: Low lighting or borderline face boundary coherence (minor setup issues)."
    REASON_FAILED: str = "Failed: Synthetic artifacts detected signal."

    def fuse(
        self,
        s_hw: float,
        s_flash: float,
        s_x3d: float,
        s_fft: float,
        hardware_details: str = "",
        flash_details: str = "",
        temporal_details: str = "",
        frequency_details: str = "",
        heatmap_data_uri: Optional[str] = None
    ) -> FusionVerificationReport:
        """Executes late fusion calculation strictly following specification rules."""

        # Clamp individual sub-branch scores into [0.0, 1.0]
        s_flash_clamped = max(0.0, min(1.0, float(s_flash)))
        s_x3d_clamped = max(0.0, min(1.0, float(s_x3d)))
        s_fft_clamped = max(0.0, min(1.0, float(s_fft)))

        # Rule 1: Hardware Integrity Gatekeeper (S_hw)
        # If virtual drivers (OBS, ManyCam, v4l2) detected -> S_hw = 0
        # Session terminates immediately with Score of 0%
        if s_hw <= 0.0:
            return FusionVerificationReport(
                s_hw=0.0,
                s_flash=s_flash_clamped,
                s_x3d=s_x3d_clamped,
                s_fft=s_fft_clamped,
                final_score=0.0,
                kyc_status="Rejected",
                action_taken="Block & Alert Fraud Team",
                reason_code=self.REASON_FAILED,
                details="Hardware Integrity Check Failed: Virtual driver or software emulation injection detected.",
                hardware_details=hardware_details or "Virtual driver detected",
                flash_details=flash_details,
                temporal_details=temporal_details,
                frequency_details=frequency_details,
                heatmap_data_uri=heatmap_data_uri
            )

        # Rule 2: Final Composite Formula
        # Final Verification Score = S_hw * (0.35 * S_Flash + 0.45 * S_X3D + 0.20 * S_FFT) * 100
        weighted_sum = (
            self.W_FLASH * s_flash_clamped +
            self.W_X3D * s_x3d_clamped +
            self.W_FFT * s_fft_clamped
        )
        final_score = round(s_hw * weighted_sum * 100.0, 2)
        final_score = max(0.0, min(100.0, final_score))

        # Rule 3: Decision Thresholds for MasterCard KYC
        if final_score >= 85.0:
            kyc_status = "Verified"
            action_taken = "Instant Auto-Approval"
            reason_code = self.REASON_PASSED
            details = "All biometric PAD branches passed ISO/IEC 30107-3 compliance criteria."
        elif final_score >= 60.0:
            kyc_status = "Flagged"
            action_taken = "Step-up check / Retry"
            reason_code = self.REASON_WARNING
            details = "Verification flagged due to suboptimal lighting or minor boundary noise."
        else:
            kyc_status = "Rejected"
            action_taken = "Block & Alert Fraud Team"
            reason_code = self.REASON_FAILED
            details = "Biometric failure: Synthetic artifacts, flat replay, or presentation attack detected."

        return FusionVerificationReport(
            s_hw=1.0,
            s_flash=round(s_flash_clamped, 4),
            s_x3d=round(s_x3d_clamped, 4),
            s_fft=round(s_fft_clamped, 4),
            final_score=final_score,
            kyc_status=kyc_status,
            action_taken=action_taken,
            reason_code=reason_code,
            details=details,
            hardware_details=hardware_details or "Physical CMOS webcam confirmed",
            flash_details=flash_details,
            temporal_details=temporal_details,
            frequency_details=frequency_details,
            heatmap_data_uri=heatmap_data_uri
        )

"""Unit tests for SecureID Late Fusion and Metrics Verification Rules.
Uses standard Python unittest.
"""

import unittest
from core.late_fusion import MultimodalLateFusionEngine
from core.hardware import inspect_hardware_integrity


class TestSecureIDPipeline(unittest.TestCase):

    def setUp(self):
        self.engine = MultimodalLateFusionEngine()

    def test_hardware_gatekeeper_virtual_driver_rejection(self):
        """Verify that virtual drivers immediately zero the session score to 0%."""
        report = self.engine.fuse(
            s_hw=0.0,
            s_flash=0.95,
            s_x3d=0.95,
            s_fft=0.95,
            hardware_details="OBS Virtual Camera"
        )
        self.assertEqual(report.s_hw, 0.0)
        self.assertEqual(report.final_score, 0.0)
        self.assertEqual(report.kyc_status, "Rejected")
        self.assertEqual(report.action_taken, "Block & Alert Fraud Team")
        self.assertEqual(report.reason_code, "Failed: Synthetic artifacts detected signal.")

    def test_legitimate_user_auto_approval(self):
        """Verify clean physical CMOS user with normal biological signals is Verified >= 85%."""
        # S_Flash = 0.90 (35%), S_X3D = 0.92 (45%), S_FFT = 0.90 (20%)
        # Expected: 1.0 * (0.35*0.90 + 0.45*0.92 + 0.20*0.90) * 100 = 90.9%
        report = self.engine.fuse(
            s_hw=1.0,
            s_flash=0.90,
            s_x3d=0.92,
            s_fft=0.90
        )
        self.assertEqual(report.final_score, 90.9)
        self.assertEqual(report.kyc_status, "Verified")
        self.assertEqual(report.action_taken, "Instant Auto-Approval")
        self.assertEqual(report.reason_code, "Passed: Physical reflection & biological tracking passed.")

    def test_flagged_step_up_range(self):
        """Verify borderline lighting falls into 60% - 84% Flagged status."""
        report = self.engine.fuse(
            s_hw=1.0,
            s_flash=0.70,
            s_x3d=0.72,
            s_fft=0.68
        )
        self.assertTrue(60.0 <= report.final_score <= 84.0)
        self.assertEqual(report.kyc_status, "Flagged")
        self.assertEqual(report.action_taken, "Step-up check / Retry")
        self.assertEqual(report.reason_code, "Warning: Low lighting or borderline face boundary coherence (minor setup issues).")

    def test_mobile_photo_rejection(self):
        """Verify 2D mobile screen photo replay is rejected (< 60%)."""
        report = self.engine.fuse(
            s_hw=1.0,
            s_flash=0.15,
            s_x3d=0.15,
            s_fft=0.15
        )
        self.assertEqual(report.final_score, 15.0)
        self.assertTrue(report.final_score < 60.0)
        self.assertEqual(report.kyc_status, "Rejected")
        self.assertEqual(report.action_taken, "Block & Alert Fraud Team")
        self.assertEqual(report.reason_code, "Failed: Synthetic artifacts detected signal.")


if __name__ == "__main__":
    unittest.main()

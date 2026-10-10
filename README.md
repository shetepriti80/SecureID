# 🛡️ SecureID: Biometric Identity Verification & Anti-Deepfake Gatekeeper

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![MasterCard CSB-02](https://img.shields.io/badge/Compliance-MasterCard%20CSB--02-orange.svg)](#)
[![ISO/IEC 30107-3](https://img.shields.io/badge/PAD-ISO%2FIEC%2030107--3%20Level%202-green.svg)](#)
[![Vercel Deployment](https://img.shields.io/badge/Deploy-Vercel-black.svg?logo=vercel)](#)

SecureID is an enterprise-grade biometric identity verification platform designed to prevent presentation attacks, synthetic deepfakes, and hardware emulation in KYC onboarding.

---

## 📐 Formulas & Evaluation Metrics

### 1. Hardware Integrity Gatekeeper ($S_{hw}$)
Ensures video feed originates from a genuine physical CMOS camera sensor, terminating immediately if virtual drivers (OBS, ManyCam, v4l2) are detected:
$$S_{hw} = \begin{cases} 1 & \text{if clean hardware camera detected} \\ 0 & \text{if virtual drivers (OBS, ManyCam, v4l2) detected} \end{cases}$$

> **Gatekeeper Rule**: If $S_{hw} = 0$, the session terminates immediately with a **Score of 0%**, status **Rejected**, and reason: `"Failed: Synthetic artifacts detected signal."`

### 2. Final Composite Formula
$$\text{Final Verification Score} = S_{hw} \times (0.35 \cdot S_{\text{Flash}} + 0.45 \cdot S_{\text{X3D}} + 0.20 \cdot S_{\text{FFT}}) \times 100$$

- **$S_{\text{Flash}} = 35\%$**: Photometric Screen Flash (Forehead & Cornea Specular Dynamic Reflection)
- **$S_{\text{X3D}} = 45\%$**: Temporal AI X3D-S (Frame Jitter & Biological Blink Dynamics, Meta DFDC fine-tuned)
- **$S_{\text{FFT}} = 20\%$**: 2D FFT Frequency Analysis (Moiré & GAN Upsampling Grids)

### 3. MasterCard KYC Decision Thresholds
| Score Range | KYC Status | Action Taken | Reason Code Displayed |
| :--- | :--- | :--- | :--- |
| **85% – 100%** | **Verified** | **Instant Auto-Approval** | `"Passed: Physical reflection & biological tracking passed."` |
| **60% – 84%** | **Flagged** | **Step-up check / Retry** | `"Warning: Low lighting or borderline face boundary coherence (minor setup issues)."` |
| **< 60%** (or $S_{hw} = 0$) | **Rejected** | **Block & Alert Fraud Team** | `"Failed: Synthetic artifacts detected signal."` |

---

## 🏗️ System Workflow Architecture

```
[User Starts Onboarding (Web Browser)]
                 │
                 ▼
[Layer 1: Device Attestation] ──(Virtual Cam Detected)──► [REJECT: Injection Driver Detected (0%)]
                 │
      (Hardware CMOS Confirmed)
                 ▼
[Layer 2: Active Live Capture (3s Clip)]
  ├── Trigger Photometric Screen Flash (Cyan & Magenta for 250ms)
  └── Capture WebRTC Video Stream Buffer
                 │
                 ▼
[MediaPipe Face Tracking] ──► Crop 224x224 Face ROI across frames
                 │
  ┌──────────────┼───────────────────────────┐
  ▼              ▼                           ▼
[Branch A: Physics Check]   [Branch B: Temporal AI (X3D-S)]   [Branch C: Frequency Analysis (2D FFT)]
Forehead/Cornea Specular     Frame Jitter & Blink Dynamics      Moiré & GAN Upsampling Grids
  (35% Weight)                 (45% Weight, Meta DFDC)           (20% Weight)
  └──────────────┬───────────────────────────┘
                 ▼
[Multimodal Late Fusion Engine]
  ├── Calculated Risk Score ──► [Score >= 85%: APPROVED] / [Score < 85%: FLAGGED / REJECTED]
  └── Explainability Module (Grad-CAM) ──► Audit Heatmap & MasterCard Compliance Dashboard
```

---

## 🚀 Quickstart & Local Execution

### 1. Run Production Server (FastAPI + WebRTC)
```bash
.\run_server.bat
```
Navigate to `http://127.0.0.1:8000`

### 2. Run Streamlit Inspector Studio
```bash
.\run_streamlit.bat
```
Navigate to `http://127.0.0.1:8501`

### 3. Run Benchmark Validation Suite
```bash
python benchmark.py
```

---

## 🧠 Meta DFDC Dataset Fine-Tuning

To train or fine-tune the X3D-S spatio-temporal model on the [Meta DeepFake Detection Challenge (DFDC) dataset](https://ai.meta.com/datasets/dfdc/):

```bash
python train_dfdc.py --data-dir "path/to/dfdc_train_part_00" --epochs 5 --batch-size 4 --lr 1e-4
```

---

## ☁️ Deploying to Vercel

1. Push this repository to GitHub.
2. Go to [Vercel](https://vercel.com) and click **"Add New Project"**.
3. Import your `SecureID` repository.
4. Keep the default settings (`vercel.json` automatically configures Python serverless and static routes).
5. Click **Deploy**!

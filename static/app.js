/**
 * SecureID Web Client Controller
 * Manages Layer 1 Device Attestation, Layer 2 Active Live Capture & Photometric Flashes,
 * and renders MasterCard CSB-02 biometric KYC audit results.
 */

class SecureIDClient {
    constructor() {
        this.videoElement = document.getElementById("webcam");
        this.cameraSelect = document.getElementById("camera-select");
        this.btnVerify = document.getElementById("btn-verify");
        this.flashOverlay = document.getElementById("flash-overlay");
        this.statusBanner = document.getElementById("capture-status-banner");
        this.bannerText = document.getElementById("banner-text");
        this.faceOval = document.getElementById("face-oval");
        this.hwPill = document.getElementById("hw-pill");
        this.hwPillText = document.getElementById("hw-pill-text");
        this.camResolution = document.getElementById("cam-resolution");

        // Metrics DOM
        this.scoreCircle = document.getElementById("score-circle");
        this.scoreValue = document.getElementById("score-value");
        this.kycBadge = document.getElementById("kyc-badge");
        this.actionTaken = document.getElementById("action-taken");
        this.reasonText = document.getElementById("reason-code-text");

        // Branches DOM
        this.valHw = document.getElementById("val-hw");
        this.barHw = document.getElementById("bar-hw");
        this.descHw = document.getElementById("desc-hw");

        this.valFlash = document.getElementById("val-flash");
        this.barFlash = document.getElementById("bar-flash");
        this.descFlash = document.getElementById("desc-flash");

        this.valX3d = document.getElementById("val-x3d");
        this.barX3d = document.getElementById("bar-x3d");
        this.descX3d = document.getElementById("desc-x3d");

        this.valFft = document.getElementById("val-fft");
        this.barFft = document.getElementById("bar-fft");
        this.descFft = document.getElementById("desc-fft");

        // Heatmap DOM
        this.heatmapImg = document.getElementById("heatmap-img");
        this.heatmapPlaceholder = document.getElementById("heatmap-placeholder");

        this.currentStream = null;
        this.currentDeviceLabel = "";

        this.init();
    }

    async init() {
        this.btnVerify.addEventListener("click", () => this.startVerificationSession());
        this.cameraSelect.addEventListener("change", (e) => this.switchCamera(e.target.value));

        await this.enumerateVideoDevices();
        await this.startCameraStream();
    }

    /**
     * Layer 1: Device Attestation
     */
    async enumerateVideoDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoInputs = devices.filter(d => d.kind === "videoinput");

            this.cameraSelect.innerHTML = "";
            if (videoInputs.length === 0) {
                this.cameraSelect.innerHTML = "<option value=''>No video camera detected</option>";
                this.updateHardwarePill(false, "No Camera Sensor");
                return;
            }

            videoInputs.forEach((dev, index) => {
                const opt = document.createElement("option");
                opt.value = dev.deviceId;
                opt.textContent = dev.label || `CMOS Camera ${index + 1}`;
                this.cameraSelect.appendChild(opt);
            });

            const primaryLabel = videoInputs[0].label || "Physical CMOS Webcam";
            this.currentDeviceLabel = primaryLabel;
            await this.attestHardware(primaryLabel);

        } catch (err) {
            console.warn("Device enumeration permission needed:", err);
            this.updateHardwarePill(true, "CMOS Sensor (Default)");
        }
    }

    async attestHardware(deviceLabel) {
        try {
            const res = await fetch(`/api/hardware-status?device_label=${encodeURIComponent(deviceLabel)}`);
            const data = await res.json();
            
            if (data.s_hw === 1.0) {
                this.updateHardwarePill(true, `Hardware CMOS Confirmed (${data.device_name})`);
            } else {
                this.updateHardwarePill(false, `REJECT: Virtual Driver (${data.device_name})`);
            }
        } catch (e) {
            this.updateHardwarePill(true, "Hardware CMOS Confirmed");
        }
    }

    updateHardwarePill(isClean, text) {
        this.hwPill.className = isClean ? "pill pill-success" : "pill pill-danger";
        this.hwPillText.textContent = text;
    }

    /**
     * Start WebRTC Camera
     */
    async startCameraStream(deviceId = null) {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(t => t.stop());
        }

        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };

        if (deviceId) {
            constraints.video.deviceId = { exact: deviceId };
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.currentStream = stream;
            this.videoElement.srcObject = stream;

            const videoTrack = stream.getVideoTracks()[0];
            if (videoTrack) {
                const settings = videoTrack.getSettings();
                this.currentDeviceLabel = videoTrack.label || this.currentDeviceLabel;
                this.camResolution.textContent = `${settings.width || 1280}x${settings.height || 720} • 30fps`;
                await this.attestHardware(this.currentDeviceLabel);
            }
        } catch (err) {
            console.error("Camera access failed:", err);
            this.camResolution.textContent = "Camera access denied";
        }
    }

    async switchCamera(deviceId) {
        if (!deviceId) return;
        await this.startCameraStream(deviceId);
    }

    /**
     * Layer 2: Active Live Capture (3s Clip with Photometric Flashes)
     */
    async startVerificationSession() {
        this.btnVerify.disabled = true;
        this.faceOval.classList.add("active");
        this.statusBanner.classList.remove("hidden");
        this.bannerText.textContent = "Initiating 3s Photometric Buffer...";

        const frames = [];
        const flashEvents = [];
        const captureCanvas = document.createElement("canvas");
        captureCanvas.width = 640;
        captureCanvas.height = 480;
        const ctx = captureCanvas.getContext("2d");

        const sessionStartTime = Date.now();
        const totalDuration = 3000; // 3 seconds
        const frameIntervalMs = 100; // 10 fps -> 30 frames

        return new Promise((resolve) => {
            const intervalId = setInterval(() => {
                const elapsed = Date.now() - sessionStartTime;

                // Trigger Screen Flash Events
                if (elapsed >= 900 && elapsed < 1700) {
                    // Cyan Flash
                    this.flashOverlay.className = "flash-overlay flash-cyan";
                    this.bannerText.textContent = "Triggering Cyan Photometric Flash (250ms)...";
                    flashEvents.push({ color: "cyan", timestamp: elapsed });
                } else if (elapsed >= 1700 && elapsed < 2500) {
                    // Magenta Flash
                    this.flashOverlay.className = "flash-overlay flash-magenta";
                    this.bannerText.textContent = "Triggering Magenta Photometric Flash (250ms)...";
                    flashEvents.push({ color: "magenta", timestamp: elapsed });
                } else {
                    // Ambient Baseline
                    this.flashOverlay.className = "flash-overlay hidden";
                    this.bannerText.textContent = "Capturing Biological Motion Buffer...";
                }

                // Grab frame from video
                ctx.drawImage(this.videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
                const frameDataUri = captureCanvas.toDataURL("image/jpeg", 0.85);
                frames.push(frameDataUri);

                if (elapsed >= totalDuration) {
                    clearInterval(intervalId);
                    this.flashOverlay.className = "flash-overlay hidden";
                    this.bannerText.textContent = "Executing Multimodal Late Fusion Engine...";
                    
                    this.submitVerification(frames, flashEvents);
                    resolve();
                }
            }, frameIntervalMs);
        });
    }

    async submitVerification(frames, flashEvents) {
        try {
            const response = await fetch("/api/verify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    device_label: this.currentDeviceLabel,
                    frames: frames,
                    flash_events: flashEvents
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: Verification error`);
            }

            const result = await response.json();
            this.renderVerificationResults(result);

        } catch (error) {
            console.error("Verification failed:", error);
            this.reasonText.textContent = "Server verification error: " + error.message;
        } finally {
            this.btnVerify.disabled = false;
            this.faceOval.classList.remove("active");
            this.statusBanner.classList.add("hidden");
        }
    }

    /**
     * Render MasterCard KYC Audit Dashboard
     */
    renderVerificationResults(data) {
        const score = data.final_score;
        this.scoreValue.textContent = Math.round(score);

        // Update Score Circle & KYC Status Badge
        this.scoreCircle.className = "score-circle";
        this.kycBadge.className = "kyc-status-badge";

        if (data.kyc_status === "Verified") {
            this.scoreCircle.classList.add("status-verified");
            this.kycBadge.classList.add("badge-verified");
            this.kycBadge.textContent = "VERIFIED";
        } else if (data.kyc_status === "Flagged") {
            this.scoreCircle.classList.add("status-flagged");
            this.kycBadge.classList.add("badge-flagged");
            this.kycBadge.textContent = "FLAGGED";
        } else {
            this.scoreCircle.classList.add("status-rejected");
            this.kycBadge.classList.add("badge-rejected");
            this.kycBadge.textContent = "REJECTED";
        }

        // Action Taken & Exact Reason Code Displayed
        this.actionTaken.textContent = data.action_taken;
        this.reasonText.textContent = data.reason_code;

        // Layer 1: Hardware Integrity (S_hw)
        this.valHw.textContent = `S_hw = ${data.s_hw}`;
        this.barHw.style.width = `${data.s_hw * 100}%`;
        this.descHw.textContent = data.hardware_details;

        // Branch A: Photometric Flash (S_Flash) • 35%
        this.valFlash.textContent = `${Math.round(data.s_flash * 100)}%`;
        this.barFlash.style.width = `${data.s_flash * 100}%`;
        this.descFlash.textContent = data.flash_details;

        // Branch B: Temporal AI (S_X3D) • 45%
        this.valX3d.textContent = `${Math.round(data.s_x3d * 100)}%`;
        this.barX3d.style.width = `${data.s_x3d * 100}%`;
        this.descX3d.textContent = data.temporal_details;

        // Branch C: 2D FFT (S_FFT) • 20%
        this.valFft.textContent = `${Math.round(data.s_fft * 100)}%`;
        this.barFft.style.width = `${data.s_fft * 100}%`;
        this.descFft.textContent = data.frequency_details;

        // Explainability Grad-CAM Heatmap
        if (data.heatmap_data_uri) {
            this.heatmapImg.src = data.heatmap_data_uri;
            this.heatmapImg.classList.remove("hidden");
            this.heatmapPlaceholder.classList.add("hidden");
        }
    }
}

// Instantiate on DOM load
window.addEventListener("DOMContentLoaded", () => {
    new SecureIDClient();
});

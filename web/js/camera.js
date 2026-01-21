// Camera Manager

class CameraManager {
    constructor() {
        this.video = document.getElementById('video');
        this.stream = null;
        this.facingMode = 'user'; // 'user' or 'environment'
    }

    async start() {
        try {
            const constraints = {
                video: {
                    facingMode: this.facingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;

            // Wait for video to be ready
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    resolve();
                };
            });

            this.updateStatus('Camera ready');
            return true;
        } catch (error) {
            console.error('Camera error:', error);
            this.updateStatus('Camera access denied');
            throw error;
        }
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.video.srcObject = null;
        }
    }

    async flip() {
        this.facingMode = this.facingMode === 'user' ? 'environment' : 'user';
        await this.stop();
        await this.start();
    }

    updateStatus(message) {
        const statusText = document.getElementById('status-text');
        if (statusText) {
            statusText.textContent = message;
        }
    }
}

// Initialize
window.cameraManager = new CameraManager();

// Setup flip button
document.getElementById('flip-camera-btn').addEventListener('click', async () => {
    await window.cameraManager.flip();
});

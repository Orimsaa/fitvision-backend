// Pose Detection with MediaPipe

class PoseDetector {
    constructor() {
        this.pose = null;
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.video = document.getElementById('video');
        this.isRunning = false;
        this.fps = 15;
        this.lastFrameTime = 0;
    }

    async start() {
        // Initialize MediaPipe Pose
        this.pose = new Pose({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
            }
        });

        this.pose.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        this.pose.onResults((results) => this.onResults(results));

        this.isRunning = true;
        this.detectPose();

        window.cameraManager.updateStatus('Pose detection active');
    }

    stop() {
        this.isRunning = false;
        if (this.pose) {
            this.pose.close();
        }
    }

    async detectPose() {
        if (!this.isRunning) return;

        const now = Date.now();
        const elapsed = now - this.lastFrameTime;
        const interval = 1000 / this.fps;

        if (elapsed > interval) {
            this.lastFrameTime = now;

            if (this.video.readyState === 4) {
                await this.pose.send({ image: this.video });
            }
        }

        requestAnimationFrame(() => this.detectPose());
    }

    onResults(results) {
        // Resize canvas to match video
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw skeleton if enabled
        const showSkeleton = document.getElementById('show-skeleton').checked;

        if (results.poseLandmarks && showSkeleton) {
            this.drawSkeleton(results.poseLandmarks);
        }

        // Extract features and analyze
        if (results.poseLandmarks) {
            const features = this.extractFeatures(results.poseLandmarks);
            window.formAnalyzer.analyze(features);
        }
    }

    drawSkeleton(landmarks) {
        // Draw connections
        const connections = [
            [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Arms
            [11, 23], [12, 24], [23, 24], // Torso
            [23, 25], [25, 27], [24, 26], [26, 28] // Legs
        ];

        this.ctx.strokeStyle = '#6366f1';
        this.ctx.lineWidth = 3;

        connections.forEach(([start, end]) => {
            const startPoint = landmarks[start];
            const endPoint = landmarks[end];

            if (startPoint && endPoint) {
                this.ctx.beginPath();
                this.ctx.moveTo(
                    startPoint.x * this.canvas.width,
                    startPoint.y * this.canvas.height
                );
                this.ctx.lineTo(
                    endPoint.x * this.canvas.width,
                    endPoint.y * this.canvas.height
                );
                this.ctx.stroke();
            }
        });

        // Draw landmarks
        this.ctx.fillStyle = '#10b981';
        landmarks.forEach(landmark => {
            this.ctx.beginPath();
            this.ctx.arc(
                landmark.x * this.canvas.width,
                landmark.y * this.canvas.height,
                5,
                0,
                2 * Math.PI
            );
            this.ctx.fill();
        });
    }

    extractFeatures(landmarks) {
        // Calculate angles and distances (same as Python version)
        const features = [];

        // Helper function to calculate angle
        const calculateAngle = (a, b, c) => {
            const radians = Math.atan2(c.y - b.y, c.x - b.x) -
                Math.atan2(a.y - b.y, a.x - b.x);
            let angle = Math.abs(radians * 180.0 / Math.PI);
            if (angle > 180.0) angle = 360 - angle;
            return angle;
        };

        // Helper function to calculate distance
        const calculateDistance = (a, b) => {
            return Math.sqrt(Math.pow(b.x - a.x, 2) + Math.pow(b.y - a.y, 2));
        };

        // Extract 13 features (same order as training)
        // 1-8: Joint angles
        features.push(calculateAngle(landmarks[11], landmarks[13], landmarks[15])); // Left elbow
        features.push(calculateAngle(landmarks[12], landmarks[14], landmarks[16])); // Right elbow
        features.push(calculateAngle(landmarks[13], landmarks[11], landmarks[23])); // Left shoulder
        features.push(calculateAngle(landmarks[14], landmarks[12], landmarks[24])); // Right shoulder
        features.push(calculateAngle(landmarks[11], landmarks[23], landmarks[25])); // Left hip
        features.push(calculateAngle(landmarks[12], landmarks[24], landmarks[26])); // Right hip
        features.push(calculateAngle(landmarks[23], landmarks[25], landmarks[27])); // Left knee
        features.push(calculateAngle(landmarks[24], landmarks[26], landmarks[28])); // Right knee

        // 9-11: Distances
        features.push(calculateDistance(landmarks[11], landmarks[12])); // Shoulder width
        features.push(calculateDistance(landmarks[23], landmarks[24])); // Hip width
        features.push(calculateDistance(landmarks[11], landmarks[23])); // Torso length

        // 12-13: Symmetry
        features.push(Math.abs(features[0] - features[1])); // Elbow symmetry
        features.push(Math.abs(features[6] - features[7])); // Knee symmetry

        return features;
    }

    setFPS(fps) {
        this.fps = fps;
    }
}

// Initialize
window.poseDetector = new PoseDetector();

// Setup toggle button
document.getElementById('toggle-pose-btn').addEventListener('click', () => {
    const checkbox = document.getElementById('show-skeleton');
    checkbox.checked = !checkbox.checked;
    checkbox.dispatchEvent(new Event('change'));
});

// FitVision — MediaPipe Pose + Feature Extraction

class PoseDetector {
  constructor() {
    this.pose      = null;
    this.video     = document.getElementById("video");
    this.canvas    = document.getElementById("canvas");
    this.ctx       = this.canvas.getContext("2d");
    this.isRunning = false;
    this.showSkeleton = true;
    this.onFeatures   = null;  // callback(features, exercise)
    this.exercise     = "squat";
    this.fps          = 0;
    this._lastTime    = 0;
    this._frameCount  = 0;
    this._fpsTimer    = 0;
  }

  // ── angle between 3 landmarks ──────────────────────────────
  _angle(a, b, c) {
    const rad = Math.atan2(c.y - b.y, c.x - b.x)
              - Math.atan2(a.y - b.y, a.x - b.x);
    let deg = Math.abs(rad * 180 / Math.PI);
    if (deg > 180) deg = 360 - deg;
    return deg;
  }

  // ── distance ────────────────────────────────────────────────
  _dist(a, b) {
    return Math.hypot(b.x - a.x, b.y - a.y);
  }

  // ── 13 features for deadlift / exercise classifier
  extractDeadliftFeatures(lm) {
    const a = (i, j, k) => this._angle(lm[i], lm[j], lm[k]);
    const d = (i, j)    => this._dist(lm[i], lm[j]);

    const leftElbow  = a(11, 13, 15);
    const rightElbow = a(12, 14, 16);
    const leftKnee   = a(23, 25, 27);
    const rightKnee  = a(24, 26, 28);

    return [
      leftElbow,                   // 0
      rightElbow,                  // 1
      a(13, 11, 23),               // 2 left shoulder
      a(14, 12, 24),               // 3 right shoulder
      a(11, 23, 25),               // 4 left hip
      a(12, 24, 26),               // 5 right hip
      leftKnee,                    // 6
      rightKnee,                   // 7
      d(11, 12),                   // 8 shoulder width
      d(23, 24),                   // 9 hip width
      d(11, 23),                   // 10 torso length
      Math.abs(leftElbow - rightElbow),  // 11 elbow symmetry
      Math.abs(leftKnee  - rightKnee),   // 12 knee symmetry
    ];
  }

  // ── 12 squat features (matching Kaggle / predictor.py) ─────
  extractSquatFeatures(lm) {
    const a = (i, j, k) => this._angle(lm[i], lm[j], lm[k]);

    // angles — MediaPipe landmark indices
    const leftKneeAngle  = a(23, 25, 27);
    const rightKneeAngle = a(24, 26, 28);
    const leftHipAngle   = a(11, 23, 25);
    const rightHipAngle  = a(12, 24, 26);
    const leftAnkleAngle = a(25, 27, 31);
    const rightAnkleAngle= a(26, 28, 32);

    // mid points
    const midHip = { x: (lm[23].x + lm[24].x) / 2, y: (lm[23].y + lm[24].y) / 2 };
    const midSh  = { x: (lm[11].x + lm[12].x) / 2, y: (lm[11].y + lm[12].y) / 2 };
    const vertAbove = { x: midHip.x, y: midHip.y - 0.5 };
    const spineAngle = this._angle(vertAbove, midHip, midSh);
    const torsoLean  = spineAngle;

    // lateral knee deviation (x offset vs ankle)
    const leftKneeLateral  = lm[25].x - lm[27].x;
    const rightKneeLateral = lm[27].x - lm[26].x;

    // symmetry score = sum of left/right angle diffs
    const symmetryScore = Math.abs(leftKneeAngle - rightKneeAngle)
                        + Math.abs(leftHipAngle  - rightHipAngle);

    // hip depth (y of midHip — bigger = lower)
    const hipDepth = midHip.y;

    return {
      left_knee_angle:    leftKneeAngle,
      right_knee_angle:   rightKneeAngle,
      left_hip_angle:     leftHipAngle,
      right_hip_angle:    rightHipAngle,
      left_ankle_angle:   leftAnkleAngle,
      right_ankle_angle:  rightAnkleAngle,
      spine_angle:        spineAngle,
      torso_lean:         torsoLean,
      left_knee_lateral:  leftKneeLateral,
      right_knee_lateral: rightKneeLateral,
      symmetry_score:     symmetryScore,
      hip_depth:          hipDepth,
    };
  }

  // ── draw skeleton ────────────────────────────────────────────
  drawSkeleton(lm, formCorrect) {
    const W = this.canvas.width, H = this.canvas.height;
    const color = formCorrect === null ? "#6366f1"
                : formCorrect          ? "#10b981"
                                       : "#ef4444";

    const connections = [
      [11,12],[11,13],[13,15],[12,14],[14,16],
      [11,23],[12,24],[23,24],
      [23,25],[25,27],[27,31],
      [24,26],[26,28],[28,32],
    ];

    this.ctx.strokeStyle = color;
    this.ctx.lineWidth   = 3;
    this.ctx.lineCap     = "round";

    for (const [s, e] of connections) {
      if (!lm[s] || !lm[e]) continue;
      this.ctx.beginPath();
      this.ctx.moveTo(lm[s].x * W, lm[s].y * H);
      this.ctx.lineTo(lm[e].x * W, lm[e].y * H);
      this.ctx.stroke();
    }

    this.ctx.fillStyle = color;
    for (const pt of lm) {
      if (!pt) continue;
      this.ctx.beginPath();
      this.ctx.arc(pt.x * W, pt.y * H, 5, 0, Math.PI * 2);
      this.ctx.fill();
    }
  }

  // ── on pose results ──────────────────────────────────────────
  _onResults(results) {
    this.canvas.width  = this.video.videoWidth  || 640;
    this.canvas.height = this.video.videoHeight || 480;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // FPS tracking
    this._frameCount++;
    const now = performance.now();
    if (now - this._fpsTimer > 1000) {
      this.fps = this._frameCount;
      this._frameCount = 0;
      this._fpsTimer   = now;
      document.getElementById("fps-display").textContent = `${this.fps} FPS`;
    }

    if (!results.poseLandmarks) return;
    const lm = results.poseLandmarks;

    // Extract features and send to callback
    if (this.onFeatures) {
      if (this.exercise === "squat") {
        const feat = this.extractSquatFeatures(lm);
        this.onFeatures(feat, "squat");
      } else if (this.exercise === "deadlift") {
        const feat = this.extractDeadliftFeatures(lm);
        this.onFeatures(feat, "deadlift");
      }
    }

    // Draw
    if (this.showSkeleton) {
      const correct = window.lastFormCorrect ?? null;
      this.drawSkeleton(lm, correct);
    }

    // Show squat-specific live angles
    if (this.exercise === "squat") {
      const knee = this._angle(lm[23], lm[25], lm[27]);
      const hip  = this._angle(lm[11], lm[23], lm[25]);
      window.uiManager?.updateLiveAngles({ knee, hip });
    }
  }

  async start() {
    // Request camera
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      this.video.srcObject = stream;
      await new Promise(r => this.video.onloadedmetadata = r);
      this.video.play();
    } catch (e) {
      console.error("Camera error:", e);
      window.uiManager?.setStatus("Camera denied", "error");
      return;
    }

    // Init MediaPipe Pose
    this.pose = new Pose({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`,
    });
    this.pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    this.pose.onResults(r => this._onResults(r));

    this.isRunning = true;
    this._fpsTimer = performance.now();
    this._loop();

    window.uiManager?.setStatus("Camera active", "active");
    document.getElementById("camera-overlay").classList.add("hidden");
  }

  _loop() {
    if (!this.isRunning) return;
    if (this.video.readyState === 4) {
      this.pose.send({ image: this.video });
    }
    setTimeout(() => this._loop(), 66); // ~15 fps
  }

  stop() {
    this.isRunning = false;
    if (this.video.srcObject) {
      this.video.srcObject.getTracks().forEach(t => t.stop());
      this.video.srcObject = null;
    }
    window.uiManager?.setStatus("Camera stopped", "");
    document.getElementById("camera-overlay").classList.remove("hidden");
  }
}

window.poseDetector = new PoseDetector();

// Form Analysis

class FormAnalyzer {
    constructor() {
        this.apiUrl = 'http://localhost:8000'; // Backend API
        this.repCount = 0;
        this.lastPrediction = null;
        this.feedbackTimeout = null;
    }

    async analyze(features) {
        try {
            // For demo: Simple client-side analysis
            // In production: Send to backend API
            const prediction = this.analyzeClientSide(features);

            this.updateUI(prediction);
            this.updateRepCount(prediction);
        } catch (error) {
            console.error('Analysis error:', error);
        }
    }

    analyzeClientSide(features) {
        // Simple heuristic-based analysis for demo
        // Replace with actual API call in production

        // Check hip angle (feature index 4 and 5)
        const leftHipAngle = features[4];
        const rightHipAngle = features[5];
        const avgHipAngle = (leftHipAngle + rightHipAngle) / 2;

        // Check knee angle (feature index 6 and 7)
        const leftKneeAngle = features[6];
        const rightKneeAngle = features[7];
        const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;

        // Simple rules for deadlift
        let formCorrect = true;
        let feedback = 'Good form!';

        if (avgHipAngle < 90) {
            formCorrect = false;
            feedback = 'Keep hips higher';
        } else if (avgHipAngle > 170) {
            formCorrect = false;
            feedback = 'Avoid over-extension';
        }

        if (avgKneeAngle < 140) {
            formCorrect = false;
            feedback = 'Straighten your knees';
        }

        return {
            formCorrect,
            feedback,
            confidence: 0.85
        };
    }

    async callAPI(features) {
        // Call backend API for prediction
        const response = await fetch(`${this.apiUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ features })
        });

        return await response.json();
    }

    updateUI(prediction) {
        // Update form status
        const formStatus = document.getElementById('form-status');
        if (formStatus) {
            formStatus.textContent = prediction.formCorrect ? '✓' : '✗';
            formStatus.style.color = prediction.formCorrect ? '#10b981' : '#ef4444';
        }

        // Show feedback if enabled
        const showFeedback = document.getElementById('show-feedback').checked;

        if (showFeedback && !prediction.formCorrect) {
            this.showFeedback(prediction.feedback, prediction.formCorrect);
        }
    }

    showFeedback(message, isCorrect) {
        const overlay = document.getElementById('feedback-overlay');
        const icon = document.getElementById('feedback-icon');
        const text = document.getElementById('feedback-text');

        if (overlay && icon && text) {
            icon.textContent = isCorrect ? '✅' : '⚠️';
            text.textContent = message;

            overlay.classList.remove('hidden');

            // Auto-hide after 2 seconds
            clearTimeout(this.feedbackTimeout);
            this.feedbackTimeout = setTimeout(() => {
                overlay.classList.add('hidden');
            }, 2000);
        }
    }

    updateRepCount(prediction) {
        // Simple rep counting logic
        // Count when transitioning from incorrect to correct
        if (this.lastPrediction === false && prediction.formCorrect === true) {
            this.repCount++;
            const repCountEl = document.getElementById('rep-count');
            if (repCountEl) {
                repCountEl.textContent = this.repCount;
            }
        }

        this.lastPrediction = prediction.formCorrect;
    }

    reset() {
        this.repCount = 0;
        this.lastPrediction = null;

        const repCountEl = document.getElementById('rep-count');
        if (repCountEl) {
            repCountEl.textContent = '0';
        }
    }
}

// Initialize
window.formAnalyzer = new FormAnalyzer();

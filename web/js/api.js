// FitVision — API client
const API_BASE = "http://localhost:8000";

const api = {
    async health() {
        const r = await fetch(`${API_BASE}/health`);
        return r.json();
    },

    async predictDeadlift(features) {
        const r = await fetch(`${API_BASE}/predict/deadlift`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features }),
        });
        return r.json();
    },

    async predictSquat(squat_features) {
        const r = await fetch(`${API_BASE}/predict/squat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(squat_features),
        });
        return r.json();
    },

    async predictBenchpress(features) {
        const r = await fetch(`${API_BASE}/predict/benchpress`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features }),
        });
        return r.json();
    },

    async predictExercise(features) {
        const r = await fetch(`${API_BASE}/predict/exercise`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features }),
        });
        return r.json();
    },
};

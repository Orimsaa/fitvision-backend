// UI Helper Functions

class UIManager {
    constructor() {
        this.init();
    }

    init() {
        // Add touch feedback to buttons
        this.addTouchFeedback();

        // Setup menu button
        this.setupMenu();
    }

    addTouchFeedback() {
        const buttons = document.querySelectorAll('button, .exercise-card');

        buttons.forEach(button => {
            button.addEventListener('touchstart', () => {
                button.style.opacity = '0.7';
            });

            button.addEventListener('touchend', () => {
                button.style.opacity = '1';
            });
        });
    }

    setupMenu() {
        const menuBtn = document.getElementById('menu-btn');
        const navMenu = document.getElementById('nav-menu');

        if (menuBtn) {
            menuBtn.addEventListener('click', () => {
                // Toggle menu visibility on mobile
                navMenu.classList.toggle('show');
            });
        }
    }

    showToast(message, type = 'info') {
        // Simple toast notification
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            background: ${type === 'error' ? '#ef4444' : '#6366f1'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            z-index: 1000;
            animation: slideUp 0.3s ease;
        `;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideDown 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// Initialize
window.uiManager = new UIManager();

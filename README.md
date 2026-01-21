# FitVision - Exercise Pose Analysis & Injury Risk Assessment System

ระบบวิเคราะห์ท่าทางการออกกำลังกายแบบฝึกด้วยน้ำหนักและประเมินความเสี่ยงการบาดเจ็บจากวิดีโอด้วย คอมพิวเตอร์วิทัศน์ และ การเรียนรู้ของเครื่อง

## Features

- 🎯 Exercise Classification (Bench Press, Squat, Deadlift)
- 📊 Real-time Pose Analysis
- ⚠️ Injury Risk Assessment
- 💬 Multi-modal Feedback (Visual, Text, Audio)
- 📈 Progress Tracking
- 🎮 Gamification

## Tech Stack

- **Computer Vision:** YOLOv5, MediaPipe, OpenCV
- **Machine Learning:** Random Forest, LSTM, Transformer
- **Deep Learning:** PyTorch
- **Web Framework:** Streamlit, Flask
- **Language:** Python 3.8+

## Installation

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run Streamlit app
streamlit run app/streamlit_app.py

# Or run Flask app
python app/main.py
```

## Project Structure

```
FitVision/
├── config/          # Configuration files
├── data/            # Data and models
├── src/             # Source code
├── app/             # Web applications
├── notebooks/       # Jupyter notebooks
├── tests/           # Unit tests
├── tools/           # Training and utility scripts
└── docs/            # Documentation
```

## Development

See [getting_started.md](docs/getting_started.md) for detailed setup instructions.

## License

MIT License

## Authors

- Your Name
- Project for: [Your Institution]

# Changelog

## [0.2.0] - 2026-01-16

### Changed
- ⬆️ Upgraded from YOLOv5 to YOLOv8 for better performance
  - Updated `ultralytics` package to v8.1.0
  - Changed model from `yolov5s.pt` to `yolov8s.pt`
  - Updated all test scripts to use YOLOv8
  - Updated configuration settings

### Improvements
- ✅ Better accuracy (mAP 53.9% vs 50.7%)
- ✅ Faster inference speed
- ✅ Smaller model size (6.2 MB vs 7.2 MB)

---

## [0.1.0] - 2026-01-16

### Added
- ✅ Initial project structure
- ✅ Configuration files
- ✅ Test scripts (MediaPipe, YOLO, Pipeline)
- ✅ Data collection tools
  - Video recorder
  - Feature extractor
  - Data labeler
  - Model trainer
- ✅ Documentation
  - Implementation plan
  - Enhancement plan
  - Getting started guide
  - Data collection guide
  - Project overview

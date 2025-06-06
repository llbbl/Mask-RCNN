# Mask-RCNN Development Notes

## Project Overview
This is a Mask R-CNN implementation for object detection and instance segmentation.

## Testing
- Run all tests: `poetry run pytest`
- Run unit tests: `poetry run pytest -m unit`
- Run integration tests: `poetry run pytest -m integration`
- Coverage reports are generated automatically
- Test markers: `unit`, `integration`, `slow`

## Development
- Install dependencies: `poetry install`
- Build extensions: Check nms/ and roialign/ directories for CUDA/C extensions

## Notes
- Requires PyTorch and CUDA for full functionality
- Custom NMS and ROI Align implementations included
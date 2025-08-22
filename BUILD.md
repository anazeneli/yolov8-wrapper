# Build Instructions

## Test Locally

```bash
make clean-pyinstaller && make pyinstaller
```

Set path in Viam local module to your directory path:
```
f"{YOLO_DIRECTORY}/build/pyinstaller_dist/main"
```

Replace `{YOLO_DIRECTORY}` with your actual project directory path.

## Build Module

Once you've tested your module locally and it's working, build and package it:

```bash
make module.tar.gz
```

This creates the packaged module. Then upload to the Viam registry:

```bash
viam module upload --version x.x.x --platform linux/arm64 --tags 'jetpack:6' module.tar.gz
```

## Makefile Options

```bash
make pyinstaller          # Build single-file executable using main.spec
make clean-pyinstaller    # Clean PyInstaller build artifacts
make clean-module         # Clean module packaging artifacts
make module.tar.gz        # Package module for distribution
```

## Build Configuration

The build uses `main.spec` for PyInstaller configuration:
- **Single file executable**: All dependencies bundled into one file
- **ARM64 target**: Built for Jetson devices with JetPack 6
- **Size**: ~1.8GB with PyTorch, YOLOv8, and all dependencies

## Make Clean Explanations

**DISCLAIMER: A COMPLETE CLEAN WILL TAKE 30 MINUTES**

```bash
make clean                # Full clean (rebuilds PyTorch/TorchVision from source)
make clean-pyinstaller    # Clean only PyInstaller artifacts (fast)
make clean-module         # Clean only packaging artifacts (fast)
```
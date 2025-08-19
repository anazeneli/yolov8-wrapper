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
make pyinstaller
make clean-pyinstaller
make module.tar.gz
```

# For LOCAL testing, replace 
main.spec -> src/main.py


## Make Clean Explanations

**DISCLAIMER: A COMPLETE CLEAN WILL TAKE 30 MINUTES**
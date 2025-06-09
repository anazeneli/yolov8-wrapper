.PHONY: setup clean pyinstaller clean-pyinstaller pytorch-wheel torchvision-wheel

MODULE_DIR=$(shell pwd)
BUILD=$(MODULE_DIR)/build

VENV_DIR=$(BUILD)/.venv
PYTHON=$(VENV_DIR)/bin/python

PYTORCH_WHEEL=torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
PYTORCH_WHEEL_URL=https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/$(PYTORCH_WHEEL)

TORCHVISION_REPO=https://github.com/pytorch/vision 
TORCHVISION_WHEEL=torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
TORCHVISION_VERSION=0.20.0

REQUIREMENTS=requirements.txt

PYINSTALLER_WORKPATH=$(BUILD)/pyinstaller_build
PYINSTALLER_DISTPATH=$(BUILD)/pyinstaller_dist
	
$(VENV_DIR):
	@echo "Building python venv"
	sudo apt install python3.10-venv
	sudo apt install python3-pip
	python3 -m venv $(VENV_DIR)
	

$(BUILD)/$(PYTORCH_WHEEL):
	@echo "Making $(BUILD)/$(PYTORCH_WHEEL)"
	wget  -P $(BUILD) $(PYTORCH_WHEEL_URL)

pytorch-wheel: $(BUILD)/$(PYTORCH_WHEEL)

$(BUILD)/$(TORCHVISION_WHEEL): $(VENV_DIR) $(BUILD)/$(PYTORCH_WHEEL)
	@echo "Installing dependencies for TorchVision"
	bin/first_run.sh
	bin/install_cusparselt.sh

	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install wheel
	$(PYTHON) -m pip install 'numpy<2' $(BUILD)/$(PYTORCH_WHEEL)

	@echo "Cloning Torchvision"
	git clone --branch v${TORCHVISION_VERSION} --recursive --depth=1 $(TORCHVISION_REPO) $(BUILD)/torchvision

	@echo "Building torchvision wheel"
	cd $(BUILD)/torchvision && $(PYTHON) setup.py --verbose bdist_wheel --dist-dir ../

torchvision-wheel: $(BUILD)/$(TORCHVISION_WHEEL)

$(PYINSTALLER_DISTPATH)/main: $(BUILD)/$(TORCHVISION_WHEEL) 
	@echo " Building pyinstaller executable"
	$(PYTHON) -m pip install -r $(REQUIREMENTS)
	$(PYTHON) -m pip install 'numpy<2' $(BUILD)/$(PYTORCH_WHEEL)
	$(PYTHON) -m pip install $(BUILD)/$(TORCHVISION_WHEEL)
	$(PYTHON) -m PyInstaller --workpath "$(PYINSTALLER_WORKPATH)" --distpath "$(PYINSTALLER_DISTPATH)" src/main.py



pyinstaller: $(PYINSTALLER_DISTPATH)/main

clean-pyinstaller:
	rm -rf $(PYINSTALLER_DISTPATH) $(PYINSTALLER_WORKPATH)

setup: torchvision-wheel 

clean:
	rm -rf $(BUILD) cuda-keyring_1.1-1_all.deb


#!/bin/bash

echo "Installing the required libraries for the 're-id-object-tracking' module..."

# Function to check if libcudnn.so.9 is present
check_cudnn() {
    echo "Checking if libcudnn.so.9 is installed..."
    if ldconfig -p | grep -q 'libcudnn.so.9'; then
        echo "libcudnn.so.9 is already installed."
        return 0
    else
        echo "libcudnn.so.9 is not installed."
        return 1
    fi
}

# Function to install cuDNN
install_cudnn() {
    echo "Starting the installation of cuDNN..."

    # Check if CUDA keyring is already installed
    echo "Checking if cuda-keyring is installed..."
    if dpkg -l | grep -q cuda-keyring; then
        echo "cuda-keyring is already installed. Skipping download and installation."
    else
        # Download the CUDA keyring package
        echo "Downloading CUDA keyring package..."
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb

        # Check if the download was successful
        if [ $? -ne 0 ]; then
            echo "Error: Download failed. Please check your internet connection and the URL."
            exit 1
        else
            echo "CUDA keyring package downloaded successfully."
        fi

        # Install the CUDA keyring package
        echo "Installing the CUDA keyring package..."
        sudo dpkg -i cuda-keyring_1.1-1_all.deb

        # Check if the keyring installation was successful
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install the CUDA keyring package."
            exit 1
        else
            echo "CUDA keyring package installed successfully."
        fi
    fi

    # Download the cuDNN package
    echo "Downloading cuDNN package..."
    wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb

    # Check if the download was successful
    if [ $? -ne 0 ]; then
        echo "Error: Download failed. Please check your internet connection and the URL."
        exit 1
    else
        echo "Download completed successfully."
    fi

    # Install the downloaded package
    echo "Installing the downloaded cuDNN package..."
    sudo dpkg -i cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb

    # Check if the package installation was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install the .deb package."
        exit 1
    else
        echo "cuDNN package installed successfully."
    fi

    # Copy the keyring to the appropriate location
    echo "Copying the keyring..."
    sudo cp /var/cudnn-local-tegra-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/

    # Check if the keyring copy was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to copy the keyring."
        exit 1
    else
        echo "Keyring copied successfully."
    fi

    # Update package lists
    echo "Updating package lists..."
    sudo apt-get update

    # Install cuDNN for CUDA 12
    echo "Installing cuDNN for CUDA 12..."
    sudo apt-get -y install cudnn-cuda-12

    # Check if the installation was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install cudnn-cuda-12."
        exit 1
    else
        echo "cuDNN installation completed successfully."
    fi
}

# Main script execution
if ! check_cudnn; then
    install_cudnn
else
    echo "No installation needed."
fi

echo "All required libraries for the 're-id-object-tracking' module have been installed successfully."

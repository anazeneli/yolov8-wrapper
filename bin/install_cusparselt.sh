#!/bin/bash

echo "Installing cuSPARSELt..."

# Function to check if libcusparselt packages are installed
check_cusparse() {
    echo "Checking if libcusparselt0 and libcusparselt-dev are installed..."
    if dpkg -l | grep -q 'libcusparselt0'; then
        if dpkg -l | grep -q 'libcusparselt-dev'; then
            echo "libcusparselt0 and libcusparselt-dev are already installed."
            return 0
        else
            echo "libcusparselt-dev is missing."
            return 1
        fi
    else
        echo "libcusparselt0 is missing."
        return 1
    fi
}

# Function to install cuSPARSE
install_cusparse() {
    echo "Starting the installation of cuSPARSE..."

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
            echo "Error: Download failed. Please check your internet connection."
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

    # Update package lists
    echo "Updating package lists..."
    sudo apt-get update

    # Install cuSPARSE libraries
    echo "Installing cuSPARSE libraries..."
    sudo apt-get -y install libcusparselt0 libcusparselt-dev

    # Check if the installation was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install cuSPARSE libraries."
        exit 1
    else
        echo "cuSPARSE libraries installed successfully."
    fi
}

# Main script execution
if ! check_cusparse; then
    install_cusparse
else
    echo "No installation needed."
fi

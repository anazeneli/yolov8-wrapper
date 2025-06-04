#!/bin/sh
cd `dirname $0`

# Create a virtual environment to run our code
VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"
ENV_ERROR="This module requires Python >=3.8, pip, and virtualenv to be installed."

# Function to check if Python 3.13 is available
check_python313() {
    command -v python3.13 >/dev/null 2>&1
}

# Function to install Python 3.13 on Debian/Ubuntu
install_python313_debian() {
    echo "Python 3.13 not found. Attempting to install..."
    
    SUDO="sudo"
    if ! command -v $SUDO >/dev/null; then
        echo "This script requires sudo access to install Python 3.13."
        return 1
    fi
    
    # Update package list
    echo "Updating package list..."
    if ! $SUDO apt -qq update >/dev/null 2>&1; then
        echo "Failed to update package list."
        return 1
    fi
    
    # Install software-properties-common for add-apt-repository
    if ! $SUDO apt install -qqy software-properties-common >/dev/null 2>&1; then
        echo "Failed to install software-properties-common."
        return 1
    fi
    
    # Add deadsnakes PPA for newer Python versions
    echo "Adding deadsnakes PPA..."
    if ! $SUDO add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1; then
        echo "Failed to add deadsnakes PPA. Python 3.13 may not be available yet."
        return 1
    fi
    
    # Update again after adding PPA
    if ! $SUDO apt -qq update >/dev/null 2>&1; then
        echo "Failed to update after adding PPA."
        return 1
    fi
    
    # Check if Python 3.13 is available before trying to install
    if ! apt-cache show python3.13 >/dev/null 2>&1; then
        echo "Python 3.13 not available in repositories."
        return 1
    fi
    
    # Install Python 3.13 and required packages
    echo "Installing Python 3.13..."
    if ! $SUDO apt install -qqy python3.13 python3.13-venv python3.13-dev >/dev/null 2>&1; then
        echo "Failed to install Python 3.13 packages."
        return 1
    fi
    
    return 0
}

# Function to install Python 3.13 on CentOS/RHEL/Fedora
install_python313_redhat() {
    echo "Python 3.13 not found. Attempting to install..."
    
    SUDO="sudo"
    if ! command -v $SUDO >/dev/null; then
        SUDO=""
    fi
    
    if command -v dnf >/dev/null; then
        # Fedora/newer RHEL
        echo "Installing Python 3.13 via dnf..."
        $SUDO dnf install -y python3.13 python3.13-devel >/dev/null 2>&1
    elif command -v yum >/dev/null; then
        # Older RHEL/CentOS
        echo "Installing EPEL and Python 3.13 via yum..."
        $SUDO yum install -y epel-release >/dev/null 2>&1
        $SUDO yum install -y python313 python313-devel >/dev/null 2>&1
    fi
}

# Function to compile Python 3.13 from source (fallback)
compile_python313() {
    echo "Attempting to compile Python 3.13 from source..."
    
    # Check if we have required build tools
    if ! command -v gcc >/dev/null || ! command -v make >/dev/null; then
        echo "Build tools (gcc, make) not found. Cannot compile Python from source."
        return 1
    fi
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Download Python 3.13 source
    echo "Downloading Python 3.13 source..."
    if command -v wget >/dev/null; then
        wget -q https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz
    elif command -v curl >/dev/null; then
        curl -s -O https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz
    else
        echo "wget or curl required to download Python source."
        return 1
    fi
    
    # Extract and compile
    echo "Extracting and compiling Python 3.13..."
    tar -xzf Python-3.13.0.tgz
    cd Python-3.13.0
    
    ./configure --prefix=/usr/local --enable-optimizations >/dev/null 2>&1
    make -j$(nproc) >/dev/null 2>&1
    
    SUDO="sudo"
    if ! command -v $SUDO >/dev/null; then
        SUDO=""
    fi
    
    $SUDO make altinstall >/dev/null 2>&1
    
    # Clean up
    cd /
    rm -rf "$TEMP_DIR"
    
    # Check if installation was successful
    if command -v python3.13 >/dev/null; then
        echo "Python 3.13 successfully compiled and installed."
        return 0
    else
        echo "Failed to compile Python 3.13."
        return 1
    fi
}

# Check if Python 3.13 is available, if not, try to install it
if ! check_python313; then
    echo "Python 3.13 not found on system."
    
    if command -v apt-get >/dev/null; then
        # Debian/Ubuntu
        if install_python313_debian; then
            echo "Python 3.13 installation attempted."
        else
            echo "Failed to install Python 3.13 automatically."
            echo "Please install Python 3.13 manually:"
            echo "  sudo add-apt-repository ppa:deadsnakes/ppa"
            echo "  sudo apt update"
            echo "  sudo apt install python3.13 python3.13-venv"
            echo $ENV_ERROR >&2
            exit 1
        fi
    elif command -v yum >/dev/null || command -v dnf >/dev/null; then
        # RedHat/CentOS/Fedora - often Python 3.13 isn't readily available
        echo "Automatic Python 3.13 installation not supported on this system."
        echo "Please install Python 3.13 manually or use pyenv:"
        echo "  curl https://pyenv.run | bash"
        echo "  pyenv install 3.13.0"
        echo $ENV_ERROR >&2
        exit 1
    else
        echo "Unsupported system for automatic Python 3.13 installation."
        echo "Please install Python 3.13 manually."
        echo $ENV_ERROR >&2
        exit 1
    fi
    
    # Check again if Python 3.13 is now available
    if ! check_python313; then
        echo "Python 3.13 still not available after installation attempt."
        echo "Please install Python 3.13 manually and run this script again."
        echo $ENV_ERROR >&2
        exit 1
    else
        echo "Python 3.13 is now available."
    fi
else
    echo "Python 3.13 found."
fi

# Create virtual environment
if ! python3.13 -m venv $VENV_NAME >/dev/null 2>&1; then
    echo "Failed to create virtualenv."
    if command -v apt-get >/dev/null; then
        echo "Detected Debian/Ubuntu, attempting to install python3-venv automatically."
        SUDO="sudo"
        if ! command -v $SUDO >/dev/null; then
            SUDO=""
        fi
        if ! apt info python3.13-venv >/dev/null 2>&1; then
            echo "Package info not found, trying apt update"
            $SUDO apt -qq update >/dev/null
        fi
        $SUDO apt install -qqy python3.13-venv >/dev/null 2>&1
        if ! python3.13 -m venv $VENV_NAME >/dev/null 2>&1; then
            echo $ENV_ERROR >&2
            exit 1
        fi
    else
        echo $ENV_ERROR >&2
        exit 1
    fi
fi

# remove -U if viam-sdk should not be upgraded whenever possible
# -qq suppresses extraneous output from pip
echo "Virtualenv found/created. Installing/upgrading Python packages..."
if ! [ -f .installed ]; then
    if ! $PYTHON -m pip install -r requirements.txt -Uqq; then
        exit 1
    else
        touch .installed
    fi
fi

#!/bin/bash

trap 'trap - ERR; return' ERR

############################ Helper Functions Below ############################

detect_profile() {
    shell_name="${SHELL##*/}"
    profile_path=""
    case "$shell_name" in
        zsh)
            profile_path="$HOME/.zshrc"
            ;;
        bash)
            if [ "$(uname)" = "Darwin" ]; then
                profile_path="$HOME/.bash_profile"
            else
                profile_path="$HOME/.bashrc"
            fi
            ;;
        fish)
            profile_path="$HOME/.config/fish/config.fish"
            ;;
        ash|sh|dash)
            profile_path="$HOME/.profile"
            ;;
        csh|tcsh)
            profile_path="$HOME/.cshrc"
            ;;
        *)
            profile_path="$HOME/.profile"
            ;;
    esac
    echo "$profile_path"
}

get_os_name() {
    case `uname -s` in
        Darwin*)
            echo "MacOS"
            ;;
        Linux*)
            echo "Linux"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            echo "Windows"
            ;;
        *)
            echo "Unknown ($os_name)"
            ;;
    esac
}
############################ Helper Functions Above ############################



############################ Setup Begins Here ############################

SHELL_PROFILE=$(detect_profile)
OS_NAME=$(get_os_name)
echo -e "\033[1;32mDetected shell profile: $SHELL_PROFILE\033[0m"
echo -e "\033[1;32mDetected operating system: $OS_NAME\033[0m"

if [[ "$OS_NAME" == *"Windows"* ]] || [[ "$OS_NAME" == *"Unknown"* ]]; then
    echo -e "\033[1;31mError: This setup script does not support Windows or unknown operating systems, got $OS_NAME. Please refer to README for other manual setup instructions.\033[0m"
    exit 1
fi

# Install necessary system packages
# Including wget git gcc make vim
if [[ "$OS_NAME" == "Linux" ]]; then
    if ! command -v apt &> /dev/null; then
        echo -e "\033[1;31mError: apt not available. This script only supports Debian-based systems.\033[0m"
        exit 1
    fi
    sudo apt update
    sudo apt-get install -y  wget git gcc g++ make vim

    # Install required fonts
    echo -e "\033[1;32mInstall required fonts...\033[0m"
    sudo apt update
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | sudo debconf-set-selections
    sudo apt-get install -y ttf-mscorefonts-installer
    rm ~/.cache/matplotlib -rf
else
    if [[ "$OS_NAME" != "MacOS" ]]; then
        echo -e "\033[1;31mError: Unsupported OS $OS_NAME\033[0m"
        exit 1
    fi
    if ! command -v brew &> /dev/null; then
        echo -e "\033[1;31mHomebrew not found. Please install homebrew first following https://docs.brew.sh/Installation\033[0m"
        exit 1
    fi
    brew update
    brew install wget git gcc make vim
fi


# Install `uv`, a Python package manager.
sleep 1  # Take a short rest.
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"
echo -e "\033[1;32mInstalling uv...\033[0m"
curl -LsSf https://astral.sh/uv/install.sh | sh

if [ -f $HOME/.local/bin/env ]; then
    source $HOME/.local/bin/env
elif [ -f $HOME/.local/bin/uv ]; then
    case ":${PATH}:" in
        *:"$HOME/.local/bin":*)
            ;;
        *)
            # Prepending path in case a system-installed binary needs to be overridden
            export PATH="$HOME/.local/bin:$PATH"
            ;;
    esac
fi
echo 'case ":${PATH}:" in *:"$HOME/.local/bin":*) ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac' >> $SHELL_PROFILE
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"

# Install Rust cargo
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"
echo -e "\033[1;32mInstalling Rust..\033[0m"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env" 
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"

# Change the ownership of the block storage
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"
echo -e "\033[1;32mCreating a working directory at /opt/tiger, which requires root privileges...\033[0m"
sudo mkdir -p /opt/tiger
export WORKDIR=/opt/tiger
echo -e "\033[1;32mUsing working directory: $WORKDIR\033[0m"
echo -e "\033[1;32mChanging ownership of working directory $WORKDIR\033[0m"
sudo chown `id -u`:`id -g` /opt/tiger
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"

# Clone the repository
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"
echo -e "\033[1;32mEntering $WORKDIR and cloning the repository...\033[0m"
cd $WORKDIR && git clone https://github.com/nyu-systems/Entangle.git
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"

# Build egger
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"
echo -e "\033[1;32mBuilding egger (our Rust-side egraph saturator)...\033[0m"
cd $WORKDIR/Entangle/egger && cargo build --release
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"

# Setup Entangle Python environment
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"
echo -e "\033[1;32mSetting up Python environment...\033[0m"
cd $WORKDIR/Entangle && uv sync --link-mode copy
echo "source $WORKDIR/Entangle/.venv/bin/activate" >> $SHELL_PROFILE

# Activate the environment
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"
echo -e "\033[1;32mActivating Python environment...\033[0m"
source .venv/bin/activate

# Install Entangle Python package
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"
echo -e "\033[1;32mInstalling Entangle Python package...\033[0m"
pip install -e . --force-reinstall
echo -e "\033[1;32m---------------------------------------------------------------------\033[0m"

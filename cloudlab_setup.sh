#!/bin/bash

# Install required fonts
echo "Install required fonts..."
sudo apt update
echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | sudo debconf-set-selections
sudo apt-get install -y ttf-mscorefonts-installer
rm ~/.cache/matplotlib -rf

# Install `uv`, a Python package manager.
echo "---------------------------------------------------------------------"
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 
echo "---------------------------------------------------------------------"

# Install Rust cargo
echo "---------------------------------------------------------------------"
echo "Installing Rust.."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env" 
echo "---------------------------------------------------------------------"

# Change the ownership of the block storage
echo "---------------------------------------------------------------------"
export WORKDIR=/opt/tiger
echo "Using working directory: $WORKDIR"
echo "Changing ownership of working directory $WORKDIR"
sudo chown -R `id -u`:`id -g` /opt/tiger
echo "---------------------------------------------------------------------"

# Clone the repository
echo "---------------------------------------------------------------------"
echo "Entering $WORKDIR and cloning the repository..."
cd $WORKDIR && git clone https://github.com/nyu-systems/Entangle.git
echo "---------------------------------------------------------------------"

# Build egger
echo "---------------------------------------------------------------------"
echo "Building egger (our Rust-side egraph saturator)..."
cd $WORKDIR/Entangle/egger && cargo build --release
echo "---------------------------------------------------------------------"

# Setup
echo "---------------------------------------------------------------------"
echo "Setting up Python environment..."
cd $WORKDIR/Entangle && uv sync
echo "source $WORKDIR/Entangle/.venv/bin/activate" >> $HOME/.bashrc
echo "---------------------------------------------------------------------"
echo "Activating Python environment..."
source .venv/bin/activate
echo "---------------------------------------------------------------------"
echo "Installing Entangle Python package..."
pip install -e .
echo "---------------------------------------------------------------------"

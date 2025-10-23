#!/bin/bash
# 从源码编译安装 FFmpeg 和 Decord

set -e

echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    nasm \
    git \
    yasm \
    pkg-config

echo "Building FFmpeg from source..."
cd /tmp
if [ -d "ffmpeg" ]; then
    rm -rf ffmpeg
fi

git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
git checkout n4.4.2
./configure --enable-shared --enable-pic --prefix=$HOME/anaconda3/envs/gr00t
make -j$(nproc)
make install

echo "Building Decord from source..."
cd /tmp
if [ -d "decord" ]; then
    rm -rf decord
fi

git clone --recursive https://github.com/dmlc/decord
cd decord
mkdir -p build && cd build

# Set PKG_CONFIG_PATH to find the correct FFmpeg
export PKG_CONFIG_PATH=$HOME/anaconda3/envs/gr00t/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/gr00t/lib:$LD_LIBRARY_PATH

cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/anaconda3/envs/gr00t \
    -DFFMPEG_INCLUDE_DIR=$HOME/anaconda3/envs/gr00t/include \
    -DFFMPEG_LIBRARIES="$HOME/anaconda3/envs/gr00t/lib/libavformat.so;$HOME/anaconda3/envs/gr00t/lib/libavfilter.so;$HOME/anaconda3/envs/gr00t/lib/libavcodec.so;$HOME/anaconda3/envs/gr00t/lib/libavutil.so;$HOME/anaconda3/envs/gr00t/lib/libswresample.so;$HOME/anaconda3/envs/gr00t/lib/libavdevice.so"

make -j$(nproc)

# Install Python package
cd ../python
source $HOME/anaconda3/bin/activate gr00t
python setup.py install

echo "Cleaning up..."
cd ~
rm -rf /tmp/ffmpeg /tmp/decord

echo "Done! FFmpeg and Decord have been installed to the gr00t conda environment."
echo "Testing decord..."
python -c 'import decord; print("Decord version:", decord.__version__); print("Decord works!")'

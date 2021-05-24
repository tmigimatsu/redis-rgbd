# redis-rgbd
Stream/calibrate RGBD cameras over Redis

## Installation

### MacOS

```sh
brew install redis

# Kinect2 dependencies.
brew install glfw3 libusb

# Compile.
mkdir build
cd build
cmake ..
make -j
```

### Linux

```sh
sudo apt install redis

# Kinect2 dependencies.
sudo apt install libglfw3-dev libturbojpeg0-dev libusb-1.0-0-dev

# Compile.
mkdir build
cd build
cmake ..
make -j
```

# Edge Detection Pipeline

This project implements an edge detection pipeline using CUDA for efficient image processing. The pipeline is designed to demonstrate the use of CUDA kernels for performing edge detection operations on images.

## Project Structure

```
EdgeDetectionPipeline
├── src
│   └── main.cpp        # Entry point of the application
├── include
│   └── kernel.cuh      # CUDA kernel declarations and functions
├── CMakeLists.txt      # CMake configuration file
└── README.md           # Project documentation
```

## Requirements

- CMake
- CUDA Toolkit
- A compatible C++ compiler

## Building the Project

1. Clone the repository or download the project files.
2. Open a terminal and navigate to the project directory.
3. Create a build directory:
   ```
   mkdir build
   cd build
   ```
4. Run CMake to configure the project:
   ```
   cmake ..
   ```
5. Build the project:
   ```
   make
   ```

## Running the Application

After building the project, you can run the application using the following command:

```
./EdgeDetectionPipeline
```

Make sure to provide the necessary input images as required by the application.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
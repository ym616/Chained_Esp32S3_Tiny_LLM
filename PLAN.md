# Deployment Plan for Chained ESP32-S3 90M Parameter LLM

This document outlines everything you need to do to get your 3-node distributed LLM cluster running on ESP32-S3 boards.
By utilizing 8-bit quantization and direct flash memory mapping, we fit an incredible 90M parameter model across the cluster (approx. 30M parameters per board), overcoming the 8MB PSRAM limit!

## Hardware Setup
1. **Boards**: You need 3x ESP32-S3 DevKitC boards with at least 8MB PSRAM and **32MB Flash** (e.g., N32R8).
2. **Wiring (UART Ring)**:
   - **Node 1 TX (GPIO 17)** -> **Node 2 RX (GPIO 18)**
   - **Node 2 TX (GPIO 17)** -> **Node 3 RX (GPIO 18)**
   - **Node 3 TX (GPIO 17)** -> **Node 1 RX (GPIO 18)**
3. **Common Ground**: Ensure the `GND` pins of all three boards are connected together.

## Software Prerequisites
1. **Python 3.x**: Ensure Python is installed with `numpy`.
2. **ESP-IDF v5.0+**: Ensure you have the ESP-IDF toolchain installed and exported in your terminal.
3. **PyTorch**: Required for training and exporting the model (`pip install torch numpy`).

## Step 1: Model Training (To be done later)
1. Run `python train_tiny_llm.py`. This will clone the `llama2.c` repository, download the TinyStories dataset, and train a 90M parameter model (12 layers, 768 dim).
2. Once training finishes, it will export the float32 model weights to `my_fine_tuned_model.bin` in the root of this project.

## Step 2: Model Splitting & Quantization
A 90M parameter float32 model is ~360MB, which is far too large for even the 32MB Flash on the ESP32-S3. 
To fix this, we quantize the weights to 8-bit integers (`q8_0`) and split the model evenly across the 3 nodes.
1. Run the deployment script (which handles splitting automatically), or run it manually:
   ```bash
   python split_model.py my_fine_tuned_model.bin
   ```
2. This generates `node1.bin`, `node2.bin`, and `node3.bin`. Each contains the transformer layers assigned to that node, quantized to 8-bit, reducing the size to <30MB per node.

## Step 3: Cluster Deployment
We use a custom `partitions.csv` that defines a 30MB raw `model` partition on the flash to store the weights.
1. Connect all 3 ESP32-S3 boards to your computer via USB.
2. Identify the COM ports for each board.
3. Run the deployment script:
   ```bash
   python deploy_cluster.py
   ```
4. The script will sequentially:
   - Change `NODE_ID` in `main/main.cpp`.
   - Compile the firmware and flash it.
   - Use `parttool.py` to write `nodeX.bin` directly into the raw `model` partition on the flash.

## Step 4: Execution & Monitoring
1. Once all boards are flashed, press the `EN` (Reset) button on all three boards.
2. The nodes will boot and use `esp_partition_mmap` to directly memory-map the 30MB of flash into the CPU's address space. This avoids the slow SPIFFS filesystem and bypasses the 8MB PSRAM limit!
3. Node 1 processes its layers using fast integer multiplication (`int8`), streams the activation to Node 2, which streams to Node 3. Node 3 samples a token and sends the Token ID back to Node 1.
4. To monitor the output:
   ```bash
   idf.py -p COM3 monitor
   ```

## Major Enhancements Made
- **`split_model.py`**: Now implements a custom Numpy-based row-wise `int8` quantization algorithm. It converts the float32 `llama2.c` binary into a highly compressed, split format.
- **`main.cpp`**: Switched from `fread` via SPIFFS to `esp_partition_mmap`. Replaced floating point matrix multiplications with accelerated integer `int8_t` dot products.
- **`partitions.csv`**: Created a massive 30MB raw data partition for mapping.
- **`train_tiny_llm.py`**: Reconfigured to target a 90M parameter architecture (Dim 768, 12 layers).

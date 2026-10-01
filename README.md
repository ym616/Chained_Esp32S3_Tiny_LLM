# Chained Esp32S3 tiny LLM

This project contains the firmware and structure for a 3-node distributed LLM inference cluster using ESP32-S3 microcontrollers, heavily inspired by pipeline parallelism.

## Hardware Requirements
- 3x ESP32-S3-DevKitC-1 (N16R8: 16MB Flash, 8MB PSRAM)
- High-speed UART wiring (see below)

## Architecture
- **Node 1 (Head)**: Handles BLE/Serial input, Tokenization, Initial Layers, and coordinates the inference loop.
- **Node 2 (Body)**: Handles intermediate transformer layers.
- **Node 3 (Tail)**: Handles final transformer layers, logits calculation, and token sampling.

## UART Ring Configuration
To pass the continuous activation tensors:
- Node 1 TX (GPIO 17) -> Node 2 RX (GPIO 18)
- Node 2 TX (GPIO 17) -> Node 3 RX (GPIO 18)
- Node 3 TX (GPIO 17) -> Node 1 RX (GPIO 18)
*Be sure to link common ground (GND) across all three boards.*

## Flashing Instructions

1. **Setup ESP-IDF** (v5.0+ recommended).
2. For each node, open `main/main.cpp` and adjust the `#define NODE_ID X` macro (1, 2, or 3).
3. Build the project:
   ```bash
   idf.py build
   ```
4. Flash the node:
   ```bash
   idf.py -p COM_PORT flash monitor
   ```

## Model Preparation
1. Run `python split_model.py my_fine_tuned_model.bin` to generate `node1.bin`, `node2.bin`, and `node3.bin`.
2. Use ESP-IDF's `spiffsgen.py` or the `parttool.py` to upload the respective `.bin` file to the SPIFFS partition of each node.

Example to flash a spiffs image:
```bash
esptool.py --chip esp32s3 -p COM_PORT write_flash 0x310000 nodeX.bin
```
*(Ensure the offset matches the `storage` partition in `partitions.csv`)*

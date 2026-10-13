# Chained ESP32-S3 Tiny LLM

A distributed 3-node inference pipeline running an 86.5M parameter LLaMA-style transformer across three ESP32-S3 (N16R8) microcontrollers over a 921,600 baud UART ring. The model is fine-tuned to produce Gordon Ramsay-style roasts in response to compiler and runtime errors sent over serial.

The complete development and chat transcript is recorded in [CHAT_EXPORT.md](CHAT_EXPORT.md).

---

## Technical Overview

Deploying an 86.5M parameter transformer on microcontrollers is constrained by both memory footprint and hardware MMU mapping:
- In float32, 86.5M parameters take ~346 MB. In 4-bit (Q4_0), the weights occupy ~43.3 MB.
- The ESP32-S3 (N16R8) has 16 MB of SPI flash and 8 MB of octal PSRAM.
- The ESP32-S3 hardware MMU data cache can map at most 3 to 4 MB of flash into virtual memory at any time. Calling `esp_partition_mmap` on larger partitions fails with `ESP_ERR_NO_MEM`.

To fit and run the model across three boards:
1. **Pipeline Parallelism**: The 12 transformer layers are split evenly across 3 boards (4 layers per node). Hidden state vectors are passed sequentially through a unidirectional UART ring.
2. **Channel-Wise 4-bit Quantization (Q4_0)**: All weight matrices are quantized to signed 4-bit nibbles with per-row float32 scale factors. The inner matrix multiplication runs integer multiply-accumulate operations, multiplying by the row scale once per output element.
3. **Flash-to-PSRAM Layer Paging**: Instead of relying on `mmap`, each node allocates a 3.57 MB layer buffer in PSRAM and uses `esp_partition_read` to page layer weights from SPI flash on demand (~120 ms per layer read).
4. **PSRAM Key-Value (KV) Cache**: Each node allocates 1.5 MB in PSRAM for multi-head attention key and value caches across its assigned layers. Sequence position (`pos`) is synchronized across boards in the packet header to prevent token drift.
5. **Framed UART Protocol**: Serial transmission uses a 6-byte sync header (`0xA5 0x5A`, packet type, token sequence position, payload length) at 921,600 baud to prevent byte drift and UART FIFO desynchronization.

---

## Architecture

```
                       +----------------------------------+
                       |        USB Serial Terminal       |
                       +----------------+-----------------+
                                        |
                                        v
               +--------------------------------------------------+
               |                Node 1 (Head Node)                |
               | - USB CDC Serial RX/TX                           |
               | - SentencePiece BPE tokenizer/detokenizer (2048) |
               | - Embedding table (2048 x 768)                   |
               | - Transformer layers 0 to 3                      |
               | - PSRAM KV cache (layers 0-3)                    |
               +------------------------+-------------------------+
                                        | TX (GPIO 17) -> RX (GPIO 18)
                                        | 921,600 baud
                                        v
               +--------------------------------------------------+
               |                Node 2 (Body Node)                |
               | - Transformer layers 4 to 7                      |
               | - PSRAM KV cache (layers 4-7)                    |
               | - Rotary position embeddings (RoPE)              |
               +------------------------+-------------------------+
                                        | TX (GPIO 17) -> RX (GPIO 18)
                                        | 921,600 baud
                                        v
               +--------------------------------------------------+
               |                Node 3 (Tail Node)                |
               | - Transformer layers 8 to 11                     |
               | - PSRAM KV cache (layers 8-11)                   |
               | - Final RMSNorm                                  |
               | - Classifier head (768 -> 2048)                  |
               | - Temperature sampling                           |
               +------------------------+-------------------------+
                                        | TX (GPIO 17) -> RX (GPIO 18)
                                        | 921,600 baud (Token ID)
                                        v
                              (Returns to Node 1)
```

### UART Packet Framing
```
[0xA5] [0x5A] [TYPE] [POS] [LEN_HI] [LEN_LO] [PAYLOAD ...]
```
- `0xA5 0x5A`: Synchronization header.
- `TYPE`: `0x01` (ping), `0x02` (layer activations, 3072 bytes FP32), `0x03` (sampled token ID).
- `POS`: Current token sequence position (`0..63`) for KV cache indexing.
- `LEN_HI` / `LEN_LO`: 16-bit payload length.

---

## Model Specifications

| Parameter | Value |
| :--- | :--- |
| Architecture | Decoder-only Transformer (LLaMA / Mistral style) |
| Parameters | 86,512,128 (~86.5M) |
| Hidden Dimension (`dim`) | 768 |
| Feed-Forward Dimension (`hidden_dim`) | 2048 (SwiGLU) |
| Total Layers | 12 (4 layers per node) |
| Attention Heads | 12 query / 12 key-value (`head_dim = 64`) |
| Context Window | 256 tokens max (KV cache sized to 64 tokens on device) |
| Vocabulary Size | 2,048 tokens (SentencePiece BPE) |
| Normalization | RMSNorm (`eps = 1e-5`) |
| Positional Encoding | Rotary Position Embeddings (RoPE) |
| Quantization | Channel-wise 4-bit (Q4_0), signed offset with FP32 row scales |

---

## Hardware and Wiring

### Components
- 3x ESP32-S3-DevKitC-1 (N16R8: 16MB Flash, 8MB Octal PSRAM)
- Connecting jumper wires
- Shared 5V and GND bus
- Single USB-C cable (connected to Node 1)

### Ring Data Wiring
Connect GPIOs in a ring:
- **Node 1 TX (GPIO 17)** -> **Node 2 RX (GPIO 18)**
- **Node 2 TX (GPIO 17)** -> **Node 3 RX (GPIO 18)**
- **Node 3 TX (GPIO 17)** -> **Node 1 RX (GPIO 18)**

### Power Distribution
- Connect the **GND** pins of all three boards together.
- Connect the **5V / VIN** pins of all three boards together on a common rail.
- Plug the USB-C cable into **Node 1**. Node 1 powers the rail and handles interactive serial communications.

---

## Flashing Instructions

### Method 1: Automated Script (upload_cluster.py)

The script handles detection, build artifacts, and flashing modes:

```bash
# On Windows, run the batch script or use the Python launcher directly:
flash.bat
# or:
py upload_cluster.py
```

The script offers two modes:
1. **Mode 1: Fast Firmware Update (default)**: Flashes only the bootloader, partition table, and firmware binary (~3 seconds per board). Use this for any code changes.
2. **Mode 2: Full Flash**: Flashes the firmware plus the 14MB model partition (`node1.bin`, `node2.bin`, `node3.bin`) to flash offset `0x13c000` (~2 minutes per board). Use this only when first provisioning new boards or when updating model weights.

Follow the on-screen prompt to plug in Node 1, Node 2, and Node 3 sequentially.

### Method 2: Manual PlatformIO CLI

To build and flash nodes individually:

```bash
# Node 1
pio run -e node1 -t upload --upload-port COM_PORT

# Node 2
pio run -e node2 -t upload --upload-port COM_PORT

# Node 3
pio run -e node3 -t upload --upload-port COM_PORT
```

To flash the 14MB model partitions manually via `esptool`:

```bash
esptool.py --chip esp32s3 -p COM_PORT --baud 921600 write_flash --flash_mode dio --flash_freq 40m 0x13c000 nodeX.bin
```

---

## Running the Cluster

1. Connect the USB-C cable to **Node 1**.
2. Open a serial monitor at **115,200 baud** with line endings set to **Both NL & CR** or **Newline**.
3. **Power Buffer**: Node 1 counts down 15 seconds to allow Nodes 2 and 3 to boot and stabilize on the 5V bus. Press `ENTER` at any time to skip the wait.
4. Once the ring ping succeeds, the prompt will appear.
5. Enter any compiler error or runtime exception to generate a response.

### Example Console Session

```text
========================================================
     ESP32-S3 NODE 1 INITIALIZING
========================================================
[Node 1] Locating model partition on flash...
[Node 1] Model Config: dim=768, hidden=2048, layers=4, heads=12, vocab=2048
[Node 1] Weights loaded into PSRAM! Free PSRAM: 3758 KB

========================================================
     ESP32-S3 CLUSTER STARTUP BUFFER (15 SECONDS)
========================================================
[Node 1] Giving Node 2 & Node 3 time to power on and boot...
[Node 1] (Press ENTER in Serial Monitor anytime to skip wait)

[Node 1] Checking cluster ring in 15 seconds...
[Node 1] Buffer skipped by user!

[Node 1] Testing ring link (Node 1 -> Node 2 -> Node 3 -> Node 1)...
[Node 1] Pinging ring (Attempt 1/5)... PASSED!

========================================================
  [+] SUCCESS: ALL 3 ESP32-S3 NODES LINKED IN THE RING! 
========================================================
Ring pipeline: Node 1 -> Node 2 -> Node 3 -> Node 1 is ACTIVE.

========================================================
     CHEF GORDON RAMSAY ESP32-S3 CLUSTER READY!         
========================================================
How to use: Type an error into Serial Monitor & hit ENTER.
Examples:   'NullPointerException' or 'SyntaxError' or 'panic'

> NullPointerException
[Chef Gordon Ramsay]: You donkey! A NullPointerException?! You're serving up empty pointers like raw chicken on a dirty plate! Initialize your variables or get out of my kitchen!
```

---

## Training and Weight Export Pipeline

To retrain or modify the model:

```bash
# 1. Generate synthetic error and roast paired dataset (300,000 samples)
py generate_dataset.py

# 2. Train custom 2048-token SentencePiece tokenizer
py gen_vocab.py

# 3. Train the transformer in PyTorch
py train_tiny_llm.py

# 4. Quantize to 4-bit (Q4_0) and split into node partitions
py split_model.py my_fine_tuned_model.bin
```

---

## Repository Structure

```
chained_esp32s3_tiny_llm/
├── flash.bat              # Shortcut for Windows cluster flashing
├── upload_cluster.py      # Automated flasher script (Fast and Full modes)
├── platformio.ini         # PlatformIO build configuration for all 3 nodes
├── partitions.csv         # 16MB partition table layout
├── PROJECT_OVERVIEW.txt   # Detailed technical documentation and bug chronicle
├── CHAT_EXPORT.md         # Full project conversation and development transcript
├── README.md              # Project overview and setup instructions
│
├── main/
│   ├── main.cpp           # Node firmware (layer paging, KV cache, ring pipeline)
│   ├── vocab.h            # Static vocabulary lookup table
│   └── CMakeLists.txt     # ESP-IDF component file
│
├── llama2.c/
│   ├── model.py           # Transformer architecture (RMSNorm, RoPE, SwiGLU)
│   └── train.py           # PyTorch training loop
│
├── generate_dataset.py    # Synthetic error-roast dataset generator
├── gen_vocab.py           # SentencePiece vocabulary generator
├── train_tiny_llm.py      # Training script for 86.5M parameter model
├── split_model.py         # 4-bit quantizer and cluster slicing script
└── test_insults.py        # Host-side validation test script
```

---

## Further Reading

For an in-depth writeup covering low-level math derivations, assembly inner loops, memory layouts, and solutions to hardware/RTOS issues encountered during development (such as task watchdog resets, MMU cache limits, and UART framing desynchronization), refer to [PROJECT_OVERVIEW.txt](file:///c:/Users/Y_M/Documents/chained_esp32s3_tiny_llm/PROJECT_OVERVIEW.txt).

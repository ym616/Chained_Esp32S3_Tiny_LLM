# Chained ESP32-S3 Tiny LLM: Distributed 86.5M Parameter Transformer Cluster
### *"Chef Gordon Ramsay: Autonomous IDE Error Roasting Cluster"*

[![ESP32-S3](https://img.shields.io/badge/Hardware-ESP32--S3%20(N16R8)-red.svg)](https://www.espressif.com/)
[![Parameters](https://img.shields.io/badge/Model%20Size-86.5M%20Params-blue.svg)](file:///c:/Users/Y_M/Documents/chained_esp32s3_tiny_llm/PROJECT_OVERVIEW.txt)
[![Quantization](https://img.shields.io/badge/Quantization-Channel--Wise%204--Bit%20(Q4__0)-green.svg)](file:///c:/Users/Y_M/Documents/chained_esp32s3_tiny_llm/split_model.py)
[![Baud](https://img.shields.io/badge/Ring%20Bus-921%2C600%20Baud%20UART-orange.svg)](file:///c:/Users/Y_M/Documents/chained_esp32s3_tiny_llm/main/main.cpp)
[![Framework](https://img.shields.io/badge/Framework-PlatformIO%20%7C%20Arduino%20%7C%20PyTorch-purple.svg)](file:///c:/Users/Y_M/Documents/chained_esp32s3_tiny_llm/platformio.ini)

A distributed pipeline-parallel Large Language Model inference engine running across a physical ring cluster of **3x ESP32-S3 microcontrollers** with **zero host computer assistance**. The cluster executes an **86.5 Million Parameter LLaMA-style Decoder-Only Transformer**, fine-tuned to act as Chef Gordon Ramsay and aggressively roast compiler errors and code bugs submitted via USB Serial.

---

## 🚀 Novelty & Technical Breakthrough

Running Large Language Models on microcontrollers is conventionally restricted to models under **15M to 30M parameters** (e.g. TinyStories-15M) due to strict hardware boundaries:
- An 86.5M parameter model in standard 32-bit floating point (FP32) consumes **~346 MB**.
- In standard 8-bit integer quantization (INT8), it consumes **~86.5 MB**.
- An ESP32-S3 (N16R8) has only **16 MB Flash** and **8 MB external PSRAM**.
- Crucially, the ESP32-S3 hardware Memory Management Unit (MMU) limits virtual flash address cache mapping (`mmap`) to a **3–4 MB window**. Attempting to map any partition larger than 4MB causes an immediate `ESP_ERR_NO_MEM` hardware panic.

### How This Project Solves It:
1. **Ring Pipeline Spatial Parallelism**: Slices the 12 transformer layers across 3 ESP32-S3 nodes connected in a high-speed 921,600 baud UART ring.
2. **Channel-Wise 4-Bit Packed Quantization (Q4_0)**: Compresses all weight matrices by **87.5%** down to 4 bits per weight with FP32 row scales, reducing the entire model from 346 MB to **~43.3 MB** (~14.4 MB per board).
3. **Dynamic Flash-to-PSRAM Layer Paging**: Bypasses the 3 MB MMU mapping ceiling by dynamically paging 3.57 MB layer chunks from SPI Flash directly into an 8 MB PSRAM layer cache via DMA reads (`esp_partition_read`) in ~120 ms.
4. **Canonical Multi-Head Attention Key-Value (KV) Cache**: Implements a full 1.5 MB KV Cache per board in PSRAM (`key_cache` and `value_cache`) with synchronized sequence positions (`pos`), eliminating context loss and sentence collapse.
5. **Zero-Float Inner Matrix Dot Products**: Uses signed 4-bit integer multiply-accumulate operations in the inner loop, scaling by floating-point scale factors only once per output dimension.

---

## 🏛️ System Architecture

```
                       +-----------------------------------+
                       |        Master USB Serial          |
                       |       (Interactive Terminal)       |
                       +-----------------+-----------------+
                                         |
                                         v
               +-------------------------------------------------------+
               |                  NODE 1: HEAD NODE                    |
               |  - CDC USB Serial RX/TX                               |
               |  - BPE Tokenizer & Detokenizer (vocab=2048)           |
               |  - Token Embeddings (2048 x 768)                      |
               |  - Transformer Layers 0, 1, 2, 3                      |
               |  - Canonical PSRAM KV Cache (Layers 0-3)              |
               +---------------------------+---------------------------+
                                           | UART Ring TX (GPIO 17)
                                           | 921,600 Baud
                                           v UART Ring RX (GPIO 18)
               +-------------------------------------------------------+
               |                  NODE 2: BODY NODE                    |
               |  - Transformer Layers 4, 5, 6, 7                      |
               |  - Canonical PSRAM KV Cache (Layers 4-7)              |
               |  - RoPE Positional Rotation                           |
               +---------------------------+---------------------------+
                                           | UART Ring TX (GPIO 17)
                                           | 921,600 Baud
                                           v UART Ring RX (GPIO 18)
               +-------------------------------------------------------+
               |                  NODE 3: TAIL NODE                    |
               |  - Transformer Layers 8, 9, 10, 11                    |
               |  - Canonical PSRAM KV Cache (Layers 8-11)             |
               |  - Final RMSNorm Layer                                |
               |  - Language Model Head Classifier (WCLS: 768 -> 2048) |
               |  - Temperature Softmax & Token Sampling               |
               +---------------------------+---------------------------+
                                           | UART Ring TX (GPIO 17)
                                           | 921,600 Baud (Token ID)
                                           v UART Ring RX (GPIO 18)
                                 (Returns to Node 1)
```

### Framed UART Ring Protocol (921,600 Baud)
All inter-node traffic uses a strict 6-byte sync frame packet protocol:
```
[ 0xA5 ] [ 0x5A ] [ TYPE ] [ POS ] [ LEN_HI ] [ LEN_LO ] [ PAYLOAD ... ]
```
- `0xA5 0x5A`: Magic synchronization bytes (prevents line noise / partial FIFO corruption).
- `TYPE`: `0x01` (Ring Ping), `0x02` (Hidden Activation Vector, 3072 bytes FP32), `0x03` (Sampled Token ID).
- `POS`: Current token sequence position (`0..63`) ensuring all 3 nodes index their KV caches in lockstep.

---

## 🧠 Machine Learning Specs

| Specification | Value | Description |
| :--- | :--- | :--- |
| **Model Architecture** | Decoder-Only Transformer | LLaMA / Mistral architecture |
| **Total Parameters** | **86,512,128** | ~86.5 Million parameters |
| **Embedding Dim (`dim`)** | **768** | Hidden representation width |
| **FFN Dim (`hidden_dim`)**| **2048** | SwiGLU projection dimension |
| **Transformer Layers** | **12** | 4 layers allocated per ESP32-S3 node |
| **Attention Heads** | **12** Query / **12** KV | Multi-Head Attention (`head_dim = 64`) |
| **Context Window** | **256** tokens | Max context length (Active MCU KV cache: 64) |
| **Vocabulary Size** | **2,048** tokens | Custom SentencePiece BPE vocabulary |
| **Normalization** | RMSNorm ($\epsilon = 10^{-5}$) | Root Mean Square Normalization |
| **Position Embeddings** | Rotary (RoPE) | Precomputed frequency tables in PSRAM |
| **Activation Function** | SwiGLU | $\text{Swish}(xW_1) \cdot (xW_3)$ |
| **Quantization** | Channel-wise 4-bit (Q4_0) | Signed 4-bit offset-binary with FP32 row scales |

---

## 🔌 Hardware Setup & Wiring

### Bill of Materials
- **3x ESP32-S3-WROOM-1 / DevKitC-1** (N16R8: 16MB SPI Flash, 8MB Octal PSRAM).
- Breadboard or terminal block power bus.
- Jumper wires.
- 1x USB-C cable (connected to Node 1).

### 1. High-Speed UART Ring Data Wiring
Connect the hardware serial pins unidirectionally in a ring:
| From Node / Pin | To Node / Pin | Signal / Function |
| :--- | :--- | :--- |
| **Node 1 GPIO 17 (TX)** | **Node 2 GPIO 18 (RX)** | Activations (Layers 0-3 &rarr; 4-7) |
| **Node 2 GPIO 17 (TX)** | **Node 3 GPIO 18 (RX)** | Activations (Layers 4-7 &rarr; 8-11) |
| **Node 3 GPIO 17 (TX)** | **Node 1 GPIO 18 (RX)** | Next Token ID (Sampling &rarr; Head) |

### 2. Common Power Rail
To avoid voltage ground loops and unstable UART levels:
- Connect all **GND** pins across all three boards to a common ground bus.
- Connect all **5V / VIN** pins together across all three boards to a common 5V bus.
- Plug the single USB-C cable into **Node 1**. It powers the entire 3-node cluster and hosts the Serial Monitor.

---

## ⚡ Flashing & Operating Guide

### Option A: The Fast Automated Flasher (Recommended)

The repository provides [upload_cluster.py](file:///c:/Users/Y_M/Documents/chained_esp32s3_tiny_llm/upload_cluster.py) and [flash.bat](file:///c:/Users/Y_M/Documents/chained_esp32s3_tiny_llm/flash.bat).

1. Double-click **`flash.bat`** (or run `py upload_cluster.py`).
2. Select your flashing mode:
   - **`[1] Fast Firmware Update` (Default, ~3 seconds per board)**: Updates bootloader, partitions, and application code. Leaves the 14MB model weights on flash intact.
   - **`[2] Full Flash`**: Flashes firmware AND uploads the 14MB model partition (`node1.bin`, `node2.bin`, `node3.bin`) to offset `0x13c000`.
3. Follow the interactive prompts to plug in Node 1 &rarr; Node 2 &rarr; Node 3.

```
=======================================================
     ESP32-S3 CLUSTER UPLOADER UTILITY
=======================================================
Select Flash Mode:
  [1] Fast Firmware Update (App only, ~3 sec per board) [DEFAULT]
  [2] Full Flash (App + 14MB Model Weights, ~2 min per board)
Choose mode (1 or 2, default=1): 1
[*] Auto-detected ESP32-S3 on COM5
[*] Flashing Node 1... PASSED!
```

---

### Option B: Manual Compilation via PlatformIO

If you want to build and flash each node individually from the command line:

```bash
# Compile and flash Node 1
pio run -e node1 -t upload --upload-port COM5

# Compile and flash Node 2
pio run -e node2 -t upload --upload-port COM5

# Compile and flash Node 3
pio run -e node3 -t upload --upload-port COM5
```

---

## 💬 Running Chef Gordon Ramsay

1. Open your favorite Serial Monitor (Arduino IDE, PuTTY, or PlatformIO Monitor) connected to **Node 1** at **115,200 baud** (Set Line Ending to **"Both NL & CR"** or **"Newline"**).
2. **Power Stabilization Buffer**: Upon startup, Node 1 initiates a 15-second countdown to allow Nodes 2 and 3 to boot and settle power on the 5V bus.
   - *Tip: Press **ENTER** at any time to skip the countdown immediately.*
3. **Ring Ping Verification**: The cluster pings around the ring (`Node 1 -> Node 2 -> Node 3 -> Node 1`).
4. **Interact**: Type any compiler error, exception, or programming mistake and hit Enter!

```
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

## 🛠️ Machine Learning Training Pipeline

If you want to train and export your own custom model from scratch:

```bash
# 1. Generate 300,000 synthetic paired error-roast examples
py generate_dataset.py

# 2. Train custom 2048-token SentencePiece BPE tokenizer
py gen_vocab.py

# 3. Train the 86.5M parameter LLaMA model in PyTorch
py train_tiny_llm.py

# 4. Quantize to 4-bit (Q4_0) and slice across 3 cluster partitions
py split_model.py my_fine_tuned_model.bin
```

---

## 📁 Repository File Structure

```
chained_esp32s3_tiny_llm/
├── flash.bat                  # One-click Windows flasher shortcut
├── upload_cluster.py          # Interactive cluster flasher (Fast vs Full modes)
├── platformio.ini             # PlatformIO build configuration for nodes 1, 2, 3
├── partitions.csv             # 16MB custom flash partition layout
├── PROJECT_OVERVIEW.txt       # Exhaustive technical manifesto & ML math breakdown
├── README.md                  # This documentation
│
├── main/
│   ├── main.cpp               # Core firmware (Paging, KV Cache, Ring Pipeline, Sampling)
│   ├── vocab.h                # Embedded BPE vocabulary array & string tables
│   └── CMakeLists.txt         # ESP-IDF component configuration
│
├── llama2.c/                  # PyTorch training & architecture reference
│   ├── model.py               # Transformer model definition (RMSNorm, RoPE, SwiGLU)
│   └── train.py               # Auto-regressive training loop
│
├── generate_dataset.py        # Synthetic dataset generator (300k multi-language pairs)
├── gen_vocab.py               # SentencePiece BPE vocabulary generator
├── train_tiny_llm.py          # PyTorch model training script
├── split_model.py             # 4-bit channel-wise quantizer and 3-way partition slicer
└── test_insults.py            # Local PC inference verification script
```

---

## 📜 Full Technical Manifesto
For deep dive documentation on mathematical formulations (RMSNorm, RoPE, Scaled Dot-Product Attention, SwiGLU), dynamic PSRAM paging benchmarks, inner assembly de-quantization loops, and the full debugging chronicle, see **[PROJECT_OVERVIEW.txt](file:///c:/Users/Y_M/Documents/chained_esp32s3_tiny_llm/PROJECT_OVERVIEW.txt)**.

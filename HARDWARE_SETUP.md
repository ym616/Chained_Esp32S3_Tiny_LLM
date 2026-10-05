# ESP32-S3 Tiny LLM Cluster Hardware Setup

This document outlines the physical wiring required to connect the three ESP32-S3 boards into a distributed inference cluster.

## 1. Shared Power Rail (Single USB-C)

Yes, you can absolutely power all three boards from a single USB-C cable! ESP32-S3 dev boards have onboard voltage regulators and can share power across their `5V` (sometimes labeled `VIN` or `VBUS`) pins.

**Power Wiring:**
1. Connect all three **`GND`** pins together. *(This is critically important for the data lines to work!)*
2. Connect all three **`5V`** pins together.
3. Plug your USB-C cable into **Node 1**. 

Node 1 will receive 5V from your PC, and it will distribute that 5V across the wires to power Node 2 and Node 3.

> **Important:** Do NOT plug multiple USB-C cables into your PC while the `5V` pins are wired together. During the initial flashing process, you must either disconnect the `5V` wire between the boards so you can plug all three into your PC safely, or flash them one by one. Once flashing is done, you can wire the 5V together and just use Node 1's USB port.

## 2. The Data Ring (UART Topology)

To divide the neural network across the boards, they must rapidly pass tensor data to each other in a circle. In our `main.cpp` code, we configured the data lines as:
* **TX Pin:** `GPIO 17`
* **RX Pin:** `GPIO 18`

You need to wire them in a perfect ring so the data flows in one continuous direction (Node 1 -> Node 2 -> Node 3 -> Node 1):

1. **Wire A:** Connect Node 1 `GPIO 17` (TX)  👉  Node 2 `GPIO 18` (RX)
2. **Wire B:** Connect Node 2 `GPIO 17` (TX)  👉  Node 3 `GPIO 18` (RX)
3. **Wire C:** Connect Node 3 `GPIO 17` (TX)  👉  Node 1 `GPIO 18` (RX)

## 3. Using the Cluster

Once everything is wired up and flashed:
1. Plug **Node 1** into your computer via USB-C. This provides power to the whole cluster.
2. Node 1 also acts as the "Master" node. You will open a Serial Monitor (like PuTTY, Arduino IDE Serial Monitor, or the ESP-IDF monitor) connected to Node 1's COM port at `115200` baud rate.
3. Type your prompt into the Serial Monitor (e.g., `<User>: IDE Error in C++... \n<Gordon>:`) and hit send.
4. Node 1 will process its chunk of the model, pass the data over Wire A to Node 2, which does its math, passes it over Wire B to Node 3, and so on, until Node 1 prints out Gordon Ramsay's insult back to your screen!

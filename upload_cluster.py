import sys
import os
import subprocess
import time
import serial.tools.list_ports

PYTHON_EXE = sys.executable

def find_esp32_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if "usb" in desc or "serial" in desc or "uart" in desc or "ch340" in desc or "cp210" in desc or "303a" in hwid:
            return p.device
    if ports:
        return ports[0].device
    return None

def flash_node(node_id, port, fast_mode=True):
    print(f"\n=======================================================")
    print(f"[*] FLASHING NODE {node_id} ON PORT {port}")
    if fast_mode:
        print(f"[*] MODE: FAST FIRMWARE UPDATE (Skipping 14MB model binary)")
    else:
        print(f"[*] MODE: FULL FLASH (Including 14MB model binary)")
    print(f"=======================================================")
    
    bootloader = f".pio/build/node{node_id}/bootloader.bin"
    partitions = f".pio/build/node{node_id}/partitions.bin"
    firmware   = f".pio/build/node{node_id}/firmware.bin"
    model_chunk= f"node{node_id}.bin"
    
    required = [bootloader, partitions, firmware]
    if not fast_mode:
        required.append(model_chunk)

    for f in required:
        if not os.path.exists(f):
            print(f"Error: Missing required file {f}")
            return False
            
    cmd = [
        PYTHON_EXE, "-m", "esptool",
        "--chip", "esp32s3",
        "--port", port,
        "--baud", "921600",
        "--before", "default_reset",
        "--after", "hard_reset",
        "write_flash", "-z",
        "--flash_mode", "dio",
        "--flash_freq", "40m",
        "--flash_size", "16MB",
        "0x0", bootloader,
        "0x8000", partitions,
        "0x10000", firmware
    ]

    if not fast_mode:
        cmd.extend(["0x13c000", model_chunk])
        print(f"Writing Bootloader (0x0), Partitions (0x8000), Firmware (0x10000), and Model (0x13c000)...")
    else:
        print(f"Writing Bootloader (0x0), Partitions (0x8000), and Firmware (0x10000) only (~3 seconds)...")
    
    res = subprocess.run(cmd)
    return res.returncode == 0

def main():
    print("=======================================================")
    print("      ESP32-S3 Tiny LLM Cluster Flash Utility          ")
    print("=======================================================")
    print("NOTE: Please make sure Arduino Serial Monitor is CLOSED")
    print("      so the COM port is free to flash.\n")
    
    print("Select Flash Mode:")
    print("  [1] Fast Firmware Update (takes ~3s per board - leaves 14MB model intact) [DEFAULT]")
    print("  [2] Full Flash (Firmware + 14MB Model Chunk - takes ~2 mins per board)")
    mode_choice = input("Enter choice (1 or 2, default is 1): ").strip()
    fast_mode = (mode_choice != "2")

    print("\nWhich nodes would you like to flash?")
    print("  [A] All 3 boards in sequence (Node 1 -> Node 2 -> Node 3) [DEFAULT]")
    print("  [1] Node 1 only")
    print("  [2] Node 2 only")
    print("  [3] Node 3 only")
    target_choice = input("Enter choice (A, 1, 2, or 3, default is A): ").strip().upper()

    if target_choice == "1":
        nodes = [1]
    elif target_choice == "2":
        nodes = [2]
    elif target_choice == "3":
        nodes = [3]
    else:
        nodes = [1, 2, 3]

    for node_id in nodes:
        while True:
            port = find_esp32_port()
            if port:
                print(f"\n>>> Board for Node {node_id} detected on {port}.")
                resp = input(f"Press [Enter] to flash Node {node_id} (or type a different COM port): ").strip()
                if resp:
                    port = resp
            else:
                port = input(f"\nPlease plug Node {node_id} into USB and enter COM port (e.g. COM5): ").strip()
                
            success = flash_node(node_id, port, fast_mode=fast_mode)
            if success:
                print(f"\n[+] SUCCESS: Node {node_id} is flashed!")
                if node_id != nodes[-1]:
                    next_node = nodes[nodes.index(node_id) + 1]
                    print(f"\n--- STEP: PREPARE NODE {next_node} ---")
                    print(f"1. Unplug the USB cable from Board {node_id}.")
                    print(f"2. Plug the USB cable into Board {next_node}.")
                    input(f"3. Press [Enter] once Board {next_node} is plugged in...")
                break
            else:
                retry = input("\n[!] Flash failed. Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    sys.exit(1)

    print("\n=======================================================")
    print("               FLASH OPERATION COMPLETE!               ")
    print("=======================================================")
    print("Next Steps:")
    print("1. Ensure data wires (TX GPIO 17 -> RX GPIO 18 in a ring) are connected.")
    print("2. Connect your shared 5V and GND power rail wires.")
    print("3. Plug your USB cable into Node 1 (Master / Head).")
    print("4. Open Serial Monitor on Node 1 (115200 baud).")
    print("5. Watch the 15-second buffer and ring sync confirmation, then enjoy!")

if __name__ == "__main__":
    main()

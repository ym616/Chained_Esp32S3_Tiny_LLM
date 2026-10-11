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

def flash_node(node_id, port):
    print(f"\n=======================================================")
    print(f"[*] FLASHING NODE {node_id} ON PORT {port}")
    print(f"=======================================================")
    
    bootloader = f".pio/build/node{node_id}/bootloader.bin"
    partitions = f".pio/build/node{node_id}/partitions.bin"
    firmware   = f".pio/build/node{node_id}/firmware.bin"
    model_chunk= f"node{node_id}.bin"
    
    for f in [bootloader, partitions, firmware, model_chunk]:
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
        "0x10000", firmware,
        "0x13c000", model_chunk
    ]
    
    print(f"Writing Bootloader (0x0), Partitions (0x8000), Firmware (0x10000), and Model (0x13c000)...")
    res = subprocess.run(cmd)
    return res.returncode == 0

def main():
    print("=======================================================")
    print("      ESP32-S3 Tiny LLM Cluster Flash Utility          ")
    print("=======================================================")
    print("This tool will guide you through flashing all 3 boards.")
    print("You only need 1 USB cable. You will flash them one-by-one.\n")
    
    for node_id in range(1, 4):
        while True:
            port = find_esp32_port()
            if port:
                print(f"\n>>> Board for Node {node_id} detected on {port}.")
                resp = input(f"Press [Enter] to flash Node {node_id} (or type a different COM port): ").strip()
                if resp:
                    port = resp
            else:
                port = input(f"\nPlease plug Node {node_id} into USB and enter COM port (e.g. COM5): ").strip()
                
            success = flash_node(node_id, port)
            if success:
                print(f"\n[+] SUCCESS: Node {node_id} is completely flashed!")
                if node_id < 3:
                    print(f"\n--- STEP {node_id} OF 3 COMPLETE ---")
                    print(f"1. Unplug the USB cable from Board {node_id}.")
                    print(f"2. Plug the USB cable into Board {node_id + 1}.")
                    input(f"3. Press [Enter] once Board {node_id + 1} is plugged in...")
                break
            else:
                retry = input("\n[!] Flash failed. Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    sys.exit(1)

    print("\n=======================================================")
    print("       ALL 3 ESP32-S3 BOARDS FLASHED SUCCESSFULLY!     ")
    print("=======================================================")
    print("Final Steps:")
    print("1. Ensure data wires (TX GPIO 17 -> RX GPIO 18 in a circle) are connected.")
    print("2. Reconnect the shared 5V and GND power rail wires.")
    print("3. Plug your single USB cable into Node 1 (Master).")
    print("4. Open Serial Monitor on Node 1 at 115200 baud to start roasting!")

if __name__ == "__main__":
    main()


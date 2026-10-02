import os
import subprocess
import sys
import re

def run_command(cmd, cwd=None):
    print(f"\n> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error executing: {cmd}")
        sys.exit(1)

def set_node_id(node_id):
    main_cpp_path = os.path.join("main", "main.cpp")
    if not os.path.exists(main_cpp_path):
        print(f"Error: {main_cpp_path} not found.")
        sys.exit(1)
        
    with open(main_cpp_path, 'r') as f:
        content = f.read()
        
    # Replace the #define NODE_ID X
    new_content = re.sub(r'#define NODE_ID \d+', f'#define NODE_ID {node_id}', content)
    
    with open(main_cpp_path, 'w') as f:
        f.write(new_content)
    
    print(f"[*] Updated main.cpp to NODE_ID {node_id}")

def main():
    print("=== Chained ESP32-S3 Cluster Deployment ===")
    
    # 1. Ask for COM ports
    ports = {}
    for i in range(1, 4):
        ports[i] = input(f"Enter the COM port for Node {i} (e.g. COM3 or /dev/ttyUSB0): ").strip()
        
    model_bin = "my_fine_tuned_model.bin"
    if not os.path.exists(model_bin):
        print(f"Warning: {model_bin} not found. Ensure you have run train_tiny_llm.py or provide a model.")
        choice = input("Do you want to continue just to flash firmware? (y/n): ")
        if choice.lower() != 'y':
            sys.exit(0)
    else:
        # 2. Split the model
        print("\n[*] Splitting the model for the 3 nodes...")
        run_command(f"python split_model.py {model_bin}")
    
    # 3. Build and flash each node
    print("\n[*] Starting build and flash process for all nodes...")
    for i in range(1, 4):
        print(f"\n=============================")
        print(f"=== DEPLOYING TO NODE {i} ===")
        print(f"=============================")
        
        # Modify the source file for this node
        set_node_id(i)
        
        # Build the firmware
        print(f"\n[*] Building firmware for Node {i}...")
        run_command("idf.py build")
        
        # Flash the firmware
        print(f"\n[*] Flashing firmware to Node {i} on {ports[i]}...")
        run_command(f"idf.py -p {ports[i]} flash")
        
        # Flash the model to SPIFFS
        spiffs_bin = f"node{i}.bin"
        if os.path.exists(spiffs_bin):
            print(f"\n[*] Flashing model partition to Node {i} on {ports[i]}...")
            run_command(f"parttool.py --port {ports[i]} write_partition --partition-name model --input {spiffs_bin}")
        else:
            print(f"Warning: {spiffs_bin} not found. Skipping SPIFFS upload for Node {i}.")
            
    print("\n=== Cluster Deployment Complete! ===")
    print("All 3 nodes have been flashed. Connect them via UART as per the README and reset them.")

if __name__ == "__main__":
    main()

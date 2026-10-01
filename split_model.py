import struct
import sys
import os

# Configuration for ~40M param model (e.g. stories42M)
# Adjust these based on the actual model architecture used in llama2.c
DIM = 512
HIDDEN_DIM = 1376
N_LAYERS = 8
N_HEADS = 8
N_KV_HEADS = 8
VOCAB_SIZE = 32000

def split_model(bin_path):
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found.")
        sys.exit(1)
        
    print(f"Splitting {bin_path}...")
    
    # Read entire binary
    with open(bin_path, 'rb') as f:
        data = f.read()

    # The llama2.c binary format (version 1 or 2):
    # Header: 256 bytes
    header = data[:256]
    weights_data = data[256:]
    
    # To properly split the model, we would divide the layers into 3 chunks.
    # For an 8 layer model:
    # Node 1: Embeddings + Layers 0, 1, 2
    # Node 2: Layers 3, 4, 5
    # Node 3: Layers 6, 7 + Classifier/Output
    
    # Due to the structure of llama2.c bin, weight tensors are grouped by type (e.g. all wq across all layers, then all wk, etc.)
    # A true "horizontal" split requires parsing the llama2.c weight layout, slicing the specific layers, and creating 3 custom bin files.
    # For this script, we'll split the file linearly by bytes as a placeholder for the actual tensor parsing logic.
    # In a real implementation, you will load the struct, slice the arrays, and serialize.
    
    total_size = len(weights_data)
    part_size = total_size // 3
    
    node1_data = weights_data[:part_size]
    node2_data = weights_data[part_size:2*part_size]
    node3_data = weights_data[2*part_size:]
    
    with open('node1.bin', 'wb') as f:
        f.write(header)
        f.write(node1_data)
        
    with open('node2.bin', 'wb') as f:
        f.write(header) # Node 2 might need header for dims
        f.write(node2_data)
        
    with open('node3.bin', 'wb') as f:
        f.write(header)
        f.write(node3_data)

    print("Splitting complete: node1.bin, node2.bin, node3.bin generated.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python split_model.py <model.bin>")
        sys.exit(1)
    split_model(sys.argv[1])

import struct
import sys
import os
import numpy as np

def quantize_tensor(tensor_bytes, rows, cols):
    """
    Quantizes a float32 tensor of shape (rows, cols) into 4-bit.
    Returns (scales_bytes, int4_packed_bytes)
    """
    tensor = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(rows, cols)
    max_abs = np.abs(tensor).max(axis=1)
    scales = max_abs / 7.0
    scales[scales == 0] = 1e-9
    
    # Quantize to -7 to 7
    quantized = np.clip(np.round(tensor / scales[:, None]), -7, 7).astype(np.int8)
    
    # Offset by +7 to make them strictly 0 to 14 (positive uint8)
    q_offset = (quantized + 7).astype(np.uint8)
    
    # Pack 2 values into 1 byte (cols must be even)
    packed = (q_offset[:, 0::2] & 0x0F) | ((q_offset[:, 1::2] & 0x0F) << 4)
    
    return scales.astype(np.float32).tobytes(), packed.tobytes()

def precompute_freqs_cis(head_dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    freqs_cos = np.cos(freqs).astype(np.float32)
    freqs_sin = np.sin(freqs).astype(np.float32)
    return freqs_cos.tobytes(), freqs_sin.tobytes()

def split_model(bin_path):
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found.")
        sys.exit(1)
        
    print(f"[*] Splitting and quantizing {bin_path}...")
    
    with open(bin_path, 'rb') as f:
        header = f.read(256)
        
        # Check if version 1 (starts with magic 0x616b3432 '24ka')
        if header[:4] == b'24ka':
            version = struct.unpack('i', header[4:8])[0]
            config = struct.unpack('7i', header[8:36])
            shared_classifier = struct.unpack('B', header[36:37])[0]
            is_v1 = True
        else:
            config = struct.unpack('7i', header[:28])
            shared_classifier = config[5] > 0
            is_v1 = False

        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len = config
        vocab_size = abs(vocab_size)
        
        print(f"Model Config: dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}")
        print(f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, vocab_size={vocab_size}, max_seq_len={max_seq_len}")
        print(f"Shared classifier: {shared_classifier}")
        
        head_size = dim // n_heads
        
        # Tensor byte sizes in float32
        S_tok_emb = vocab_size * dim * 4
        S_rms_att = dim * 4
        S_wq = dim * dim * 4
        S_wk = (n_kv_heads * head_size) * dim * 4
        S_wv = (n_kv_heads * head_size) * dim * 4
        S_wo = dim * dim * 4
        S_rms_ffn = dim * 4
        S_w1 = hidden_dim * dim * 4
        S_w2 = dim * hidden_dim * 4
        S_w3 = hidden_dim * dim * 4
        S_rms_final = dim * 4
        
        if is_v1:
            # Version 1 order:
            # attention_norm for all layers
            # ffn_norm for all layers
            # norm.weight (rms_final)
            # tok_embeddings
            # wq for all layers
            # wk for all layers
            # wv for all layers
            # wo for all layers
            # w1 for all layers
            # w2 for all layers
            # w3 for all layers
            # output (if not shared)
            rms_att = [f.read(S_rms_att) for _ in range(n_layers)]
            rms_ffn = [f.read(S_rms_ffn) for _ in range(n_layers)]
            rms_final = f.read(S_rms_final)
            tok_emb = f.read(S_tok_emb)
            wq = [f.read(S_wq) for _ in range(n_layers)]
            wk = [f.read(S_wk) for _ in range(n_layers)]
            wv = [f.read(S_wv) for _ in range(n_layers)]
            wo = [f.read(S_wo) for _ in range(n_layers)]
            w1 = [f.read(S_w1) for _ in range(n_layers)]
            w2 = [f.read(S_w2) for _ in range(n_layers)]
            w3 = [f.read(S_w3) for _ in range(n_layers)]
            if not shared_classifier:
                wcls = f.read(S_tok_emb)
            else:
                wcls = tok_emb
        else:
            # Legacy v0 order
            tok_emb = f.read(S_tok_emb)
            rms_att = [f.read(S_rms_att) for _ in range(n_layers)]
            wq = [f.read(S_wq) for _ in range(n_layers)]
            wk = [f.read(S_wk) for _ in range(n_layers)]
            wv = [f.read(S_wv) for _ in range(n_layers)]
            wo = [f.read(S_wo) for _ in range(n_layers)]
            rms_ffn = [f.read(S_rms_ffn) for _ in range(n_layers)]
            w1 = [f.read(S_w1) for _ in range(n_layers)]
            w2 = [f.read(S_w2) for _ in range(n_layers)]
            w3 = [f.read(S_w3) for _ in range(n_layers)]
            rms_final = f.read(S_rms_final)
            f.read(max_seq_len * (head_size // 2) * 4) # skip freq_cis_real
            f.read(max_seq_len * (head_size // 2) * 4) # skip freq_cis_imag
            if not shared_classifier:
                wcls = f.read(S_tok_emb)
            else:
                wcls = tok_emb
                
        freq_cis_real, freq_cis_imag = precompute_freqs_cis(head_size, max_seq_len)
            
    print("[*] Quantizing tensors to 4-bit packed format...")
    q_tok_emb = quantize_tensor(tok_emb, vocab_size, dim)
    q_wcls = quantize_tensor(wcls, vocab_size, dim)
    
    q_wq, q_wk, q_wv, q_wo = [], [], [], []
    q_w1, q_w2, q_w3 = [], [], []
    
    for l in range(n_layers):
        q_wq.append(quantize_tensor(wq[l], dim, dim))
        q_wk.append(quantize_tensor(wk[l], n_kv_heads * head_size, dim))
        q_wv.append(quantize_tensor(wv[l], n_kv_heads * head_size, dim))
        q_wo.append(quantize_tensor(wo[l], dim, dim))
        q_w1.append(quantize_tensor(w1[l], hidden_dim, dim))
        q_w2.append(quantize_tensor(w2[l], dim, hidden_dim))
        q_w3.append(quantize_tensor(w3[l], hidden_dim, dim))

    # Node partition logic
    layers_per_node = [n_layers // 3] * 3
    for i in range(n_layers % 3):
        layers_per_node[i] += 1
        
    node_boundaries = [0, layers_per_node[0], layers_per_node[0] + layers_per_node[1], n_layers]
    
    for node_idx in range(1, 4):
        start_l = node_boundaries[node_idx - 1]
        end_l = node_boundaries[node_idx]
        l_count = end_l - start_l
        
        out_path = f'node{node_idx}.bin'
        with open(out_path, 'wb') as out:
            # Node header (256 bytes):
            # First 28 bytes are Config struct expected by ESP32 C++ main.cpp
            mod_config = struct.pack('7i', dim, hidden_dim, l_count, n_heads, n_kv_heads, vocab_size, max_seq_len)
            out.write(mod_config)
            out.write(b'\0' * (256 - len(mod_config))) # pad to 256 bytes
            
            if node_idx == 1:
                out.write(q_tok_emb[0]); out.write(q_tok_emb[1]) # scales, weights
                
            for i in range(start_l, end_l):
                out.write(rms_att[i])
                out.write(q_wq[i][0]); out.write(q_wq[i][1])
                out.write(q_wk[i][0]); out.write(q_wk[i][1])
                out.write(q_wv[i][0]); out.write(q_wv[i][1])
                out.write(q_wo[i][0]); out.write(q_wo[i][1])
                out.write(rms_ffn[i])
                out.write(q_w1[i][0]); out.write(q_w1[i][1])
                out.write(q_w2[i][0]); out.write(q_w2[i][1])
                out.write(q_w3[i][0]); out.write(q_w3[i][1])
            
            if node_idx == 3:
                out.write(rms_final)
                out.write(q_wcls[0]); out.write(q_wcls[1])
                
            out.write(freq_cis_real)
            out.write(freq_cis_imag)

        size_mb = os.path.getsize(out_path) / (1024*1024)
        print(f"[*] Node {node_idx}.bin generated with {l_count} layers ({start_l} to {end_l-1}). Size: {size_mb:.2f} MB")

    print("\n[+] Model split complete! Generated node1.bin, node2.bin, and node3.bin")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python split_model.py <model.bin>")
        sys.exit(1)
    split_model(sys.argv[1])

import struct
import sys
import os
import numpy as np

def read_int(f):
    b = f.read(4)
    if not b: return None
    return struct.unpack('i', b)[0]

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

def quantize_1d(tensor_bytes, size):
    """
    1D tensors like RMSNorm weights are left as float32 to preserve precision.
    Returns original bytes.
    """
    return tensor_bytes

def split_model(bin_path):
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found.")
        sys.exit(1)
        
    print(f"Splitting and quantizing {bin_path}...")
    
    with open(bin_path, 'rb') as f:
        header = f.read(256)
        config = struct.unpack('7i', header[:28])
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len = config
        
        print(f"Model Config: dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}")
        print(f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, vocab_size={vocab_size}, max_seq_len={max_seq_len}")
        
        head_size = dim // n_heads
        
        # Sizes in float32
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
        S_freq_cis_real = max_seq_len * (head_size // 2) * 4
        S_freq_cis_imag = max_seq_len * (head_size // 2) * 4
        
        # Read tensors
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
        freq_cis_real = f.read(S_freq_cis_real)
        freq_cis_imag = f.read(S_freq_cis_imag)
        
        # Check if weights are shared (wcls == tok_emb)
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        expected_size_shared = 256 + S_tok_emb + n_layers * (S_rms_att + S_wq + S_wk + S_wv + S_wo + S_rms_ffn + S_w1 + S_w2 + S_w3) + S_rms_final + S_freq_cis_real + S_freq_cis_imag
        
        if file_size > expected_size_shared:
            f.seek(expected_size_shared)
            wcls = f.read(S_tok_emb)
        else:
            wcls = tok_emb
            
    print("Quantizing tensors to 8-bit...")
    # Quantize everything
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
        
        with open(f'node{node_idx}.bin', 'wb') as out:
            # Modify config to specify l_count and perhaps a flag for quantization
            mod_config = struct.pack('7i', dim, hidden_dim, l_count, n_heads, n_kv_heads, vocab_size, max_seq_len)
            out.write(mod_config)
            out.write(header[28:]) # remaining padding
            
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

        size_mb = os.path.getsize(f'node{node_idx}.bin') / (1024*1024)
        print(f"Node {node_idx}.bin generated with {l_count} layers ({start_l} to {end_l-1}). Size: {size_mb:.2f} MB")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python split_model.py <model.bin>")
        sys.exit(1)
    split_model(sys.argv[1])

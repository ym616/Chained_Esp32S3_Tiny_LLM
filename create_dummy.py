import struct
import numpy as np

def create_dummy_bin(filepath):
    # Config: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
    # Let's use small numbers to test:
    # dim=16, hidden_dim=32, n_layers=3, n_heads=4, n_kv_heads=4, vocab_size=64, max_seq_len=128
    dim = 16
    hidden_dim = 32
    n_layers = 3
    n_heads = 4
    n_kv_heads = 4
    vocab_size = 64
    max_seq_len = 128
    
    header = struct.pack('7i', dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len)
    header += b'\x00' * (256 - len(header))
    
    head_size = dim // n_heads
    
    with open(filepath, 'wb') as f:
        f.write(header)
        
        # tok_emb
        f.write(np.random.randn(vocab_size, dim).astype(np.float32).tobytes())
        
        # layers
        for _ in range(n_layers):
            f.write(np.random.randn(dim).astype(np.float32).tobytes()) # rms_att
            f.write(np.random.randn(dim, dim).astype(np.float32).tobytes()) # wq
            f.write(np.random.randn(dim, n_kv_heads * head_size).astype(np.float32).tobytes()) # wk
            f.write(np.random.randn(dim, n_kv_heads * head_size).astype(np.float32).tobytes()) # wv
            f.write(np.random.randn(dim, dim).astype(np.float32).tobytes()) # wo
            f.write(np.random.randn(dim).astype(np.float32).tobytes()) # rms_ffn
            f.write(np.random.randn(hidden_dim, dim).astype(np.float32).tobytes()) # w1
            f.write(np.random.randn(dim, hidden_dim).astype(np.float32).tobytes()) # w2
            f.write(np.random.randn(hidden_dim, dim).astype(np.float32).tobytes()) # w3
            
        # rms_final
        f.write(np.random.randn(dim).astype(np.float32).tobytes())
        
        # freq_cis
        f.write(np.random.randn(max_seq_len, head_size // 2).astype(np.float32).tobytes()) # real
        f.write(np.random.randn(max_seq_len, head_size // 2).astype(np.float32).tobytes()) # imag
        
        # wcls (unshared)
        # To test shared, we can either append it or not. Let's not append to test shared_weights=True
        
if __name__ == "__main__":
    create_dummy_bin("dummy.bin")
    print("Created dummy.bin")

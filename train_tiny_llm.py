import os
import subprocess
import sys
import shutil

def run_command(cmd, cwd=None):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error executing: {cmd}")
        sys.exit(1)

def main():
    print("=== Tiny LLM Training Setup (using llama2.c) ===")
    
    # 1. Clone the repository
    repo_dir = "llama2.c"
    if not os.path.exists(repo_dir):
        print("\n[*] Cloning llama2.c repository...")
        run_command("git clone https://github.com/karpathy/llama2.c.git")
    else:
        print("\n[*] llama2.c repository already exists. Skipping clone.")

    # 2. Download and pre-tokenize dataset (TinyStories)
    print("\n[*] Downloading and preparing TinyStories dataset...")
    # This downloads the dataset and pretokenizes it using the custom tokenizer
    run_command("python tinystories.py download", cwd=repo_dir)
    run_command("python tinystories.py pretokenize", cwd=repo_dir)

    # 3. Train a tiny model
    # We will configure a tiny model (e.g. 15M parameters) so it trains relatively quickly
    # and fits on the ESP32s. You can tweak these parameters.
    print("\n[*] Starting training (this will take a while)...")
    # For CPU training or quick testing, you can use: --device=cpu --compile=False --max_iters=100
    # Here we set a very small configuration suitable for ESP32 constraints.
    train_cmd = (
        "python train.py "
        "--out_dir=out "
        "--dataset=tinystories "
        "--dim=288 "          # Hidden dimension
        "--n_layers=6 "       # Number of layers
        "--n_heads=6 "        # Number of attention heads
        "--n_kv_heads=6 "
        "--multiple_of=32 "
        "--max_seq_len=256 "  # Max context length
        "--vocab_source=custom "
        "--vocab_size=4096 "  # Smaller vocab for memory savings
        "--batch_size=16 "    # Lower batch size
        "--max_iters=5000 "   # Adjust based on desired quality vs time
        "--device=cpu "       # Change to 'cuda' if you have an Nvidia GPU
        "--compile=False"     # Disable PyTorch compile for simpler setup
    )
    run_command(train_cmd, cwd=repo_dir)

    # 4. Export the model to binary format
    print("\n[*] Exporting trained model to binary format...")
    # Export the final model. model.pt is saved in the out_dir.
    export_cmd = "python export.py ../my_fine_tuned_model.bin --meta-vocab tinystories --version 1"
    run_command(export_cmd, cwd=repo_dir)

    print("\n=== Training and Export Complete! ===")
    print("Your model is saved as 'my_fine_tuned_model.bin'.")
    print("Next step: Use deploy_cluster.py to split and flash the model to the ESP32s.")

if __name__ == "__main__":
    main()

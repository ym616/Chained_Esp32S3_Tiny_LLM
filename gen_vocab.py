import os

vocab_file = "llama2.c/data/tok2048.vocab"
out_file = "main/vocab.h"

with open(vocab_file, "r", encoding="utf-8") as f:
    lines = [line.strip().split("\t")[0] for line in f if line.strip()]

print(f"Loaded {len(lines)} tokens from {vocab_file}")

with open(out_file, "w", encoding="utf-8") as f:
    f.write("#pragma once\n")
    f.write("#include <Arduino.h>\n\n")
    f.write("static const char* const VOCAB_TABLE[2048] = {\n")
    for i in range(2048):
        tok = lines[i] if i < len(lines) else ""
        tok = tok.replace("\u2581", " ")
        tok = tok.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        f.write(f'    "{tok}", // {i}\n')
    f.write("};\n\n")
    f.write("inline void print_token(int id) {\n")
    f.write("    if (id < 0 || id >= 2048) return;\n")
    f.write("    const char* str = VOCAB_TABLE[id];\n")
    f.write("    int len = strlen(str);\n")
    f.write("    if (len > 0 && str[0] == '<' && str[len - 1] == '>') return;\n")
    f.write("    Serial.print(str);\n")
    f.write("}\n")

print(f"Generated {out_file} successfully!")


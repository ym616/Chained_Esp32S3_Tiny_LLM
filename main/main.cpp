#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_partition.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include <Arduino.h>

static const char *TAG = "CHAINED_ESP32S3_LLM";

#ifndef NODE_ID
#define NODE_ID 1
#endif

// UART Ring Configuration
#define UART_NUM UART_NUM_2
#define TXD_PIN (GPIO_NUM_17)
#define RXD_PIN (GPIO_NUM_18)
#define UART_BAUD_RATE 921600

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    float* token_embedding_scales;
    uint8_t* token_embedding_table;
    float* rms_final_weight;
    float* wcls_scales;
    uint8_t* wcls;
    float* freq_cis_real;
    float* freq_cis_imag;
} GlobalWeights;

typedef struct {
    float* rms_att_weight;
    float* wq_scales; uint8_t* wq;
    float* wk_scales; uint8_t* wk;
    float* wv_scales; uint8_t* wv;
    float* wo_scales; uint8_t* wo;
    float* rms_ffn_weight;
    float* w1_scales; uint8_t* w1;
    float* w2_scales; uint8_t* w2;
    float* w3_scales; uint8_t* w3;
} SingleLayerWeights;

Config config;
GlobalWeights weights;
SingleLayerWeights cur_layer;
const esp_partition_t* model_part = NULL;

size_t layer_size = 0;
size_t layer_flash_offsets[32];
uint8_t* layer_cache = NULL;

float* activation_vector = NULL;
int8_t* xq_buffer = NULL;
float* xb = NULL;
float* xb2 = NULL;
float* hb = NULL;
float* hb2 = NULL;
float* q = NULL;
float* k = NULL;
float* v = NULL;

#define MAX_SEQ_LEN 64
float* key_cache = NULL;
float* value_cache = NULL;
float* att_buffer = NULL;


void init_uart_ring() {
    uart_config_t uart_config = {
        .baud_rate = UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk = UART_SCLK_APB,
    };
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM, 1024 * 32, 1024 * 32, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
}

void load_model() {
    Serial.printf("[Node %d] Locating model partition on flash...\n", NODE_ID);
    model_part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!model_part) {
        Serial.printf("[Node %d] ERROR: 'model' partition not found on flash!\n", NODE_ID);
        return;
    }

    uint8_t header[256];
    esp_partition_read(model_part, 0, header, sizeof(Config));
    memcpy(&config, header, sizeof(Config));

    Serial.printf("[Node %d] Model Config: dim=%d, hidden=%d, layers=%d, heads=%d, vocab=%d\n",
                  NODE_ID, config.dim, config.hidden_dim, config.n_layers, config.n_heads, config.vocab_size);

    int head_size = config.dim / config.n_heads;
    int kv_dim = config.n_kv_heads * head_size;
    size_t offset = 256;

    if (NODE_ID == 1) {
        size_t tok_emb_size = config.vocab_size * sizeof(float) + (config.vocab_size * config.dim) / 2;
        weights.token_embedding_scales = (float*)heap_caps_malloc(tok_emb_size, MALLOC_CAP_SPIRAM);
        weights.token_embedding_table = (uint8_t*)(weights.token_embedding_scales + config.vocab_size);
        esp_partition_read(model_part, offset, weights.token_embedding_scales, tok_emb_size);
        offset += tok_emb_size;
    }

    layer_size = config.dim * sizeof(float)
               + 4 * (config.dim * sizeof(float) + (config.dim * config.dim) / 2)
               + config.dim * sizeof(float)
               + 2 * (config.hidden_dim * sizeof(float) + (config.hidden_dim * config.dim) / 2)
               + (config.dim * sizeof(float) + (config.dim * config.hidden_dim) / 2);

    for (int l = 0; l < config.n_layers; l++) {
        layer_flash_offsets[l] = offset;
        offset += layer_size;
    }

    if (NODE_ID == 3) {
        size_t rms_final_size = config.dim * sizeof(float);
        weights.rms_final_weight = (float*)heap_caps_malloc(rms_final_size, MALLOC_CAP_SPIRAM);
        esp_partition_read(model_part, offset, weights.rms_final_weight, rms_final_size);
        offset += rms_final_size;

        size_t wcls_size = config.vocab_size * sizeof(float) + (config.vocab_size * config.dim) / 2;
        weights.wcls_scales = (float*)heap_caps_malloc(wcls_size, MALLOC_CAP_SPIRAM);
        weights.wcls = (uint8_t*)(weights.wcls_scales + config.vocab_size);
        esp_partition_read(model_part, offset, weights.wcls_scales, wcls_size);
        offset += wcls_size;
    }

    size_t freq_size = config.seq_len * (head_size / 2) * sizeof(float);
    weights.freq_cis_real = (float*)heap_caps_malloc(freq_size, MALLOC_CAP_SPIRAM);
    esp_partition_read(model_part, offset, weights.freq_cis_real, freq_size);
    offset += freq_size;

    weights.freq_cis_imag = (float*)heap_caps_malloc(freq_size, MALLOC_CAP_SPIRAM);
    esp_partition_read(model_part, offset, weights.freq_cis_imag, freq_size);

    layer_cache = (uint8_t*)heap_caps_malloc(layer_size, MALLOC_CAP_SPIRAM);
    activation_vector = (float*)heap_caps_malloc(config.dim * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    xb = (float*)heap_caps_malloc(config.dim * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    xb2 = (float*)heap_caps_malloc(config.dim * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    hb = (float*)heap_caps_malloc(config.hidden_dim * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    hb2 = (float*)heap_caps_malloc(config.hidden_dim * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    q = (float*)heap_caps_malloc(config.dim * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    k = (float*)heap_caps_malloc(config.dim * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    v = (float*)heap_caps_malloc(config.dim * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    xq_buffer = (int8_t*)heap_caps_malloc((config.hidden_dim > config.dim ? config.hidden_dim : config.dim), MALLOC_CAP_8BIT);

    size_t kv_cache_size = config.n_layers * MAX_SEQ_LEN * kv_dim * sizeof(float);
    key_cache = (float*)heap_caps_malloc(kv_cache_size, MALLOC_CAP_SPIRAM);
    value_cache = (float*)heap_caps_malloc(kv_cache_size, MALLOC_CAP_SPIRAM);
    att_buffer = (float*)heap_caps_malloc(config.n_heads * MAX_SEQ_LEN * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    Serial.printf("[Node %d] Weights & KV Cache loaded into PSRAM! Free PSRAM: %d KB\n",
                  NODE_ID, heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);
}

void load_layer(int l) {
    esp_partition_read(model_part, layer_flash_offsets[l], layer_cache, layer_size);
    uint8_t* ptr = layer_cache;
    int head_size = config.dim / config.n_heads;
    int kv_dim = config.n_kv_heads * head_size;

    cur_layer.rms_att_weight = (float*)ptr; ptr += config.dim * sizeof(float);
    cur_layer.wq_scales = (float*)ptr; ptr += config.dim * sizeof(float);
    cur_layer.wq = (uint8_t*)ptr; ptr += (config.dim * config.dim) / 2;
    cur_layer.wk_scales = (float*)ptr; ptr += kv_dim * sizeof(float);
    cur_layer.wk = (uint8_t*)ptr; ptr += (kv_dim * config.dim) / 2;
    cur_layer.wv_scales = (float*)ptr; ptr += kv_dim * sizeof(float);
    cur_layer.wv = (uint8_t*)ptr; ptr += (kv_dim * config.dim) / 2;
    cur_layer.wo_scales = (float*)ptr; ptr += config.dim * sizeof(float);
    cur_layer.wo = (uint8_t*)ptr; ptr += (config.dim * config.dim) / 2;

    cur_layer.rms_ffn_weight = (float*)ptr; ptr += config.dim * sizeof(float);
    cur_layer.w1_scales = (float*)ptr; ptr += config.hidden_dim * sizeof(float);
    cur_layer.w1 = (uint8_t*)ptr; ptr += (config.hidden_dim * config.dim) / 2;
    cur_layer.w2_scales = (float*)ptr; ptr += config.dim * sizeof(float);
    cur_layer.w2 = (uint8_t*)ptr; ptr += (config.dim * config.hidden_dim) / 2;
    cur_layer.w3_scales = (float*)ptr; ptr += config.hidden_dim * sizeof(float);
    cur_layer.w3 = (uint8_t*)ptr; ptr += (config.hidden_dim * config.dim) / 2;
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

void matmul_q(float* xout, float* x, uint8_t* w_packed, float* scales, int n, int d) {
    float max_val = 0.0f;
    for (int i = 0; i < n; i++) {
        float f = fabsf(x[i]);
        if (f > max_val) max_val = f;
    }
    float x_scale = max_val / 127.0f;
    if (x_scale == 0.0f) x_scale = 1e-9f;
    float inv_x_scale = 1.0f / x_scale;

    for (int i = 0; i < n; i++) {
        xq_buffer[i] = (int8_t)roundf(x[i] * inv_x_scale);
    }

    for (int i = 0; i < d; i++) {
        int32_t val = 0;
        int idx = i * (n / 2);
        for (int j = 0; j < (n / 2); j++) {
            uint8_t packed = w_packed[idx + j];
            int8_t w0 = (packed & 0x0F) - 7;
            int8_t w1 = ((packed >> 4) & 0x0F) - 7;
            val += w0 * xq_buffer[j * 2];
            val += w1 * xq_buffer[j * 2 + 1];
        }
        xout[i] = ((float)val) * (scales[i] * x_scale);
    }
}

static void softmax_f(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

void compute_layers(float* x, int pos) {
    if (pos >= MAX_SEQ_LEN) pos = MAX_SEQ_LEN - 1;
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int head_size = dim / config.n_heads;
    int kv_dim = config.n_kv_heads * head_size;
    int kv_mul = config.n_heads / config.n_kv_heads;

    for (int l = 0; l < config.n_layers; l++) {
        load_layer(l);

        rmsnorm(xb, x, cur_layer.rms_att_weight, dim);

        int loff = l * MAX_SEQ_LEN * kv_dim;
        float* k_dest = key_cache + loff + pos * kv_dim;
        float* v_dest = value_cache + loff + pos * kv_dim;

        matmul_q(q, xb, cur_layer.wq, cur_layer.wq_scales, dim, dim);
        matmul_q(k_dest, xb, cur_layer.wk, cur_layer.wk_scales, dim, kv_dim);
        matmul_q(v_dest, xb, cur_layer.wv, cur_layer.wv_scales, dim, kv_dim);

        // RoPE on q and k_dest
        for (int i = 0; i < dim; i += 2) {
            float fcr = weights.freq_cis_real[pos * (head_size / 2) + (i % head_size) / 2];
            float fci = weights.freq_cis_imag[pos * (head_size / 2) + (i % head_size) / 2];
            float q0 = q[i]; float q1 = q[i + 1];
            q[i]     = q0 * fcr - q1 * fci;
            q[i + 1] = q0 * fci + q1 * fcr;
            if (i < kv_dim) {
                float k0 = k_dest[i]; float k1 = k_dest[i + 1];
                k_dest[i]     = k0 * fcr - k1 * fci;
                k_dest[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // Multihead attention across all timesteps 0..pos
        for (int h = 0; h < config.n_heads; h++) {
            float* q_head = q + h * head_size;
            float* att_head = att_buffer + h * MAX_SEQ_LEN;

            for (int t = 0; t <= pos; t++) {
                float* k_t = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q_head[i] * k_t[i];
                }
                score /= sqrtf((float)head_size);
                att_head[t] = score;
            }

            softmax_f(att_head, pos + 1);

            float* xb_head = xb2 + h * head_size;
            memset(xb_head, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v_t = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att_head[t];
                for (int i = 0; i < head_size; i++) {
                    xb_head[i] += a * v_t[i];
                }
            }
        }

        matmul_q(xb, xb2, cur_layer.wo, cur_layer.wo_scales, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += xb[i];

        rmsnorm(xb, x, cur_layer.rms_ffn_weight, dim);

        matmul_q(hb, xb, cur_layer.w1, cur_layer.w1_scales, dim, hidden_dim);
        matmul_q(hb2, xb, cur_layer.w3, cur_layer.w3_scales, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; i++) {
            float val = hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= hb2[i];
            hb[i] = val;
        }

        matmul_q(xb, hb, cur_layer.w2, cur_layer.w2_scales, hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += xb[i];

        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

// ========================================================
// Bulletproof Ring Packet Protocol with Sequence Position
// ========================================================
#define MAGIC_0 0xA5
#define MAGIC_1 0x5A

#define PKT_PING       0x01
#define PKT_ACTIVATION 0x02
#define PKT_TOKEN      0x03

static void send_packet(uint8_t type, uint8_t pos, const uint8_t* payload, uint16_t len) {
    uint8_t header[6] = {
        MAGIC_0,
        MAGIC_1,
        type,
        pos,
        (uint8_t)(len >> 8),
        (uint8_t)(len & 0xFF)
    };
    uart_write_bytes(UART_NUM, (const char*)header, 6);
    if (len > 0 && payload != NULL) {
        uart_write_bytes(UART_NUM, (const char*)payload, len);
    }
}

static bool receive_packet(uint8_t* type, uint8_t* pos, uint8_t* payload, uint16_t max_len, uint16_t* out_len, uint32_t timeout_ms) {
    TickType_t start = xTaskGetTickCount();
    TickType_t max_ticks = pdMS_TO_TICKS(timeout_ms);

    // 1. Sync on MAGIC_0 and MAGIC_1
    uint8_t b = 0;
    while (1) {
        if ((xTaskGetTickCount() - start) > max_ticks) return false;
        int n = uart_read_bytes(UART_NUM, &b, 1, pdMS_TO_TICKS(20));
        if (n <= 0) {
            vTaskDelay(pdMS_TO_TICKS(2));
            continue;
        }
        if (b == MAGIC_0) {
            n = uart_read_bytes(UART_NUM, &b, 1, pdMS_TO_TICKS(50));
            if (n > 0 && b == MAGIC_1) {
                break;
            }
        }
    }

    // 2. Read type, pos, and payload length (4 bytes)
    uint8_t meta[4];
    int meta_read = 0;
    while (meta_read < 4) {
        if ((xTaskGetTickCount() - start) > max_ticks) return false;
        int n = uart_read_bytes(UART_NUM, meta + meta_read, 4 - meta_read, pdMS_TO_TICKS(50));
        if (n > 0) meta_read += n;
        else vTaskDelay(pdMS_TO_TICKS(2));
    }

    *type = meta[0];
    if (pos) *pos = meta[1];
    uint16_t payload_len = ((uint16_t)meta[2] << 8) | meta[3];
    if (out_len) *out_len = payload_len;

    if (payload_len > max_len) {
        return false;
    }

    // 3. Read exact payload
    uint16_t payload_read = 0;
    while (payload_read < payload_len) {
        if ((xTaskGetTickCount() - start) > max_ticks) return false;
        int n = uart_read_bytes(UART_NUM, payload + payload_read, payload_len - payload_read, pdMS_TO_TICKS(50));
        if (n > 0) payload_read += n;
        else vTaskDelay(pdMS_TO_TICKS(2));
    }

    return true;
}

#if NODE_ID == 1
#include "vocab.h"

static bool ping_ring() {
    uart_flush_input(UART_NUM);
    send_packet(PKT_PING, 0, NULL, 0);

    uint8_t rx_type = 0;
    uint8_t rx_pos = 0;
    uint16_t rx_len = 0;
    uint8_t dummy[16];
    bool ok = receive_packet(&rx_type, &rx_pos, dummy, sizeof(dummy), &rx_len, 2000);
    return (ok && rx_type == PKT_PING);
}

void head_node_task(void *pvParameters) {
    Serial.println("\n========================================================");
    Serial.println("     ESP32-S3 CLUSTER STARTUP BUFFER (15 SECONDS)");
    Serial.println("========================================================");
    Serial.println("[Node 1] Giving Node 2 & Node 3 time to power on and boot...");
    Serial.println("[Node 1] (Press ENTER in Serial Monitor anytime to skip wait)\n");

    for (int sec = 15; sec > 0; sec--) {
        if (Serial.available()) {
            Serial.readStringUntil('\n');
            Serial.println("[Node 1] Buffer skipped by user!");
            break;
        }
        Serial.printf("[Node 1] Checking cluster ring in %d seconds...\n", sec);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    bool ring_synced = false;
    Serial.println("\n[Node 1] Testing ring link (Node 1 -> Node 2 -> Node 3 -> Node 1)...");
    for (int attempt = 1; attempt <= 5; attempt++) {
        Serial.printf("[Node 1] Pinging ring (Attempt %d/5)... ", attempt);
        if (ping_ring()) {
            ring_synced = true;
            Serial.println("PASSED!");
            break;
        } else {
            Serial.println("no response yet.");
        }
        vTaskDelay(pdMS_TO_TICKS(1500));
    }

    if (ring_synced) {
        Serial.println("\n========================================================");
        Serial.println("  [+] SUCCESS: ALL 3 ESP32-S3 NODES LINKED IN THE RING! ");
        Serial.println("========================================================");
        Serial.println("Ring pipeline: Node 1 -> Node 2 -> Node 3 -> Node 1 is ACTIVE.");
    } else {
        Serial.println("\n========================================================");
        Serial.println("  [!] NOTICE: Ring ping check timed out.");
        Serial.println("========================================================");
        Serial.println("  - Check that Node 2 & Node 3 are powered on 5V and GND.");
        Serial.println("  - Check UART ring: Node 1 TX (17) -> Node 2 RX (18)");
        Serial.println("                     Node 2 TX (17) -> Node 3 RX (18)");
        Serial.println("                     Node 3 TX (17) -> Node 1 RX (18)");
        Serial.println("  - You can type 'ping' in Serial Monitor at any time to re-test!\n");
    }

    Serial.println("\n========================================================");
    Serial.println("     CHEF GORDON RAMSAY ESP32-S3 CLUSTER READY!         ");
    Serial.println("========================================================");
    Serial.println("How to use: Type an error into Serial Monitor & hit ENTER.");
    Serial.println("Examples:   'NullPointerException' or 'SyntaxError' or 'ping'");
    Serial.println("Or simply hit ENTER for an instant Gordon Ramsay roast!");
    Serial.print("\n> ");

    int vector_bytes = config.dim * sizeof(float);
    int prompt_tokens[48];

    while(1) {
        if (Serial.available()) {
            String user_input = Serial.readStringUntil('\n');
            user_input.trim();
            if (user_input.equalsIgnoreCase("ping")) {
                Serial.println("\n[Node 1] Re-testing cluster ring connectivity...");
                if (ping_ring()) {
                    ring_synced = true;
                    Serial.println("[+] Ring test PASSED! All 3 nodes responding.");
                } else {
                    Serial.println("[-] Ring test failed. Check wires and power.");
                }
                Serial.print("\n> ");
                continue;
            }

            if (user_input.length() == 0) {
                user_input = "SyntaxError: unexpected EOF while parsing";
            }
            Serial.printf("\n[User Error]: \"%s\"\n", user_input.c_str());

            // Build the exact prompt format from training
            String prompt_str = "<User>: IDE Error:\n" + user_input + "\n<Gordon>:";
            int num_prompt_tokens = tokenize_prompt(prompt_str.c_str(), prompt_tokens, 48);

            Serial.print("[Chef Gordon Ramsay]: ");

            int pos = 0;
            int token_id = 0;
            bool success = true;

            // 1. Ingest prompt tokens quietly into the KV cache across all nodes
            for (int p = 0; p < num_prompt_tokens; p++) {
                token_id = prompt_tokens[p];

                float scale = weights.token_embedding_scales[token_id];
                uint8_t* row = weights.token_embedding_table + token_id * (config.dim / 2);
                for (int i = 0; i < (config.dim / 2); i++) {
                    uint8_t packed = row[i];
                    int8_t w0 = (packed & 0x0F) - 7;
                    int8_t w1 = ((packed >> 4) & 0x0F) - 7;
                    activation_vector[i * 2]     = ((float)w0) * scale;
                    activation_vector[i * 2 + 1] = ((float)w1) * scale;
                }

                compute_layers(activation_vector, p);
                send_packet(PKT_ACTIVATION, (uint8_t)p, (const uint8_t*)activation_vector, vector_bytes);

                uint8_t rx_type = 0;
                uint8_t rx_pos = 0;
                uint16_t rx_len = 0;
                int next_tok = -1;
                bool ok = receive_packet(&rx_type, &rx_pos, (uint8_t*)&next_tok, sizeof(int), &rx_len, 8000);

                if (!ok || rx_type != PKT_TOKEN) {
                    Serial.println("\n[!] Pipeline timeout waiting for cluster response.");
                    Serial.println("    Type 'ping' to verify ring wiring.");
                    success = false;
                    break;
                }
                token_id = next_tok;
            }

            pos = num_prompt_tokens;

            // 2. Stream Gordon's response tokens
            if (success) {
                for (int gen = 0; gen < 25; gen++) {
                    if (is_stop_token(token_id)) break;

                    print_token(token_id);

                    // End roast if sentence ends on ! or ? after generating a few words
                    const char* tstr = VOCAB_TABLE[token_id];
                    if (gen >= 5 && (strchr(tstr, '!') != NULL || strchr(tstr, '?') != NULL)) {
                        break;
                    }

                    float scale = weights.token_embedding_scales[token_id];
                    uint8_t* row = weights.token_embedding_table + token_id * (config.dim / 2);
                    for (int i = 0; i < (config.dim / 2); i++) {
                        uint8_t packed = row[i];
                        int8_t w0 = (packed & 0x0F) - 7;
                        int8_t w1 = ((packed >> 4) & 0x0F) - 7;
                        activation_vector[i * 2]     = ((float)w0) * scale;
                        activation_vector[i * 2 + 1] = ((float)w1) * scale;
                    }

                    compute_layers(activation_vector, pos);
                    send_packet(PKT_ACTIVATION, (uint8_t)pos, (const uint8_t*)activation_vector, vector_bytes);

                    uint8_t rx_type = 0;
                    uint8_t rx_pos = 0;
                    uint16_t rx_len = 0;
                    int next_tok = -1;
                    bool ok = receive_packet(&rx_type, &rx_pos, (uint8_t*)&next_tok, sizeof(int), &rx_len, 8000);

                    if (!ok || rx_type != PKT_TOKEN) {
                        Serial.println("\n[!] Generation halted: Node 3 response timeout.");
                        break;
                    }

                    token_id = next_tok;
                    pos++;
                    if (pos >= MAX_SEQ_LEN) break;
                }
            }

            Serial.println("\n\nType next error (or press ENTER to roast again):");
            Serial.print("> ");
        }

        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

#elif NODE_ID == 2
void body_node_task(void *pvParameters) {
    uart_flush_input(UART_NUM);
    Serial.println("[Node 2] Online! Listening for ring packets...");
    int vector_bytes = config.dim * sizeof(float);

    uint8_t rx_type = 0;
    uint8_t rx_pos = 0;
    uint16_t rx_len = 0;

    while(1) {
        bool ok = receive_packet(&rx_type, &rx_pos, (uint8_t*)activation_vector, vector_bytes, &rx_len, 5000);
        if (!ok) {
            continue;
        }

        if (rx_type == PKT_PING) {
            send_packet(PKT_PING, 0, NULL, 0);
            Serial.println("[Node 2] Cluster ping received & forwarded to Node 3!");
            continue;
        }

        if (rx_type == PKT_ACTIVATION && rx_len == vector_bytes) {
            compute_layers(activation_vector, rx_pos);
            send_packet(PKT_ACTIVATION, rx_pos, (const uint8_t*)activation_vector, vector_bytes);
        }
    }
}

#elif NODE_ID == 3
static int sample_token_t(float* logits, int vocab_size, float temp) {
    logits[0] = -1e9f; // Never output <unk>

    if (temp <= 0.05f) {
        int best = 1;
        float max_v = logits[1];
        for (int i = 2; i < vocab_size; i++) {
            if (logits[i] > max_v) {
                max_v = logits[i];
                best = i;
            }
        }
        return best;
    }

    float max_v = logits[1];
    for (int i = 2; i < vocab_size; i++) {
        if (logits[i] > max_v) max_v = logits[i];
    }
    float sum = 0.0f;
    for (int i = 1; i < vocab_size; i++) {
        logits[i] = expf((logits[i] - max_v) / temp);
        sum += logits[i];
    }
    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float acc = 0.0f;
    for (int i = 1; i < vocab_size; i++) {
        acc += logits[i];
        if (acc >= r) return i;
    }
    return vocab_size - 1;
}

void tail_node_task(void *pvParameters) {
    uart_flush_input(UART_NUM);
    Serial.println("[Node 3] Online! Listening for ring packets...");
    int vector_bytes = config.dim * sizeof(float);
    float* logits = (float*)heap_caps_malloc(config.vocab_size * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    uint8_t rx_type = 0;
    uint8_t rx_pos = 0;
    uint16_t rx_len = 0;

    while(1) {
        bool ok = receive_packet(&rx_type, &rx_pos, (uint8_t*)activation_vector, vector_bytes, &rx_len, 5000);
        if (!ok) {
            continue;
        }

        if (rx_type == PKT_PING) {
            send_packet(PKT_PING, 0, NULL, 0);
            Serial.println("[Node 3] Cluster ping received & forwarded to Node 1!");
            continue;
        }

        if (rx_type == PKT_ACTIVATION && rx_len == vector_bytes) {
            compute_layers(activation_vector, rx_pos);
            rmsnorm(activation_vector, activation_vector, weights.rms_final_weight, config.dim);
            matmul_q(logits, activation_vector, weights.wcls, weights.wcls_scales, config.dim, config.vocab_size);

            int next_token_id = sample_token_t(logits, config.vocab_size, 0.5f);

            send_packet(PKT_TOKEN, rx_pos, (const uint8_t*)&next_token_id, sizeof(int));
        }
    }
}
#endif

void setup() {
    Serial.begin(115200);
    delay(2000); // 2 second USB-CDC discovery delay

    Serial.println("\n\n========================================================");
    Serial.printf("     ESP32-S3 NODE %d INITIALIZING\n", NODE_ID);
    Serial.println("========================================================");

    init_uart_ring();
    load_model();

#if NODE_ID == 1
    xTaskCreatePinnedToCore(head_node_task, "head_task", 8192, NULL, 5, NULL, 1);
#elif NODE_ID == 2
    xTaskCreatePinnedToCore(body_node_task, "body_task", 8192, NULL, 5, NULL, 1);
#elif NODE_ID == 3
    xTaskCreatePinnedToCore(tail_node_task, "tail_task", 8192, NULL, 5, NULL, 1);
#endif
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}

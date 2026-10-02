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
    int8_t* token_embedding_table;
    float* rms_final_weight;
    float* wcls_scales;
    int8_t* wcls;
    float* freq_cis_real;
    float* freq_cis_imag;
} GlobalWeights;

typedef struct {
    float* rms_att_weight[32];
    float* wq_scales[32]; int8_t* wq[32];
    float* wk_scales[32]; int8_t* wk[32];
    float* wv_scales[32]; int8_t* wv[32];
    float* wo_scales[32]; int8_t* wo[32];
    float* rms_ffn_weight[32];
    float* w1_scales[32]; int8_t* w1[32];
    float* w2_scales[32]; int8_t* w2[32];
    float* w3_scales[32]; int8_t* w3[32];
} LayerWeights;

Config config;
GlobalWeights weights;
LayerWeights layer_weights;

float* activation_vector;
int8_t* xq_buffer; // For quantized inputs
float* xb;
float* xb2;
float* hb;
float* hb2;
float* q;
float* k;
float* v;

void init_uart_ring() {
    uart_config_t uart_config = {
        .baud_rate = UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM, 4096 * 2, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_LOGI(TAG, "UART2 Ring Initialized at %d baud", UART_BAUD_RATE);
}

void load_model_mmap() {
    ESP_LOGI(TAG, "Mapping model partition...");
    const esp_partition_t* part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!part) {
        ESP_LOGE(TAG, "Model partition not found!");
        return;
    }
    
    const void* map_ptr;
    esp_partition_mmap_handle_t map_handle;
    esp_err_t err = esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &map_ptr, &map_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mmap model partition: %s", esp_err_to_name(err));
        return;
    }
    
    uint8_t* ptr = (uint8_t*)map_ptr;
    memcpy(&config, ptr, sizeof(Config));
    ptr += 256; // Skip header
    
    ESP_LOGI(TAG, "Config: dim=%d, hidden=%d, layers=%d, heads=%d, vocab=%d", 
             config.dim, config.hidden_dim, config.n_layers, config.n_heads, config.vocab_size);

    int head_size = config.dim / config.n_heads;
    int kv_dim = config.n_kv_heads * head_size;
    
    if (NODE_ID == 1) {
        weights.token_embedding_scales = (float*)ptr; ptr += config.vocab_size * sizeof(float);
        weights.token_embedding_table = (int8_t*)ptr; ptr += config.vocab_size * config.dim;
    }
    
    for (int i = 0; i < config.n_layers; i++) {
        layer_weights.rms_att_weight[i] = (float*)ptr; ptr += config.dim * sizeof(float);
        layer_weights.wq_scales[i] = (float*)ptr; ptr += config.dim * sizeof(float);
        layer_weights.wq[i] = (int8_t*)ptr; ptr += config.dim * config.dim;
        
        layer_weights.wk_scales[i] = (float*)ptr; ptr += kv_dim * sizeof(float);
        layer_weights.wk[i] = (int8_t*)ptr; ptr += kv_dim * config.dim;
        
        layer_weights.wv_scales[i] = (float*)ptr; ptr += kv_dim * sizeof(float);
        layer_weights.wv[i] = (int8_t*)ptr; ptr += kv_dim * config.dim;
        
        layer_weights.wo_scales[i] = (float*)ptr; ptr += config.dim * sizeof(float);
        layer_weights.wo[i] = (int8_t*)ptr; ptr += config.dim * config.dim;
        
        layer_weights.rms_ffn_weight[i] = (float*)ptr; ptr += config.dim * sizeof(float);
        
        layer_weights.w1_scales[i] = (float*)ptr; ptr += config.hidden_dim * sizeof(float);
        layer_weights.w1[i] = (int8_t*)ptr; ptr += config.hidden_dim * config.dim;
        
        layer_weights.w2_scales[i] = (float*)ptr; ptr += config.dim * sizeof(float);
        layer_weights.w2[i] = (int8_t*)ptr; ptr += config.dim * config.hidden_dim;
        
        layer_weights.w3_scales[i] = (float*)ptr; ptr += config.hidden_dim * sizeof(float);
        layer_weights.w3[i] = (int8_t*)ptr; ptr += config.hidden_dim * config.dim;
    }
    
    if (NODE_ID == 3) {
        weights.rms_final_weight = (float*)ptr; ptr += config.dim * sizeof(float);
        weights.wcls_scales = (float*)ptr; ptr += config.vocab_size * sizeof(float);
        weights.wcls = (int8_t*)ptr; ptr += config.vocab_size * config.dim;
    }
    
    weights.freq_cis_real = (float*)ptr; ptr += config.seq_len * (head_size / 2) * sizeof(float);
    weights.freq_cis_imag = (float*)ptr; ptr += config.seq_len * (head_size / 2) * sizeof(float);

    activation_vector = (float*)heap_caps_malloc(config.dim * sizeof(float), MALLOC_CAP_8BIT);
    xb = (float*)heap_caps_malloc(config.dim * sizeof(float), MALLOC_CAP_8BIT);
    xb2 = (float*)heap_caps_malloc(config.dim * sizeof(float), MALLOC_CAP_8BIT);
    hb = (float*)heap_caps_malloc(config.hidden_dim * sizeof(float), MALLOC_CAP_8BIT);
    hb2 = (float*)heap_caps_malloc(config.hidden_dim * sizeof(float), MALLOC_CAP_8BIT);
    q = (float*)heap_caps_malloc(config.dim * sizeof(float), MALLOC_CAP_8BIT);
    k = (float*)heap_caps_malloc(config.dim * sizeof(float), MALLOC_CAP_8BIT);
    v = (float*)heap_caps_malloc(config.dim * sizeof(float), MALLOC_CAP_8BIT);
    xq_buffer = (int8_t*)heap_caps_malloc((config.hidden_dim > config.dim ? config.hidden_dim : config.dim), MALLOC_CAP_8BIT);
    
    ESP_LOGI(TAG, "Model mapped successfully!");
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void matmul_q(float* xout, float* x, int8_t* w, float* scales, int n, int d) {
    float max_val = 0.0f;
    for(int i=0; i<n; i++) {
        float f = fabsf(x[i]);
        if(f > max_val) max_val = f;
    }
    float x_scale = max_val / 127.0f;
    if (x_scale == 0.0f) x_scale = 1e-9f;
    float inv_x_scale = 1.0f / x_scale;
    
    for(int i=0; i<n; i++) {
        xq_buffer[i] = (int8_t)roundf(x[i] * inv_x_scale);
    }
    
    for (int i = 0; i < d; i++) {
        int32_t val = 0;
        int idx = i * n;
        for (int j = 0; j < n; j++) {
            val += w[idx + j] * xq_buffer[j];
        }
        xout[i] = ((float)val) * (scales[i] * x_scale);
    }
}

void compute_layers(float* x, int pos) {
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int head_size = dim / config.n_heads;
    int kv_dim = config.n_kv_heads * head_size;
    int kv_mul = config.n_heads / config.n_kv_heads;
    
    for (int l = 0; l < config.n_layers; l++) {
        rmsnorm(xb, x, layer_weights.rms_att_weight[l], dim);
        
        matmul_q(q, xb, layer_weights.wq[l], layer_weights.wq_scales[l], dim, dim);
        matmul_q(k, xb, layer_weights.wk[l], layer_weights.wk_scales[l], dim, kv_dim);
        matmul_q(v, xb, layer_weights.wv[l], layer_weights.wv_scales[l], dim, kv_dim);
        
        // RoPE
        for (int i = 0; i < dim; i+=2) {
            float q0 = q[i]; float q1 = q[i+1];
            float fcr = weights.freq_cis_real[pos * (head_size / 2) + (i % head_size) / 2];
            float fci = weights.freq_cis_imag[pos * (head_size / 2) + (i % head_size) / 2];
            q[i]   = q0 * fcr - q1 * fci;
            q[i+1] = q0 * fci + q1 * fcr;
        }
        for (int i = 0; i < kv_dim; i+=2) {
            float k0 = k[i]; float k1 = k[i+1];
            float fcr = weights.freq_cis_real[pos * (head_size / 2) + (i % head_size) / 2];
            float fci = weights.freq_cis_imag[pos * (head_size / 2) + (i % head_size) / 2];
            k[i]   = k0 * fcr - k1 * fci;
            k[i+1] = k0 * fci + k1 * fcr;
        }
        
        // Simplified Attention
        for (int h = 0; h < config.n_heads; h++) {
            float* q_head = q + h * head_size;
            float* k_head = k + (h / kv_mul) * head_size;
            float* v_head = v + (h / kv_mul) * head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q_head[i] * k_head[i];
            }
            score /= sqrtf((float)head_size);
            float val = 1.0f; // Softmax for 1 token is 1.0
            for (int i = 0; i < head_size; i++) {
                xb2[h * head_size + i] = val * v_head[i];
            }
        }
        
        matmul_q(xb, xb2, layer_weights.wo[l], layer_weights.wo_scales[l], dim, dim);
        for (int i = 0; i < dim; i++) x[i] += xb[i];
        
        rmsnorm(xb, x, layer_weights.rms_ffn_weight[l], dim);
        
        matmul_q(hb, xb, layer_weights.w1[l], layer_weights.w1_scales[l], dim, hidden_dim);
        matmul_q(hb2, xb, layer_weights.w3[l], layer_weights.w3_scales[l], dim, hidden_dim);
        
        for (int i = 0; i < hidden_dim; i++) {
            float val = hb[i];
            val *= (1.0f / (1.0f + expf(-val))); // silu
            val *= hb2[i];
            hb[i] = val;
        }
        
        matmul_q(xb, hb, layer_weights.w2[l], layer_weights.w2_scales[l], hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += xb[i];
    }
}

#if NODE_ID == 1
void head_node_task(void *pvParameters) {
    ESP_LOGI(TAG, "--- NODE 1 (HEAD) ONLINE ---");
    int pos = 0;
    int token_id = 1; 
    int vector_bytes = config.dim * sizeof(float);
    
    while(1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        
        float scale = weights.token_embedding_scales[token_id];
        int8_t* row = weights.token_embedding_table + token_id * config.dim;
        for(int i=0; i<config.dim; i++) {
            activation_vector[i] = ((float)row[i]) * scale;
        }
        
        compute_layers(activation_vector, pos);
        uart_write_bytes(UART_NUM, (const char*)activation_vector, vector_bytes);
        
        int next_token_id = -1;
        int len = uart_read_bytes(UART_NUM, &next_token_id, sizeof(int), portMAX_DELAY);
        
        if (len == sizeof(int)) {
            ESP_LOGI(TAG, "[Node 1] Sampled Token ID: %d", next_token_id);
            token_id = next_token_id;
            pos++;
            if (pos >= config.seq_len) pos = 0;
        }
    }
}

#elif NODE_ID == 2
void body_node_task(void *pvParameters) {
    ESP_LOGI(TAG, "--- NODE 2 (BODY) ONLINE ---");
    int vector_bytes = config.dim * sizeof(float);
    int pos = 0;
    
    while(1) {
        int len = uart_read_bytes(UART_NUM, activation_vector, vector_bytes, portMAX_DELAY);
        if (len == vector_bytes) {
            compute_layers(activation_vector, pos);
            uart_write_bytes(UART_NUM, (const char*)activation_vector, vector_bytes);
            pos++;
            if (pos >= config.seq_len) pos = 0;
        }
    }
}

#elif NODE_ID == 3
void tail_node_task(void *pvParameters) {
    ESP_LOGI(TAG, "--- NODE 3 (TAIL) ONLINE ---");
    int vector_bytes = config.dim * sizeof(float);
    int pos = 0;
    
    float* logits = (float*)heap_caps_malloc(config.vocab_size * sizeof(float), MALLOC_CAP_8BIT);
    
    while(1) {
        int len = uart_read_bytes(UART_NUM, activation_vector, vector_bytes, portMAX_DELAY);
        if (len == vector_bytes) {
            compute_layers(activation_vector, pos);
            rmsnorm(activation_vector, activation_vector, weights.rms_final_weight, config.dim);
            
            matmul_q(logits, activation_vector, weights.wcls, weights.wcls_scales, config.dim, config.vocab_size);
            
            int next_token_id = 0;
            float max_val = logits[0];
            for(int i=1; i < config.vocab_size; i++) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    next_token_id = i;
                }
            }
            
            ESP_LOGI(TAG, "[Node 3] Sampled token ID %d", next_token_id);
            uart_write_bytes(UART_NUM, (const char*)&next_token_id, sizeof(int));
            pos++;
            if (pos >= config.seq_len) pos = 0;
        }
    }
}
#endif

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Chained Esp32S3 tiny LLM Cluster initializing...");
    init_uart_ring();
    load_model_mmap();
    
#if NODE_ID == 1
    xTaskCreatePinnedToCore(head_node_task, "head_task", 8192, NULL, 5, NULL, 1);
#elif NODE_ID == 2
    xTaskCreatePinnedToCore(body_node_task, "body_task", 8192, NULL, 5, NULL, 1);
#elif NODE_ID == 3
    xTaskCreatePinnedToCore(tail_node_task, "tail_task", 8192, NULL, 5, NULL, 1);
#else
    #error "Invalid NODE_ID specified!"
#endif
}

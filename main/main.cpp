#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_spiffs.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"

static const char *TAG = "CHAINED_ESP32S3_LLM";

// -------------------------------------------------------------
// CLUSTER CONFIGURATION
// -------------------------------------------------------------
// SET NODE_ID to 1, 2, or 3 before compiling and flashing each board.
#define NODE_ID 1

// UART Ring Configuration
#define UART_NUM UART_NUM_2
#define TXD_PIN (GPIO_NUM_17)
#define RXD_PIN (GPIO_NUM_18)
#define UART_BAUD_RATE 921600

// Model Configuration (~40M Params)
#define DIM 512
// Size of activation vector in bytes
#define ACT_VECTOR_BYTES (DIM * sizeof(float))

// -------------------------------------------------------------
// GLOBALS
// -------------------------------------------------------------
float* activation_vector;
float* layer_weights;

// -------------------------------------------------------------
// HARDWARE INIT
// -------------------------------------------------------------
void init_uart_ring() {
    uart_config_t uart_config = {
        .baud_rate = UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    int intr_alloc_flags = 0;

    ESP_ERROR_CHECK(uart_driver_install(UART_NUM, ACT_VECTOR_BYTES * 2, 0, 0, NULL, intr_alloc_flags));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_LOGI(TAG, "UART2 Ring Initialized at %d baud", UART_BAUD_RATE);
}

void init_spiffs() {
    ESP_LOGI(TAG, "Initializing SPIFFS...");
    esp_vfs_spiffs_conf_t conf = {
      .base_path = "/spiffs",
      .partition_label = NULL,
      .max_files = 5,
      .format_if_mount_failed = false
    };
    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mount or format SPIFFS");
        return;
    }
    size_t total = 0, used = 0;
    esp_spiffs_info(NULL, &total, &used);
    ESP_LOGI(TAG, "Partition size: total: %d, used: %d", total, used);
}

void load_model_slice(int node) {
    char filepath[32];
    snprintf(filepath, sizeof(filepath), "/spiffs/node%d.bin", node);
    ESP_LOGI(TAG, "Loading model slice from %s into PSRAM...", filepath);
    
    FILE* f = fopen(filepath, "rb");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to open %s. Did you upload it via esptool?", filepath);
        return;
    }
    
    // Allocate weight buffer in PSRAM
    // E.g., for ~13.5M params, ~27MB if FP16, or 13.5MB if INT8.
    // ESP32S3 N16R8 has 8MB PSRAM, so we assume INT8 quantized weights (~13.5MB / 3 nodes = 4.5MB per node).
    size_t slice_size = 4.5 * 1024 * 1024; // Example size
    layer_weights = (float*)heap_caps_malloc(slice_size, MALLOC_CAP_SPIRAM);
    
    if (layer_weights == NULL) {
        ESP_LOGE(TAG, "Failed to allocate %d bytes in PSRAM for weights", slice_size);
        fclose(f);
        return;
    }
    
    fread(layer_weights, 1, slice_size, f);
    fclose(f);
    ESP_LOGI(TAG, "Model slice loaded into PSRAM.");
}

// -------------------------------------------------------------
// DUMMY TRANSFORMER MATH (Replace with llama2.c functions)
// -------------------------------------------------------------
void compute_layers(int start_layer, int end_layer, float* act_vec) {
    // Vector operations placeholder
    // In actual implementation: run QKV matmuls, RoPE, Attention, SwiGLU, etc.
    for (int i = 0; i < DIM; i++) {
        act_vec[i] += 0.01f; // Dummy transform
    }
}

// -------------------------------------------------------------
// NODE LOGIC
// -------------------------------------------------------------

#if NODE_ID == 1

// HEAD NODE
void head_node_task(void *pvParameters) {
    ESP_LOGI(TAG, "--- NODE 1 (HEAD) ONLINE ---");
    // Placeholder for NimBLE init
    ESP_LOGI(TAG, "Waiting for BLE prompt...");
    
    // Allocate activation vector in fast internal RAM if possible, or PSRAM
    activation_vector = (float*)heap_caps_malloc(ACT_VECTOR_BYTES, MALLOC_CAP_8BIT);
    
    while(1) {
        // 1. Await input (simulate here)
        vTaskDelay(pdMS_TO_TICKS(5000));
        ESP_LOGI(TAG, "Prompt received: 'I want a Crunchwrap Supreme with extra beans.'");
        
        // 2. Tokenize & Initial Embeddings
        ESP_LOGI(TAG, "[Node 1] Generating embeddings...");
        memset(activation_vector, 0, ACT_VECTOR_BYTES); // Dummy embed
        
        // 3. Process Layers 1..X
        compute_layers(0, 2, activation_vector);
        
        // 4. Stream activation vector to Node 2 via UART TX
        ESP_LOGI(TAG, "[Node 1] Transmitting activation vector to Node 2...");
        uart_write_bytes(UART_NUM, (const char*)activation_vector, ACT_VECTOR_BYTES);
        
        // 5. Wait for token ID from Node 3 via UART RX
        int next_token_id = -1;
        ESP_LOGI(TAG, "[Node 1] Waiting for token ID from Node 3...");
        int len = uart_read_bytes(UART_NUM, &next_token_id, sizeof(int), pdMS_TO_TICKS(1000));
        
        if (len == sizeof(int)) {
            ESP_LOGI(TAG, "[Node 1] Sampled Token ID: %d. Output: 'You absolute DONUT...'", next_token_id);
            // In real app, append token to context and loop until EOS
        } else {
            ESP_LOGW(TAG, "[Node 1] Timeout waiting for token from Tail.");
        }
    }
}

#elif NODE_ID == 2

// BODY NODE
void body_node_task(void *pvParameters) {
    ESP_LOGI(TAG, "--- NODE 2 (BODY) ONLINE ---");
    activation_vector = (float*)heap_caps_malloc(ACT_VECTOR_BYTES, MALLOC_CAP_8BIT);
    
    while(1) {
        // 1. Await activation vector from Node 1 via UART RX
        int len = uart_read_bytes(UART_NUM, activation_vector, ACT_VECTOR_BYTES, portMAX_DELAY);
        
        if (len == ACT_VECTOR_BYTES) {
            ESP_LOGI(TAG, "[Node 2] Received vector. Computing middle layers...");
            
            // 2. Process middle layers
            compute_layers(3, 5, activation_vector);
            
            // 3. Transmit vector to Node 3 via UART TX
            ESP_LOGI(TAG, "[Node 2] Transmitting updated vector to Node 3...");
            uart_write_bytes(UART_NUM, (const char*)activation_vector, ACT_VECTOR_BYTES);
        }
    }
}

#elif NODE_ID == 3

// TAIL NODE
void tail_node_task(void *pvParameters) {
    ESP_LOGI(TAG, "--- NODE 3 (TAIL) ONLINE ---");
    activation_vector = (float*)heap_caps_malloc(ACT_VECTOR_BYTES, MALLOC_CAP_8BIT);
    
    while(1) {
        // 1. Await activation vector from Node 2 via UART RX
        int len = uart_read_bytes(UART_NUM, activation_vector, ACT_VECTOR_BYTES, portMAX_DELAY);
        
        if (len == ACT_VECTOR_BYTES) {
            ESP_LOGI(TAG, "[Node 3] Received vector. Computing final layers...");
            
            // 2. Process final layers
            compute_layers(6, 7, activation_vector);
            
            // 3. Calculate logits & sample token
            ESP_LOGI(TAG, "[Node 3] Calculating logits & sampling...");
            int sampled_token = 404; // Dummy token ID
            
            // 4. Send token ID back to Node 1 via UART TX
            ESP_LOGI(TAG, "[Node 3] Transmitting token ID %d to Node 1...", sampled_token);
            uart_write_bytes(UART_NUM, (const char*)&sampled_token, sizeof(int));
        }
    }
}

#endif

// -------------------------------------------------------------
// ENTRY POINT
// -------------------------------------------------------------
extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Chained Esp32S3 tiny LLM Cluster initializing...");
    
    // 1. Init UART Communication Ring
    init_uart_ring();
    
    // 2. Mount SPIFFS & Load Weights
    init_spiffs();
    load_model_slice(NODE_ID);
    
    // 3. Start specific node task
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

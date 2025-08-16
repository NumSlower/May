#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include <jansson.h>
#include <dirent.h>
#include <errno.h>
#include <curl/curl.h>
#include <sqlite3.h>
#include <openssl/sha.h>

// Constants
#define MAX_TOKENS 1500
#define MAX_TOKEN_LENGTH 128
#define MAX_RESPONSE_LENGTH 2000
#define MAX_MEMORY_ENTRIES 2000
#define MAX_VOCAB_SIZE 5000
#define MAX_KNOWLEDGE_ENTRIES 2500
#define MAX_HISTORY 1500
#define MAX_SUBJECT_LENGTH 64
#define MAX_GRADE_LENGTH 32
#define MAX_RECENT_TOPICS 20
#define MAX_INTENT_LENGTH 64
#define MAX_LOG_ENTRY 1024
#define MAX_CONFIG_KEY 64
#define MAX_CONFIG_VALUE 256
#define MAX_BEAM_WIDTH 5

// Neural Network Definitions
#define EMBEDDING_SIZE 128
#define HIDDEN_SIZE 256
#define MAX_SEQ_LENGTH 30
#define LEARNING_RATE 0.001
#define MOMENTUM 0.9
#define DROPOUT_RATE 0.3
#define GRADIENT_CLIP 5.0
#define NUM_EPOCHS 500
#define BATCH_SIZE 8
#define L2_REG 0.0001
#define ATTENTION_SIZE 64

// Structures
typedef struct {
    float** weights_embed;      // vocab_size x embedding_size
    float** weights_enc_ih;     // embedding_size x hidden_size
    float** weights_enc_hh;     // hidden_size x hidden_size
    float* biases_enc;          // hidden_size
    float** weights_dec_ih;     // embedding_size x hidden_size
    float** weights_dec_hh;     // hidden_size x hidden_size
    float** weights_dec_ho;     // hidden_size x vocab_size
    float* biases_dec;          // hidden_size
    float* biases_out;          // vocab_size
    float** weights_attn;       // hidden_size x attention_size
    float** weights_attn_v;     // attention_size x 1
    int vocab_size;
} Seq2SeqNetwork;

typedef struct {
    char key[128];
    char value[256];
    time_t timestamp;
} MemoryEntry;

typedef struct {
    char word[MAX_TOKEN_LENGTH];
    char synonym[MAX_TOKEN_LENGTH];
    char antonym[MAX_TOKEN_LENGTH];
    int sentiment_score;
    float frequency;
    int index;
} VocabEntry;

typedef struct {
    char subject[MAX_SUBJECT_LENGTH];
    char grade[MAX_GRADE_LENGTH];
    char question_id[64];
    char text[512];
    int difficulty;
    float relevance_score;
    int is_question;
    time_t last_accessed;
} KnowledgeEntry;

typedef struct {
    char intent[MAX_INTENT_LENGTH];
    float confidence;
} IntentResult;

typedef struct {
    MemoryEntry* memory;
    int memory_count;
    VocabEntry* vocabulary;
    int vocab_count;
    KnowledgeEntry* knowledge;
    int knowledge_count;
    char (*conversation_history)[1024];
    int history_count;
    char current_subject[MAX_SUBJECT_LENGTH];
    char current_grade[MAX_GRADE_LENGTH];
    int grade_filter_active;
    char recent_topics[MAX_RECENT_TOPICS][MAX_SUBJECT_LENGTH];
    int recent_topic_count;
    Seq2SeqNetwork nn;
    FILE* log_file;
    sqlite3* db;
    int sentiment_score;
    char config_file[256];
    float learning_rate;
} ConversationContext;

// Training Data
typedef struct {
    const char* input;
    const char* output;
} TrainingPair;

TrainingPair training_data[] = {
    {"hi", "Hello! How can I assist you today?"},
    {"how are you", "I'm doing great, thanks! How about you?"},
    {"what is your name", "I'm May, created by xAI. Nice to meet you!"},
    {"tell me about history", "History is a vast subject! Want to focus on a specific era?"},
    {"bye", "Goodbye! See you next time."},
    {"what’s the weather like", "I can’t check the weather, but I can tell you about climate history!"},
    {"who are you", "I’m May, an AI designed to chat and help with knowledge."},
    {"tell me a joke", "Why don’t skeletons fight each other? They don’t have the guts!"},
    {"what is python", "Python is a versatile programming language known for its simplicity."},
    {"good morning", "Good morning! How’s your day starting?"},
    {"tell me about math", "Math is the study of numbers and patterns—want to dive into algebra or geometry?"},
    {"set subject to science", "Subject set to science."},
    {"fact: The Earth is round", "Fact added: The Earth is round."}
};
#define NUM_TRAINING_PAIRS (sizeof(training_data) / sizeof(training_data[0]))

// CURL callback for web data
struct MemoryStruct {
    char* memory;
    size_t size;
};

static size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    struct MemoryStruct* mem = (struct MemoryStruct*)userp;

    char* ptr = realloc(mem->memory, mem->size + realsize + 1);
    if (!ptr) return 0;

    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;

    return realsize;
}

// Function Declarations
void init_seq2seq_network(Seq2SeqNetwork* nn, int vocab_size);
void free_seq2seq_network(Seq2SeqNetwork* nn);
void encode_sequence(Seq2SeqNetwork* nn, int* input_indices, int input_len, float** encoder_outputs, float* hidden, ConversationContext* context);
void decode_sequence_beam(Seq2SeqNetwork* nn, float** encoder_outputs, int input_len, float* hidden, char* output, int max_len, ConversationContext* context);
void train_seq2seq_network(Seq2SeqNetwork* nn, ConversationContext* context);
int word_to_index(ConversationContext* context, const char* word);
const char* index_to_word(ConversationContext* context, int index);
void init_conversation_context(ConversationContext* context);
void free_conversation_context(ConversationContext* context);
int tokenize(const char* input, char (*tokens)[MAX_TOKEN_LENGTH], int max_tokens);
int contains_word(char (*tokens)[MAX_TOKEN_LENGTH], int num_tokens, const char* word);
void detect_intent(char (*tokens)[MAX_TOKEN_LENGTH], int num_tokens, IntentResult* result);
void add_to_memory(ConversationContext* context, const char* key, const char* value);
const char* get_from_memory(ConversationContext* context, const char* key);
int init_database(ConversationContext* context);
int load_vocabulary_from_db(ConversationContext* context);
int load_knowledge_from_db(ConversationContext* context);
int fetch_external_knowledge(ConversationContext* context, const char* query, char* response, int max_length);
void set_context_subject_grade(ConversationContext* context, const char* subject, const char* grade);
void generate_enhanced_response(const char* input, char* out_response, int max_length, ConversationContext* context);
void add_recent_topic(ConversationContext* context, const char* topic);
void log_message(ConversationContext* context, const char* message);
void apply_dropout(float* layer, int size, float rate);
float compute_sentiment(ConversationContext* context, const char* input);
void load_config(ConversationContext* context);
void save_config(ConversationContext* context);
void update_frequency(ConversationContext* context, const char* word);
void clean_text(char* text);
int validate_input(const char* input);
void handle_error(ConversationContext* context, const char* error_msg);
void layer_normalize(float* layer, int size);

// Utility Functions
float relu(float x) { return x > 0 ? x : 0; }
float random_float(float min, float max) { return min + (float)rand() / RAND_MAX * (max - min); }

void softmax(float* logits, float* probs, int size) {
    float max_logit = logits[0];
    for (int i = 1; i < size; i++) if (logits[i] > max_logit) max_logit = logits[i];
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        probs[i] = exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < size; i++) probs[i] /= sum;
}

void layer_normalize(float* layer, int size) {
    float mean = 0.0, variance = 0.0;
    for (int i = 0; i < size; i++) mean += layer[i];
    mean /= size;
    for (int i = 0; i < size; i++) variance += (layer[i] - mean) * (layer[i] - mean);
    variance = sqrt(variance / size + 1e-5);
    for (int i = 0; i < size; i++) layer[i] = (layer[i] - mean) / variance;
}

void apply_dropout(float* layer, int size, float rate) {
    for (int i = 0; i < size; i++) if (random_float(0, 1) < rate) layer[i] = 0;
}

void log_message(ConversationContext* context, const char* message) {
    if (!context || !context->log_file) return;
    time_t now;
    time(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    fprintf(context->log_file, "[%s] %s\n", timestamp, message);
    fflush(context->log_file);
}

void init_seq2seq_network(Seq2SeqNetwork* nn, int vocab_size) {
    if (!nn) return;
    nn->vocab_size = vocab_size;

    float scale_embed = sqrt(6.0 / (vocab_size + EMBEDDING_SIZE));
    float scale_hidden = sqrt(6.0 / (EMBEDDING_SIZE + HIDDEN_SIZE));
    float scale_out = sqrt(6.0 / (HIDDEN_SIZE + vocab_size));
    float scale_attn = sqrt(6.0 / (HIDDEN_SIZE + ATTENTION_SIZE));

    nn->weights_embed = (float**)malloc(vocab_size * sizeof(float*));
    for (int i = 0; i < vocab_size; i++) {
        nn->weights_embed[i] = (float*)calloc(EMBEDDING_SIZE, sizeof(float));
        for (int j = 0; j < EMBEDDING_SIZE; j++) nn->weights_embed[i][j] = random_float(-scale_embed, scale_embed);
    }

    nn->weights_enc_ih = (float**)malloc(EMBEDDING_SIZE * sizeof(float*));
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        nn->weights_enc_ih[i] = (float*)calloc(HIDDEN_SIZE, sizeof(float));
        for (int j = 0; j < HIDDEN_SIZE; j++) nn->weights_enc_ih[i][j] = random_float(-scale_hidden, scale_hidden);
    }
    nn->weights_enc_hh = (float**)malloc(HIDDEN_SIZE * sizeof(float*));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->weights_enc_hh[i] = (float*)calloc(HIDDEN_SIZE, sizeof(float));
        for (int j = 0; j < HIDDEN_SIZE; j++) nn->weights_enc_hh[i][j] = random_float(-scale_hidden, scale_hidden);
    }
    nn->biases_enc = (float*)calloc(HIDDEN_SIZE, sizeof(float));

    nn->weights_dec_ih = (float**)malloc(EMBEDDING_SIZE * sizeof(float*));
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        nn->weights_dec_ih[i] = (float*)calloc(HIDDEN_SIZE, sizeof(float));
        for (int j = 0; j < HIDDEN_SIZE; j++) nn->weights_dec_ih[i][j] = random_float(-scale_hidden, scale_hidden);
    }
    nn->weights_dec_hh = (float**)malloc(HIDDEN_SIZE * sizeof(float*));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->weights_dec_hh[i] = (float*)calloc(HIDDEN_SIZE, sizeof(float));
        for (int j = 0; j < HIDDEN_SIZE; j++) nn->weights_dec_hh[i][j] = random_float(-scale_hidden, scale_hidden);
    }
    nn->weights_dec_ho = (float**)malloc(HIDDEN_SIZE * sizeof(float*));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->weights_dec_ho[i] = (float*)calloc(vocab_size, sizeof(float));
        for (int j = 0; j < vocab_size; j++) nn->weights_dec_ho[i][j] = random_float(-scale_out, scale_out);
    }
    nn->biases_dec = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    nn->biases_out = (float*)calloc(vocab_size, sizeof(float));

    nn->weights_attn = (float**)malloc(HIDDEN_SIZE * sizeof(float*));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->weights_attn[i] = (float*)calloc(ATTENTION_SIZE, sizeof(float));
        for (int j = 0; j < ATTENTION_SIZE; j++) nn->weights_attn[i][j] = random_float(-scale_attn, scale_attn);
    }
    nn->weights_attn_v = (float**)malloc(ATTENTION_SIZE * sizeof(float*));
    for (int i = 0; i < ATTENTION_SIZE; i++) {
        nn->weights_attn_v[i] = (float*)calloc(1, sizeof(float));
        nn->weights_attn_v[i][0] = random_float(-scale_attn, scale_attn);
    }
}

void free_seq2seq_network(Seq2SeqNetwork* nn) {
    if (!nn) return;
    for (int i = 0; i < nn->vocab_size; i++) free(nn->weights_embed[i]);
    free(nn->weights_embed);
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        free(nn->weights_enc_ih[i]);
        free(nn->weights_dec_ih[i]);
    }
    free(nn->weights_enc_ih);
    free(nn->weights_dec_ih);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        free(nn->weights_enc_hh[i]);
        free(nn->weights_dec_hh[i]);
        free(nn->weights_dec_ho[i]);
        free(nn->weights_attn[i]);
    }
    free(nn->weights_enc_hh);
    free(nn->weights_dec_hh);
    free(nn->weights_dec_ho);
    free(nn->biases_enc);
    free(nn->biases_dec);
    free(nn->biases_out);
    for (int i = 0; i < ATTENTION_SIZE; i++) free(nn->weights_attn_v[i]);
    free(nn->weights_attn_v);
    free(nn->weights_attn);
}

void encode_sequence(Seq2SeqNetwork* nn, int* input_indices, int input_len, float** encoder_outputs, float* hidden, ConversationContext* context) {
    if (!nn || !input_indices || !encoder_outputs || !hidden) {
        log_message(context, "Invalid arguments in encode_sequence");
        return;
    }
    float* input_embed = (float*)calloc(EMBEDDING_SIZE, sizeof(float));
    float* next_hidden = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    if (!input_embed || !next_hidden) {
        log_message(context, "Memory allocation failed in encode_sequence");
        free(input_embed);
        free(next_hidden);
        return;
    }

    memset(hidden, 0, HIDDEN_SIZE * sizeof(float));
    for (int t = 0; t < input_len; t++) {
        int token_idx = input_indices[t];
        if (token_idx >= nn->vocab_size || token_idx < 0) token_idx = 2; // <UNK>
        for (int i = 0; i < EMBEDDING_SIZE; i++) input_embed[i] = nn->weights_embed[token_idx][i];
        apply_dropout(input_embed, EMBEDDING_SIZE, DROPOUT_RATE);
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            next_hidden[i] = nn->biases_enc[i];
            for (int j = 0; j < EMBEDDING_SIZE; j++) next_hidden[i] += nn->weights_enc_ih[j][i] * input_embed[j];
            for (int j = 0; j < HIDDEN_SIZE; j++) next_hidden[i] += nn->weights_enc_hh[j][i] * hidden[j];
            next_hidden[i] = tanh(next_hidden[i]);
        }
        layer_normalize(next_hidden, HIDDEN_SIZE);
        memcpy(encoder_outputs[t], next_hidden, HIDDEN_SIZE * sizeof(float));
        memcpy(hidden, next_hidden, HIDDEN_SIZE * sizeof(float));
    }
    free(input_embed);
    free(next_hidden);
}

void compute_attention(Seq2SeqNetwork* nn, float** encoder_outputs, int input_len, float* hidden, float* context_vector) {
    if (!nn || !encoder_outputs || !hidden || !context_vector) return;
    float* attn_scores = (float*)calloc(input_len, sizeof(float));
    float* attn_weights = (float*)calloc(input_len, sizeof(float));
    float* temp = (float*)calloc(ATTENTION_SIZE, sizeof(float));
    if (!attn_scores || !attn_weights || !temp) {
        free(attn_scores);
        free(attn_weights);
        free(temp);
        return;
    }

    for (int t = 0; t < input_len; t++) {
        for (int i = 0; i < ATTENTION_SIZE; i++) {
            temp[i] = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                temp[i] += encoder_outputs[t][j] * nn->weights_attn[j][i];
            }
        }
        attn_scores[t] = 0;
        for (int i = 0; i < ATTENTION_SIZE; i++) {
            attn_scores[t] += temp[i] * nn->weights_attn_v[i][0];
        }
    }
    softmax(attn_scores, attn_weights, input_len);

    memset(context_vector, 0, HIDDEN_SIZE * sizeof(float));
    for (int t = 0; t < input_len; t++) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            context_vector[i] += attn_weights[t] * encoder_outputs[t][i];
        }
    }

    free(attn_scores);
    free(attn_weights);
    free(temp);
}

void decode_sequence_beam(Seq2SeqNetwork* nn, float** encoder_outputs, int input_len, float* hidden, char* output, int max_len, ConversationContext* context) {
    if (!nn || !encoder_outputs || !hidden || !output || !context) {
        log_message(context, "Invalid arguments in decode_sequence_beam");
        snprintf(output, max_len, "Error: Unable to generate response.");
        return;
    }

    typedef struct {
        float* hidden_state;
        char sequence[1024];
        float score;
    } BeamState;

    BeamState* beams = (BeamState*)calloc(MAX_BEAM_WIDTH, sizeof(BeamState));
    if (!beams) {
        log_message(context, "Memory allocation failed for beams");
        snprintf(output, max_len, "Error: Memory allocation failed.");
        return;
    }

    float* input_embed = (float*)calloc(EMBEDDING_SIZE, sizeof(float));
    float* next_hidden = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    float* logits = (float*)calloc(nn->vocab_size, sizeof(float));
    float* probs = (float*)calloc(nn->vocab_size, sizeof(float));
    float* context_vector = (float*)calloc(HIDDEN_SIZE, sizeof(float));

    if (!input_embed || !next_hidden || !logits || !probs || !context_vector) {
        log_message(context, "Memory allocation failed for decode buffers");
        snprintf(output, max_len, "Error: Memory allocation failed.");
        goto cleanup;
    }

    for (int b = 0; b < MAX_BEAM_WIDTH; b++) {
        beams[b].hidden_state = (float*)calloc(HIDDEN_SIZE, sizeof(float));
        if (!beams[b].hidden_state) {
            log_message(context, "Memory allocation failed for beam hidden state");
            snprintf(output, max_len, "Error: Memory allocation failed.");
            goto cleanup;
        }
        memcpy(beams[b].hidden_state, hidden, HIDDEN_SIZE * sizeof(float));
        beams[b].sequence[0] = '\0';
        beams[b].score = 0.0;
    }

    int current_token = 0; // <SOS>
    int output_len = 0;

    while (output_len < max_len / MAX_TOKEN_LENGTH && output_len < MAX_SEQ_LENGTH) {
        BeamState* next_beams = (BeamState*)calloc(MAX_BEAM_WIDTH * nn->vocab_size, sizeof(BeamState));
        if (!next_beams) {
            log_message(context, "Memory allocation failed for next_beams");
            snprintf(output, max_len, "Error: Memory allocation failed.");
            goto cleanup;
        }
        int next_beam_count = 0;

        for (int b = 0; b < MAX_BEAM_WIDTH; b++) {
            if (strlen(beams[b].sequence) >= max_len - MAX_TOKEN_LENGTH) continue;

            for (int i = 0; i < EMBEDDING_SIZE; i++) input_embed[i] = nn->weights_embed[current_token][i];
            compute_attention(nn, encoder_outputs, input_len, beams[b].hidden_state, context_vector);
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                next_hidden[i] = nn->biases_dec[i];
                for (int j = 0; j < EMBEDDING_SIZE; j++) next_hidden[i] += nn->weights_dec_ih[j][i] * input_embed[j];
                for (int j = 0; j < HIDDEN_SIZE; j++) next_hidden[i] += nn->weights_dec_hh[j][i] * beams[b].hidden_state[j];
                next_hidden[i] += context_vector[i];
                next_hidden[i] = tanh(next_hidden[i]);
            }
            layer_normalize(next_hidden, HIDDEN_SIZE);

            for (int i = 0; i < nn->vocab_size; i++) {
                logits[i] = nn->biases_out[i];
                for (int j = 0; j < HIDDEN_SIZE; j++) logits[i] += nn->weights_dec_ho[j][i] * next_hidden[j];
            }
            softmax(logits, probs, nn->vocab_size);

            for (int i = 0; i < nn->vocab_size && next_beam_count < MAX_BEAM_WIDTH * nn->vocab_size; i++) {
                if (probs[i] < 0.001) continue;
                next_beams[next_beam_count].hidden_state = (float*)calloc(HIDDEN_SIZE, sizeof(float));
                if (!next_beams[next_beam_count].hidden_state) {
                    log_message(context, "Memory allocation failed for next beam state");
                    continue;
                }
                memcpy(next_beams[next_beam_count].hidden_state, next_hidden, HIDDEN_SIZE * sizeof(float));
                strncpy(next_beams[next_beam_count].sequence, beams[b].sequence, sizeof(next_beams[next_beam_count].sequence) - 1);
                next_beams[next_beam_count].sequence[sizeof(next_beams[next_beam_count].sequence) - 1] = '\0';

                const char* word = index_to_word(context, i);
                size_t current_len = strlen(next_beams[next_beam_count].sequence);
                if (current_len > 0 && current_len + strlen(word) + 1 < sizeof(next_beams[next_beam_count].sequence)) {
                    strcat(next_beams[next_beam_count].sequence, " ");
                }
                if (current_len + strlen(word) < sizeof(next_beams[next_beam_count].sequence)) {
                    strcat(next_beams[next_beam_count].sequence, word);
                }
                next_beams[next_beam_count].score = beams[b].score + log(probs[i] + 1e-10);
                next_beam_count++;
            }
        }

        for (int i = 0; i < next_beam_count - 1; i++) {
            for (int j = i + 1; j < next_beam_count; j++) {
                if (next_beams[i].score < next_beams[j].score) {
                    BeamState temp = next_beams[i];
                    next_beams[i] = next_beams[j];
                    next_beams[j] = temp;
                }
            }
        }

        for (int b = 0; b < MAX_BEAM_WIDTH && b < next_beam_count; b++) {
            free(beams[b].hidden_state);
            beams[b] = next_beams[b];
        }
        for (int b = MAX_BEAM_WIDTH; b < next_beam_count; b++) free(next_beams[b].hidden_state);
        free(next_beams);

        output_len++;
        if (strstr(beams[0].sequence, "<EOS>")) break;
        current_token = word_to_index(context, strrchr(beams[0].sequence, ' ') ? strrchr(beams[0].sequence, ' ') + 1 : beams[0].sequence);
    }

    strncpy(output, beams[0].sequence, max_len - 1);
    output[max_len - 1] = '\0';

cleanup:
    for (int b = 0; b < MAX_BEAM_WIDTH; b++) free(beams[b].hidden_state);
    free(beams);
    free(input_embed);
    free(next_hidden);
    free(logits);
    free(probs);
    free(context_vector);
}

void train_seq2seq_network(Seq2SeqNetwork* nn, ConversationContext* context) {
    if (!nn || !context) {
        log_message(context, "Invalid arguments in train_seq2seq_network");
        return;
    }

    float* hidden = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    float** encoder_outputs = (float**)malloc(MAX_SEQ_LENGTH * sizeof(float*));
    for (int i = 0; i < MAX_SEQ_LENGTH; i++) encoder_outputs[i] = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    float* input_embed = (float*)calloc(EMBEDDING_SIZE, sizeof(float));
    float* next_hidden = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    float* logits = (float*)calloc(nn->vocab_size, sizeof(float));
    float* probs = (float*)calloc(nn->vocab_size, sizeof(float));
    float* target = (float*)calloc(nn->vocab_size, sizeof(float));
    float* context_vector = (float*)calloc(HIDDEN_SIZE, sizeof(float));

    if (!hidden || !encoder_outputs || !input_embed || !next_hidden || !logits || !probs || !target || !context_vector) {
        log_message(context, "Memory allocation failed in train_seq2seq_network");
        goto cleanup;
    }

    float best_loss = INFINITY;
    int patience = 20, patience_counter = 0;
    float lr = context->learning_rate;

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float total_loss = 0.0;
        for (int batch = 0; batch < NUM_TRAINING_PAIRS; batch += BATCH_SIZE) {
            int batch_size = (batch + BATCH_SIZE > NUM_TRAINING_PAIRS) ? NUM_TRAINING_PAIRS - batch : BATCH_SIZE;
            for (int b = 0; b < batch_size; b++) {
                int pair_idx = batch + b;
                char (*input_tokens)[MAX_TOKEN_LENGTH] = (char(*)[MAX_TOKEN_LENGTH])calloc(MAX_TOKENS, sizeof(char[MAX_TOKEN_LENGTH]));
                char (*output_tokens)[MAX_TOKEN_LENGTH] = (char(*)[MAX_TOKEN_LENGTH])calloc(MAX_TOKENS, sizeof(char[MAX_TOKEN_LENGTH]));
                if (!input_tokens || !output_tokens) {
                    log_message(context, "Memory allocation failed for tokens in training");
                    free(input_tokens);
                    free(output_tokens);
                    continue;
                }

                int input_len = tokenize(training_data[pair_idx].input, input_tokens, MAX_TOKENS);
                int output_len = tokenize(training_data[pair_idx].output, output_tokens, MAX_TOKENS);

                int* input_indices = (int*)calloc(MAX_SEQ_LENGTH, sizeof(int));
                int* target_indices = (int*)calloc(MAX_SEQ_LENGTH, sizeof(int));
                if (!input_indices || !target_indices) {
                    log_message(context, "Memory allocation failed for indices in training");
                    free(input_tokens);
                    free(output_tokens);
                    free(input_indices);
                    free(target_indices);
                    continue;
                }

                input_len = input_len < MAX_SEQ_LENGTH ? input_len : MAX_SEQ_LENGTH;
                output_len = output_len < MAX_SEQ_LENGTH ? output_len : MAX_SEQ_LENGTH;

                for (int i = 0; i < input_len; i++) input_indices[i] = word_to_index(context, input_tokens[i]);
                for (int i = 0; i < output_len; i++) target_indices[i] = word_to_index(context, output_tokens[i]);
                target_indices[output_len] = 1; // <EOS>

                encode_sequence(nn, input_indices, input_len, encoder_outputs, hidden, context);
                int current_token = 0; // <SOS>
                for (int t = 0; t <= output_len; t++) {
                    for (int i = 0; i < EMBEDDING_SIZE; i++) input_embed[i] = nn->weights_embed[current_token][i];
                    compute_attention(nn, encoder_outputs, input_len, hidden, context_vector);
                    for (int i = 0; i < HIDDEN_SIZE; i++) {
                        next_hidden[i] = nn->biases_dec[i];
                        for (int j = 0; j < EMBEDDING_SIZE; j++) next_hidden[i] += nn->weights_dec_ih[j][i] * input_embed[j];
                        for (int j = 0; j < HIDDEN_SIZE; j++) next_hidden[i] += nn->weights_dec_hh[j][i] * hidden[j];
                        next_hidden[i] += context_vector[i];
                        next_hidden[i] = tanh(next_hidden[i]);
                    }
                    layer_normalize(next_hidden, HIDDEN_SIZE);
                    memcpy(hidden, next_hidden, HIDDEN_SIZE * sizeof(float));

                    for (int i = 0; i < nn->vocab_size; i++) {
                        logits[i] = nn->biases_out[i];
                        for (int j = 0; j < HIDDEN_SIZE; j++) logits[i] += nn->weights_dec_ho[j][i] * hidden[j];
                    }
                    softmax(logits, probs, nn->vocab_size);

                    int target_idx = target_indices[t];
                    for (int i = 0; i < nn->vocab_size; i++) {
                        target[i] = (i == target_idx) ? 1.0 : 0.0;
                        float error = -target[i] * log(probs[i] + 1e-10);
                        total_loss += error;
                    }
                    current_token = target_idx; // Teacher forcing
                }

                free(input_tokens);
                free(output_tokens);
                free(input_indices);
                free(target_indices);
            }
        }

        total_loss /= NUM_TRAINING_PAIRS;
        total_loss += L2_REG * (nn->weights_enc_ih[0][0] * nn->weights_enc_ih[0][0]);
        if (total_loss < best_loss) {
            best_loss = total_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
            if (patience_counter >= patience) {
                log_message(context, "Early stopping triggered");
                break;
            }
        }
        lr *= 0.995;
        context->learning_rate = lr;

        char log_msg[256];
        snprintf(log_msg, sizeof(log_msg), "Epoch %d, Loss: %.4f, LR: %.6f", epoch + 1, total_loss, lr);
        log_message(context, log_msg);
        printf("%s\n", log_msg);
    }

cleanup:
    free(hidden);
    for (int i = 0; i < MAX_SEQ_LENGTH; i++) free(encoder_outputs[i]);
    free(encoder_outputs);
    free(input_embed);
    free(next_hidden);
    free(logits);
    free(probs);
    free(target);
    free(context_vector);
}

int word_to_index(ConversationContext* context, const char* word) {
    if (!context || !word || !context->db) return 2; // <UNK>
    sqlite3_stmt* stmt;
    char query[256];
    snprintf(query, sizeof(query), "SELECT idx FROM vocabulary WHERE word = ?");
    if (sqlite3_prepare_v2(context->db, query, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, word, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            int idx = sqlite3_column_int(stmt, 0);
            sqlite3_finalize(stmt);
            update_frequency(context, word);
            return idx;
        }
        sqlite3_finalize(stmt);
    }
    return 2; // <UNK>
}

const char* index_to_word(ConversationContext* context, int index) {
    if (!context || !context->db) return "<UNK>";
    static char word[MAX_TOKEN_LENGTH];
    sqlite3_stmt* stmt;
    char query[256];
    snprintf(query, sizeof(query), "SELECT word FROM vocabulary WHERE idx = ?");
    if (sqlite3_prepare_v2(context->db, query, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, index);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            strncpy(word, (const char*)sqlite3_column_text(stmt, 0), MAX_TOKEN_LENGTH - 1);
            word[MAX_TOKEN_LENGTH - 1] = '\0';
            sqlite3_finalize(stmt);
            return word;
        }
        sqlite3_finalize(stmt);
    }
    return "<UNK>";
}

void update_frequency(ConversationContext* context, const char* word) {
    if (!context || !word || !context->db) return;
    sqlite3_stmt* stmt;
    char query[256];
    snprintf(query, sizeof(query), "UPDATE vocabulary SET frequency = frequency + 0.1 WHERE word = ?");
    if (sqlite3_prepare_v2(context->db, query, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, word, -1, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
}

int init_database(ConversationContext* context) {
    if (!context) return 0;
    if (sqlite3_open("ai_bot.db", &context->db) != SQLITE_OK) {
        log_message(context, "Failed to open SQLite database");
        return 0;
    }

    const char* create_vocab_table = "CREATE TABLE IF NOT EXISTS vocabulary (word TEXT PRIMARY KEY, synonym TEXT, antonym TEXT, sentiment_score INTEGER, frequency REAL, idx INTEGER)";
    const char* create_knowledge_table = "CREATE TABLE IF NOT EXISTS knowledge (subject TEXT, grade TEXT, question_id TEXT, text TEXT, difficulty INTEGER, relevance_score REAL, is_question INTEGER, last_accessed INTEGER)";
    const char* create_history_table = "CREATE TABLE IF NOT EXISTS history (timestamp INTEGER, entry TEXT)";

    char* err_msg = 0;
    if (sqlite3_exec(context->db, create_vocab_table, 0, 0, &err_msg) != SQLITE_OK ||
        sqlite3_exec(context->db, create_knowledge_table, 0, 0, &err_msg) != SQLITE_OK ||
        sqlite3_exec(context->db, create_history_table, 0, 0, &err_msg) != SQLITE_OK) {
        log_message(context, err_msg);
        sqlite3_free(err_msg);
        sqlite3_close(context->db);
        return 0;
    }

    // Insert initial vocabulary
    sqlite3_stmt* stmt;
    const char* insert_vocab = "INSERT OR IGNORE INTO vocabulary (word, sentiment_score, frequency, idx) VALUES (?, 0, 1.0, ?)";
    if (sqlite3_prepare_v2(context->db, insert_vocab, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, "<SOS>", -1, SQLITE_STATIC); sqlite3_bind_int(stmt, 2, 0); sqlite3_step(stmt); sqlite3_reset(stmt);
        sqlite3_bind_text(stmt, 1, "<EOS>", -1, SQLITE_STATIC); sqlite3_bind_int(stmt, 2, 1); sqlite3_step(stmt); sqlite3_reset(stmt);
        sqlite3_bind_text(stmt, 1, "<UNK>", -1, SQLITE_STATIC); sqlite3_bind_int(stmt, 2, 2); sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    return 1;
}

void init_conversation_context(ConversationContext* context) {
    if (!context) return;
    context->memory = (MemoryEntry*)calloc(MAX_MEMORY_ENTRIES, sizeof(MemoryEntry));
    context->vocabulary = (VocabEntry*)calloc(MAX_VOCAB_SIZE, sizeof(VocabEntry));
    context->knowledge = (KnowledgeEntry*)calloc(MAX_KNOWLEDGE_ENTRIES, sizeof(KnowledgeEntry));
    context->conversation_history = (char(*)[1024])calloc(MAX_HISTORY, sizeof(char[1024]));
    if (!context->memory || !context->vocabulary || !context->knowledge || !context->conversation_history) {
        log_message(context, "Failed to allocate memory for context");
        exit(1);
    }

    context->memory_count = 0;
    context->vocab_count = 0;
    context->knowledge_count = 0;
    context->history_count = 0;
    context->grade_filter_active = 0;
    context->recent_topic_count = 0;
    context->sentiment_score = 0;
    context->learning_rate = LEARNING_RATE;
    strcpy(context->current_subject, "general");
    strcpy(context->current_grade, "all");
    strcpy(context->config_file, "ai_config.json");

    context->log_file = fopen("ai_bot.log", "a");
    if (!context->log_file) perror("Failed to open log file");

    if (!init_database(context)) {
        log_message(context, "Database initialization failed");
        exit(1);
    }

    for (int i = 0; i < NUM_TRAINING_PAIRS; i++) {
        char (*tokens)[MAX_TOKEN_LENGTH] = (char(*)[MAX_TOKEN_LENGTH])calloc(MAX_TOKENS, sizeof(char[MAX_TOKEN_LENGTH]));
        int num_tokens = tokenize(training_data[i].input, tokens, MAX_TOKENS);
        sqlite3_stmt* stmt;
        const char* insert = "INSERT OR IGNORE INTO vocabulary (word, sentiment_score, frequency, idx) VALUES (?, 0, 1.0, ?)";
        if (sqlite3_prepare_v2(context->db, insert, -1, &stmt, NULL) == SQLITE_OK) {
            for (int j = 0; j < num_tokens; j++) {
                if (word_to_index(context, tokens[j]) == 2 && context->vocab_count < MAX_VOCAB_SIZE) {
                    sqlite3_bind_text(stmt, 1, tokens[j], -1, SQLITE_STATIC);
                    sqlite3_bind_int(stmt, 2, context->vocab_count);
                    sqlite3_step(stmt);
                    sqlite3_reset(stmt);
                    context->vocab_count++;
                }
            }
            sqlite3_finalize(stmt);
        }
        num_tokens = tokenize(training_data[i].output, tokens, MAX_TOKENS);
        if (sqlite3_prepare_v2(context->db, insert, -1, &stmt, NULL) == SQLITE_OK) {
            for (int j = 0; j < num_tokens; j++) {
                if (word_to_index(context, tokens[j]) == 2 && context->vocab_count < MAX_VOCAB_SIZE) {
                    sqlite3_bind_text(stmt, 1, tokens[j], -1, SQLITE_STATIC);
                    sqlite3_bind_int(stmt, 2, context->vocab_count);
                    sqlite3_step(stmt);
                    sqlite3_reset(stmt);
                    context->vocab_count++;
                }
            }
            sqlite3_finalize(stmt);
        }
        free(tokens);
    }

    init_seq2seq_network(&context->nn, context->vocab_count);
    train_seq2seq_network(&context->nn, context);
    load_config(context);
    log_message(context, "Conversation context initialized");
}

void free_conversation_context(ConversationContext* context) {
    if (!context) return;
    free_seq2seq_network(&context->nn);
    free(context->memory);
    free(context->vocabulary);
    free(context->knowledge);
    free(context->conversation_history);
    if (context->log_file) fclose(context->log_file);
    if (context->db) sqlite3_close(context->db);
}

int tokenize(const char* input, char (*tokens)[MAX_TOKEN_LENGTH], int max_tokens) {
    if (!input || !tokens) return 0;
    int num_tokens = 0;
    const char* p = input;
    char buffer[MAX_TOKEN_LENGTH];
    int buf_idx = 0;

    while (*p && num_tokens < max_tokens) {
        while (*p && (isspace((unsigned char)*p) || ispunct((unsigned char)*p))) p++;
        if (!*p) break;
        buf_idx = 0;
        while (*p && !isspace((unsigned char)*p) && !ispunct((unsigned char)*p) && buf_idx < MAX_TOKEN_LENGTH - 1) {
            buffer[buf_idx++] = tolower((unsigned char)*p);
            p++;
        }
        if (buf_idx > 0) {
            buffer[buf_idx] = '\0';
            strcpy(tokens[num_tokens], buffer);
            num_tokens++;
        }
    }
    return num_tokens;
}

int contains_word(char (*tokens)[MAX_TOKEN_LENGTH], int num_tokens, const char* word) {
    if (!tokens || !word) return 0;
    for (int i = 0; i < num_tokens; i++) {
        if (strcmp(tokens[i], word) == 0) return 1;
    }
    return 0;
}

void detect_intent(char (*tokens)[MAX_TOKEN_LENGTH], int num_tokens, IntentResult* result) {
    if (!tokens || !result) {
        strcpy(result->intent, "empty");
        result->confidence = 0.0;
        return;
    }
    result->confidence = 0.0;
    if (num_tokens == 0) {
        strcpy(result->intent, "empty");
        return;
    }

    float weights[] = {1.0, 0.8, 0.6};
    if (contains_word(tokens, num_tokens, "hi") || contains_word(tokens, num_tokens, "hello")) {
        strcpy(result->intent, "greeting");
        result->confidence = weights[0] * 0.95;
    } else if (contains_word(tokens, num_tokens, "bye") || contains_word(tokens, num_tokens, "goodbye")) {
        strcpy(result->intent, "farewell");
        result->confidence = weights[0] * 0.90;
    } else if (contains_word(tokens, num_tokens, "help")) {
        strcpy(result->intent, "help");
        result->confidence = weights[0] * 0.85;
    } else if (contains_word(tokens, num_tokens, "what") && contains_word(tokens, num_tokens, "are") && contains_word(tokens, num_tokens, "you")) {
        strcpy(result->intent, "self_query");
        result->confidence = weights[0] * 0.90;
    } else if (contains_word(tokens, num_tokens, "what") || contains_word(tokens, num_tokens, "how") || contains_word(tokens, num_tokens, "why")) {
        strcpy(result->intent, "question");
        result->confidence = weights[1] * 0.80;
    } else if (num_tokens >= 3 && strcmp(tokens[0], "my") == 0 && strcmp(tokens[1], "name") == 0) {
        strcpy(result->intent, "memory_set_name");
        result->confidence = weights[0] * 0.95;
    } else if (num_tokens >= 2 && strcmp(tokens[0], "i") == 0 && strcmp(tokens[1], "like") == 0) {
        strcpy(result->intent, "memory_set_like");
        result->confidence = weights[0] * 0.90;
    } else if (contains_word(tokens, num_tokens, "tell") && contains_word(tokens, num_tokens, "me") && contains_word(tokens, num_tokens, "about")) {
        strcpy(result->intent, "knowledge_request");
        result->confidence = weights[0] * 0.90;
    } else if (contains_word(tokens, num_tokens, "set") && contains_word(tokens, num_tokens, "subject")) {
        strcpy(result->intent, "set_subject");
        result->confidence = weights[1] * 0.85;
    } else if (contains_word(tokens, num_tokens, "set") && contains_word(tokens, num_tokens, "grade")) {
        strcpy(result->intent, "set_grade");
        result->confidence = weights[1] * 0.85;
    } else if (contains_word(tokens, num_tokens, "fact")) {
        strcpy(result->intent, "add_fact");
        result->confidence = weights[0] * 0.85;
    } else if (contains_word(tokens, num_tokens, "joke")) {
        strcpy(result->intent, "tell_joke");
        result->confidence = weights[0] * 0.90;
    } else {
        strcpy(result->intent, "statement");
        result->confidence = weights[2] * 0.70;
    }
}

void add_to_memory(ConversationContext* context, const char* key, const char* value) {
    if (!context || !key || !value) return;
    for (int i = 0; i < context->memory_count; i++) {
        if (strcmp(context->memory[i].key, key) == 0) {
            strcpy(context->memory[i].value, value);
            time(&context->memory[i].timestamp);
            return;
        }
    }
    if (context->memory_count < MAX_MEMORY_ENTRIES) {
        strcpy(context->memory[context->memory_count].key, key);
        strcpy(context->memory[context->memory_count].value, value);
        time(&context->memory[context->memory_count].timestamp);
        context->memory_count++;
        char log_msg[512];
        snprintf(log_msg, sizeof(log_msg), "Added to memory: %s = %s", key, value);
        log_message(context, log_msg);
    }
}

const char* get_from_memory(ConversationContext* context, const char* key) {
    if (!context || !key) return NULL;
    for (int i = 0; i < context->memory_count; i++) {
        if (strcmp(context->memory[i].key, key) == 0) return context->memory[i].value;
    }
    return NULL;
}

int load_vocabulary_from_db(ConversationContext* context) {
    if (!context || !context->db) return 0;
    sqlite3_stmt* stmt;
    const char* query = "SELECT word, sentiment_score, frequency, idx FROM vocabulary";
    if (sqlite3_prepare_v2(context->db, query, -1, &stmt, NULL) != SQLITE_OK) {
        log_message(context, "Failed to prepare vocabulary query");
        return 0;
    }

    context->vocab_count = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW && context->vocab_count < MAX_VOCAB_SIZE) {
        strcpy(context->vocabulary[context->vocab_count].word, (const char*)sqlite3_column_text(stmt, 0));
        context->vocabulary[context->vocab_count].sentiment_score = sqlite3_column_int(stmt, 1);
        context->vocabulary[context->vocab_count].frequency = sqlite3_column_double(stmt, 2);
        context->vocabulary[context->vocab_count].index = sqlite3_column_int(stmt, 3);
        context->vocab_count++;
    }
    sqlite3_finalize(stmt);
    return 1;
}

int load_knowledge_from_db(ConversationContext* context) {
    if (!context || !context->db) return 0;
    sqlite3_stmt* stmt;
    const char* query = "SELECT subject, grade, question_id, text, difficulty, relevance_score, is_question, last_accessed FROM knowledge";
    if (sqlite3_prepare_v2(context->db, query, -1, &stmt, NULL) != SQLITE_OK) {
        log_message(context, "Failed to prepare knowledge query");
        return 0;
    }

    context->knowledge_count = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW && context->knowledge_count < MAX_KNOWLEDGE_ENTRIES) {
        strcpy(context->knowledge[context->knowledge_count].subject, (const char*)sqlite3_column_text(stmt, 0));
        strcpy(context->knowledge[context->knowledge_count].grade, (const char*)sqlite3_column_text(stmt, 1));
        strcpy(context->knowledge[context->knowledge_count].question_id, (const char*)sqlite3_column_text(stmt, 2));
        strcpy(context->knowledge[context->knowledge_count].text, (const char*)sqlite3_column_text(stmt, 3));
        context->knowledge[context->knowledge_count].difficulty = sqlite3_column_int(stmt, 4);
        context->knowledge[context->knowledge_count].relevance_score = sqlite3_column_double(stmt, 5);
        context->knowledge[context->knowledge_count].is_question = sqlite3_column_int(stmt, 6);
        context->knowledge[context->knowledge_count].last_accessed = sqlite3_column_int64(stmt, 7);
        context->knowledge_count++;
    }
    sqlite3_finalize(stmt);
    return 1;
}

int fetch_external_knowledge(ConversationContext* context, const char* query, char* response, int max_length) {
    if (!context || !query || !response) return 0;

    CURL* curl = curl_easy_init();
    if (!curl) {
        log_message(context, "Failed to initialize CURL");
        return 0;
    }

    struct MemoryStruct chunk;
    chunk.memory = malloc(1);
    chunk.size = 0;

    char url[512];
    snprintf(url, sizeof(url), "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=%s&format=json", curl_easy_escape(curl, query, 0));

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&chunk);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "AI-Bot/1.0");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        char log_msg[256];
        snprintf(log_msg, sizeof(log_msg), "CURL error: %s", curl_easy_strerror(res));
        log_message(context, log_msg);
        free(chunk.memory);
        curl_easy_cleanup(curl);
        return 0;
    }

    json_error_t error;
    json_t* root = json_loads(chunk.memory, 0, &error);
    free(chunk.memory);
    if (!root) {
        log_message(context, "Failed to parse Wikipedia JSON");
        curl_easy_cleanup(curl);
        return 0;
    }

    json_t* search = json_object_get(json_object_get(root, "query"), "search");
    if (json_is_array(search) && json_array_size(search) > 0) {
        json_t* first_result = json_array_get(search, 0);
        const char* snippet = json_string_value(json_object_get(first_result, "snippet"));
        if (snippet) {
            char cleaned_snippet[512];
            strncpy(cleaned_snippet, snippet, sizeof(cleaned_snippet) - 1);
            cleaned_snippet[sizeof(cleaned_snippet) - 1] = '\0';
            clean_text(cleaned_snippet); // Remove HTML tags
            snprintf(response, max_length, "Here's what I found: %s", cleaned_snippet);
        } else {
            snprintf(response, max_length, "Found something on Wikipedia, but no snippet available.");
        }
    } else {
        snprintf(response, max_length, "No relevant info found on Wikipedia.");
    }

    json_decref(root);
    curl_easy_cleanup(curl);
    return 1;
}

void add_recent_topic(ConversationContext* context, const char* topic) {
    if (!context || !topic || strlen(topic) == 0) return;
    for (int i = 0; i < context->recent_topic_count; i++) {
        if (strcmp(context->recent_topics[i], topic) == 0) return;
    }
    if (context->recent_topic_count < MAX_RECENT_TOPICS) {
        strcpy(context->recent_topics[context->recent_topic_count], topic);
        context->recent_topic_count++;
    } else {
        for (int i = 0; i < MAX_RECENT_TOPICS - 1; i++) strcpy(context->recent_topics[i], context->recent_topics[i + 1]);
        strcpy(context->recent_topics[MAX_RECENT_TOPICS - 1], topic);
    }
}

void set_context_subject_grade(ConversationContext* context, const char* subject, const char* grade) {
    if (!context) return;
    if (subject) {
        strcpy(context->current_subject, subject);
        add_recent_topic(context, subject);
    }
    if (grade) {
        strcpy(context->current_grade, grade);
        context->grade_filter_active = 1;
    }
}

float compute_sentiment(ConversationContext* context, const char* input) {
    if (!context || !input || !context->db) return 0.0;
    char (*tokens)[MAX_TOKEN_LENGTH] = (char(*)[MAX_TOKEN_LENGTH])calloc(MAX_TOKENS, sizeof(char[MAX_TOKEN_LENGTH]));
    int num_tokens = tokenize(input, tokens, MAX_TOKENS);
    float sentiment = 0.0;
    int count = 0;

    sqlite3_stmt* stmt;
    const char* query = "SELECT sentiment_score FROM vocabulary WHERE word = ?";
    if (sqlite3_prepare_v2(context->db, query, -1, &stmt, NULL) == SQLITE_OK) {
        for (int i = 0; i < num_tokens; i++) {
            sqlite3_bind_text(stmt, 1, tokens[i], -1, SQLITE_STATIC);
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                sentiment += sqlite3_column_int(stmt, 0);
                count++;
            }
            sqlite3_reset(stmt);
        }
        sqlite3_finalize(stmt);
    }

    free(tokens);
    return count > 0 ? sentiment / count : 0.0;
}

void clean_text(char* text) {
    if (!text) return;
    char* src = text;
    char* dst = text;
    while (*src) {
        if (isalnum((unsigned char)*src) || isspace((unsigned char)*src) || ispunct((unsigned char)*src)) {
            if (*src != '<' && *src != '>') *dst++ = *src; // Remove HTML tags
        }
        src++;
    }
    *dst = '\0';
}

int validate_input(const char* input) {
    if (!input || strlen(input) == 0 || strlen(input) > 512) return 0;
    return 1;
}

void handle_error(ConversationContext* context, const char* error_msg) {
    if (!context || !error_msg) return;
    log_message(context, error_msg);
    printf("Error: %s\n", error_msg);
}

void generate_enhanced_response(const char* input, char* out_response, int max_length, ConversationContext* context) {
    if (!input || !out_response || !context || !validate_input(input)) {
        snprintf(out_response, max_length, "Sorry, invalid input. Please try again.");
        return;
    }

    char cleaned_input[512];
    strncpy(cleaned_input, input, sizeof(cleaned_input));
    cleaned_input[sizeof(cleaned_input) - 1] = '\0';
    clean_text(cleaned_input);

    char (*tokens)[MAX_TOKEN_LENGTH] = (char(*)[MAX_TOKEN_LENGTH])calloc(MAX_TOKENS, sizeof(char[MAX_TOKEN_LENGTH]));
    if (!tokens) {
        snprintf(out_response, max_length, "Error: Memory allocation failed.");
        log_message(context, "Memory allocation failed for tokens");
        return;
    }
    int num_tokens = tokenize(cleaned_input, tokens, MAX_TOKENS);
    IntentResult intent;
    detect_intent(tokens, num_tokens, &intent);

    char log_msg[512];
    snprintf(log_msg, sizeof(log_msg), "Input: %s, Intent: %s (%.2f)", cleaned_input, intent.intent, intent.confidence);
    log_message(context, log_msg);

    if (strcmp(intent.intent, "greeting") == 0) {
        snprintf(out_response, max_length, "Hello! How can I assist you today?");
    } else {
        int* input_indices = (int*)calloc(MAX_SEQ_LENGTH, sizeof(int));
        if (!input_indices) {
            snprintf(out_response, max_length, "Error: Memory allocation failed.");
            log_message(context, "Memory allocation failed for input_indices");
            free(tokens);
            return;
        }
        int input_len = num_tokens < MAX_SEQ_LENGTH ? num_tokens : MAX_SEQ_LENGTH;
        for (int i = 0; i < input_len; i++) input_indices[i] = word_to_index(context, tokens[i]);

        float* hidden = (float*)calloc(HIDDEN_SIZE, sizeof(float));
        float** encoder_outputs = (float**)calloc(MAX_SEQ_LENGTH, sizeof(float*));
        if (!hidden || !encoder_outputs) {
            snprintf(out_response, max_length, "Error: Memory allocation failed.");
            log_message(context, "Memory allocation failed for hidden or encoder_outputs");
            free(input_indices);
            free(hidden);
            free(encoder_outputs);
            free(tokens);
            return;
        }
        for (int i = 0; i < MAX_SEQ_LENGTH; i++) {
            encoder_outputs[i] = (float*)calloc(HIDDEN_SIZE, sizeof(float));
            if (!encoder_outputs[i]) {
                snprintf(out_response, max_length, "Error: Memory allocation failed.");
                log_message(context, "Memory allocation failed for encoder_outputs[i]");
                for (int j = 0; j < i; j++) free(encoder_outputs[j]);
                free(input_indices);
                free(hidden);
                free(encoder_outputs);
                free(tokens);
                return;
            }
        }

        encode_sequence(&context->nn, input_indices, input_len, encoder_outputs, hidden, context);
        char nn_response[1024] = {0};
        decode_sequence_beam(&context->nn, encoder_outputs, input_len, hidden, nn_response, sizeof(nn_response), context);

        context->sentiment_score = (int)(compute_sentiment(context, cleaned_input) * 10);

        if (strcmp(intent.intent, "farewell") == 0) {
            snprintf(out_response, max_length, "Goodbye! Take care!");
        } else if (strcmp(intent.intent, "help") == 0) {
            snprintf(out_response, max_length, "I can chat, answer questions, or set context. Try 'tell me about history' or 'set subject to math'.");
        } else if (strcmp(intent.intent, "self_query") == 0) {
            snprintf(out_response, max_length, "I’m May, an AI built by xAI to assist and chat with you!");
        } else if (strcmp(intent.intent, "memory_set_name") == 0 && num_tokens >= 4) {
            char name[128];
            strcpy(name, tokens[3]);
            add_to_memory(context, "name", name);
            snprintf(out_response, max_length, "Nice to meet you, %s! How can I help?", name);
        } else if (strcmp(intent.intent, "set_subject") == 0) {
            char subject[MAX_SUBJECT_LENGTH] = "";
            for (int i = 0; i < num_tokens; i++) {
                if (strcmp(tokens[i], "to") == 0 && i + 1 < num_tokens) {
                    strcpy(subject, tokens[i + 1]);
                    break;
                }
            }
            if (strlen(subject) > 0) {
                set_context_subject_grade(context, subject, NULL);
                snprintf(out_response, max_length, "Subject set to %s.", subject);
            } else {
                snprintf(out_response, max_length, "Please specify a subject, e.g., 'set subject to history'.");
            }
        } else if (strcmp(intent.intent, "add_fact") == 0) {
            if (context->knowledge_count < MAX_KNOWLEDGE_ENTRIES) {
                char fact[512] = "";
                for (int i = 1; i < num_tokens; i++) {
                    strcat(fact, tokens[i]);
                    if (i < num_tokens - 1) strcat(fact, " ");
                }
                sqlite3_stmt* stmt;
                const char* insert = "INSERT INTO knowledge (subject, grade, text, relevance_score, is_question, last_accessed) VALUES (?, ?, ?, 1.0, 0, ?)";
                if (sqlite3_prepare_v2(context->db, insert, -1, &stmt, NULL) == SQLITE_OK) {
                    sqlite3_bind_text(stmt, 1, context->current_subject, -1, SQLITE_STATIC);
                    sqlite3_bind_text(stmt, 2, context->current_grade, -1, SQLITE_STATIC);
                    sqlite3_bind_text(stmt, 3, fact, -1, SQLITE_STATIC);
                    sqlite3_bind_int64(stmt, 4, time(NULL));
                    sqlite3_step(stmt);
                    sqlite3_finalize(stmt);
                    context->knowledge_count++;
                }
                snprintf(out_response, max_length, "Fact added to my knowledge base!");
            }
        } else if (strcmp(intent.intent, "tell_joke") == 0) {
            snprintf(out_response, max_length, "Why don’t skeletons fight? They don’t have the guts!");
        } else if (strcmp(intent.intent, "knowledge_request") == 0 || strcmp(intent.intent, "question") == 0) {
            char knowledge_response[512];
            if (fetch_external_knowledge(context, cleaned_input, knowledge_response, sizeof(knowledge_response))) {
                snprintf(out_response, max_length, "%s", knowledge_response);
            } else {
                snprintf(out_response, max_length, "%s", nn_response);
            }
        } else {
            snprintf(out_response, max_length, "%s", nn_response);
        }

        if (context->history_count < MAX_HISTORY) {
            snprintf(context->conversation_history[context->history_count], 1024, "You: %s\nMay: %s", cleaned_input, out_response);
            sqlite3_stmt* stmt;
            const char* insert = "INSERT INTO history (timestamp, entry) VALUES (?, ?)";
            if (sqlite3_prepare_v2(context->db, insert, -1, &stmt, NULL) == SQLITE_OK) {
                sqlite3_bind_int64(stmt, 1, time(NULL));
                sqlite3_bind_text(stmt, 2, context->conversation_history[context->history_count], -1, SQLITE_STATIC);
                sqlite3_step(stmt);
                sqlite3_finalize(stmt);
            }
            context->history_count++;
        }

        free(input_indices);
        free(hidden);
        for (int i = 0; i < MAX_SEQ_LENGTH; i++) free(encoder_outputs[i]);
        free(encoder_outputs);
    }

    free(tokens);
}

void load_config(ConversationContext* context) {
    if (!context) return;
    FILE* file = fopen(context->config_file, "r");
    if (!file) {
        log_message(context, "No config file found, using defaults");
        return;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* json_data = (char*)malloc(length + 1);
    fread(json_data, 1, length, file);
    json_data[length] = '\0';
    fclose(file);

    json_error_t error;
    json_t* root = json_loads(json_data, 0, &error);
    free(json_data);
    if (!root) {
        log_message(context, "Error parsing config JSON");
        return;
    }

    const char* subject = json_string_value(json_object_get(root, "default_subject"));
    if (subject) strcpy(context->current_subject, subject);
    json_decref(root);
}

void save_config(ConversationContext* context) {
    if (!context) return;
    json_t* root = json_object();
    json_object_set_new(root, "default_subject", json_string(context->current_subject));

    FILE* file = fopen(context->config_file, "w");
    if (file) {
        json_dumpf(root, file, JSON_INDENT(2));
        fclose(file);
    }
    json_decref(root);
}

int main() {
    srand(time(NULL));
    curl_global_init(CURL_GLOBAL_ALL);

    ConversationContext context;
    init_conversation_context(&context);

    printf("General Conversational AI Bot\n");
    printf("Hi! I’m May, built by xAI. I can chat, answer questions (e.g., 'tell me about history'), set context (e.g., 'set subject to math'), or learn (e.g., 'fact: The sky is blue'). Type 'exit' to quit.\n");

    char input[512];
    char* response = (char*)calloc(MAX_RESPONSE_LENGTH, sizeof(char));
    if (!response) {
        handle_error(&context, "Memory allocation failed for response");
        free_conversation_context(&context);
        curl_global_cleanup();
        return 1;
    }

    while (1) {
        printf("\nYou: ");
        if (!fgets(input, sizeof(input), stdin)) {
            handle_error(&context, "Failed to read input");
            break;
        }
        input[strcspn(input, "\n")] = '\0';
        if (strcmp(input, "exit") == 0) break;

        memset(response, 0, MAX_RESPONSE_LENGTH);
        generate_enhanced_response(input, response, MAX_RESPONSE_LENGTH, &context);
        printf("May: %s\n", response);
    }

    free(response);
    free_conversation_context(&context);
    curl_global_cleanup();
    return 0;
}
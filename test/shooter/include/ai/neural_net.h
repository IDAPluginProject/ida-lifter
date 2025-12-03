/*
 * Neural Network with AVX/AVX2 Matrix Operations
 * Simple feedforward neural network for AI behavior decisions.
 * Heavily uses AVX/AVX2 for all matrix operations.
 */

#ifndef SHOOTER_NEURAL_NET_H
#define SHOOTER_NEURAL_NET_H

#include <immintrin.h>
#include <stdbool.h>
#include <string.h>
#include "../config.h"
#include "../math/avx_math.h"

/* ==========================================================================
 * NETWORK CONFIGURATION
 * ========================================================================== */

#define NN_MAX_LAYERS 4
#define NN_MAX_NEURONS_PER_LAYER 32
#define NN_INPUT_SIZE 24          /* Input features */
#define NN_HIDDEN_SIZE 16         /* Hidden layer neurons */
#define NN_OUTPUT_SIZE 8          /* Output actions */

/* Activation functions */
typedef enum {
    NN_ACT_RELU,
    NN_ACT_SIGMOID,
    NN_ACT_TANH,
    NN_ACT_SOFTMAX,
    NN_ACT_LINEAR
} NNActivation;

/* ==========================================================================
 * NETWORK STRUCTURES - All aligned for AVX
 * ========================================================================== */

typedef struct __attribute__((aligned(32))) {
    int input_size;
    int output_size;
    NNActivation activation;

    /* Weights: output_size x input_size (row-major) */
    /* Padded to multiples of 8 for AVX */
    float weights[NN_MAX_NEURONS_PER_LAYER][NN_MAX_NEURONS_PER_LAYER];
    float biases[NN_MAX_NEURONS_PER_LAYER];

    /* Intermediate storage */
    float pre_activation[NN_MAX_NEURONS_PER_LAYER];
    float output[NN_MAX_NEURONS_PER_LAYER];
} NNLayer;

typedef struct __attribute__((aligned(32))) {
    int layer_count;
    NNLayer layers[NN_MAX_LAYERS];

    /* Network input/output */
    float input[NN_MAX_NEURONS_PER_LAYER];
    float output[NN_MAX_NEURONS_PER_LAYER];
} NeuralNet;

/* ==========================================================================
 * AVX ACTIVATION FUNCTIONS
 * ========================================================================== */

/* AVX ReLU: max(0, x) for 8 values */
static inline __m256 nn_avx_relu(__m256 x) {
    return _mm256_max_ps(x, _mm256_setzero_ps());
}

/* AVX ReLU derivative: 1 if x > 0, else 0 */
static inline __m256 nn_avx_relu_grad(__m256 x) {
    __m256 zero = _mm256_setzero_ps();
    __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
    return _mm256_and_ps(mask, _mm256_set1_ps(1.0f));
}

/* AVX Leaky ReLU: x if x > 0, else 0.01*x */
static inline __m256 nn_avx_leaky_relu(__m256 x) {
    __m256 zero = _mm256_setzero_ps();
    __m256 alpha = _mm256_set1_ps(0.01f);
    __m256 neg_part = _mm256_mul_ps(x, alpha);
    __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
    return _mm256_blendv_ps(neg_part, x, mask);
}

/* AVX Sigmoid approximation: 1 / (1 + exp(-x))
 * Uses fast approximation: 0.5 + 0.5 * tanh(x * 0.5) */
static inline __m256 nn_avx_sigmoid(__m256 x) {
    /* Clamp to avoid overflow */
    x = _mm256_max_ps(x, _mm256_set1_ps(-10.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(10.0f));

    /* Fast sigmoid using polynomial approximation */
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);

    /* Approximate sigmoid with: 0.5 * (1 + x / (1 + |x|)) */
    __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    __m256 denom = _mm256_add_ps(one, abs_x);
    __m256 frac = _mm256_div_ps(x, denom);

    return _mm256_mul_ps(half, _mm256_add_ps(one, frac));
}

/* AVX Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x)) */
static inline __m256 nn_avx_sigmoid_grad(__m256 sigmoid_output) {
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_mul_ps(sigmoid_output, _mm256_sub_ps(one, sigmoid_output));
}

/* AVX Tanh approximation */
static inline __m256 nn_avx_tanh(__m256 x) {
    /* Clamp to avoid overflow */
    x = _mm256_max_ps(x, _mm256_set1_ps(-5.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(5.0f));

    /* Approximate tanh with: x / (1 + |x|) * (1 + small correction) */
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    __m256 x2 = _mm256_mul_ps(x, x);

    /* Better approximation: x * (27 + x^2) / (27 + 9*x^2) */
    __m256 c27 = _mm256_set1_ps(27.0f);
    __m256 c9 = _mm256_set1_ps(9.0f);
    __m256 num = _mm256_mul_ps(x, _mm256_add_ps(c27, x2));
    __m256 denom = _mm256_add_ps(c27, _mm256_mul_ps(c9, x2));

    return _mm256_div_ps(num, denom);
}

/* AVX Tanh derivative: 1 - tanh(x)^2 */
static inline __m256 nn_avx_tanh_grad(__m256 tanh_output) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 sq = _mm256_mul_ps(tanh_output, tanh_output);
    return _mm256_sub_ps(one, sq);
}

/* Softmax for output layer (scalar, needs horizontal ops) */
static inline void nn_softmax(float* values, int size) {
    /* Find max for numerical stability */
    __m256 vmax = _mm256_set1_ps(-1e30f);
    int i;
    for (i = 0; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&values[i]);
        vmax = _mm256_max_ps(vmax, v);
    }

    /* Horizontal max */
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 max4 = _mm_max_ps(lo, hi);
    __m128 max2 = _mm_max_ps(max4, _mm_shuffle_ps(max4, max4, _MM_SHUFFLE(3,2,3,2)));
    __m128 max1 = _mm_max_ss(max2, _mm_shuffle_ps(max2, max2, 1));
    float max_val = _mm_cvtss_f32(max1);

    /* Handle tail */
    for (; i < size; i++) {
        if (values[i] > max_val) max_val = values[i];
    }

    /* Subtract max and compute exp */
    __m256 vmax_broadcast = _mm256_set1_ps(max_val);
    __m256 sum = _mm256_setzero_ps();

    float __attribute__((aligned(32))) temp[32];

    for (i = 0; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&values[i]);
        v = _mm256_sub_ps(v, vmax_broadcast);

        /* Fast exp approximation: (1 + x/256)^256 ≈ exp(x) for small x */
        /* Use polynomial: exp(x) ≈ 1 + x + x^2/2 + x^3/6 */
        __m256 x = v;
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);

        __m256 exp_approx = _mm256_add_ps(one, x);
        exp_approx = _mm256_add_ps(exp_approx,
            _mm256_mul_ps(x2, _mm256_set1_ps(0.5f)));
        exp_approx = _mm256_add_ps(exp_approx,
            _mm256_mul_ps(x3, _mm256_set1_ps(0.166667f)));

        /* Clamp to positive */
        exp_approx = _mm256_max_ps(exp_approx, _mm256_set1_ps(1e-10f));

        _mm256_storeu_ps(&temp[i], exp_approx);
        sum = _mm256_add_ps(sum, exp_approx);
    }

    /* Horizontal sum */
    float total_sum = hsum256_ps(sum);

    /* Handle tail with scalar exp */
    for (; i < size; i++) {
        float v = values[i] - max_val;
        /* Clamp for safety */
        if (v < -20.0f) v = -20.0f;
        if (v > 20.0f) v = 20.0f;

        float exp_v = 1.0f + v + v*v*0.5f + v*v*v*0.166667f;
        if (exp_v < 1e-10f) exp_v = 1e-10f;
        temp[i] = exp_v;
        total_sum += exp_v;
    }

    /* Normalize */
    __m256 inv_sum = _mm256_set1_ps(1.0f / total_sum);
    for (i = 0; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&temp[i]);
        v = _mm256_mul_ps(v, inv_sum);
        _mm256_storeu_ps(&values[i], v);
    }
    for (; i < size; i++) {
        values[i] = temp[i] / total_sum;
    }
}

/* ==========================================================================
 * AVX MATRIX OPERATIONS
 * ========================================================================== */

/* Matrix-vector multiply: out = W * in + b
 * W is output_size x input_size, in is input_size, out is output_size */
static inline void nn_avx_matvec(
    const float* W,            /* Row-major weights */
    const float* in,
    const float* bias,
    float* out,
    int output_size,
    int input_size
) {
    /* Pad input to multiple of 8 */
    float __attribute__((aligned(32))) padded_in[NN_MAX_NEURONS_PER_LAYER];
    memset(padded_in, 0, sizeof(padded_in));
    memcpy(padded_in, in, input_size * sizeof(float));

    int padded_input_size = (input_size + 7) & ~7;

    for (int row = 0; row < output_size; row++) {
        __m256 sum = _mm256_setzero_ps();

        const float* W_row = &W[row * NN_MAX_NEURONS_PER_LAYER];

        /* Process 8 elements at a time */
        for (int col = 0; col < padded_input_size; col += 8) {
            __m256 w = _mm256_loadu_ps(&W_row[col]);
            __m256 x = _mm256_loadu_ps(&padded_in[col]);
            sum = fmadd_ps(w, x, sum);
        }

        /* Horizontal sum + bias */
        out[row] = hsum256_ps(sum) + bias[row];
    }
}

/* Apply activation function to 8 values */
static inline void nn_avx_activate(float* values, int size, NNActivation act) {
    switch (act) {
        case NN_ACT_RELU:
            for (int i = 0; i + 8 <= size; i += 8) {
                __m256 v = _mm256_loadu_ps(&values[i]);
                v = nn_avx_relu(v);
                _mm256_storeu_ps(&values[i], v);
            }
            for (int i = (size / 8) * 8; i < size; i++) {
                if (values[i] < 0) values[i] = 0;
            }
            break;

        case NN_ACT_SIGMOID:
            for (int i = 0; i + 8 <= size; i += 8) {
                __m256 v = _mm256_loadu_ps(&values[i]);
                v = nn_avx_sigmoid(v);
                _mm256_storeu_ps(&values[i], v);
            }
            for (int i = (size / 8) * 8; i < size; i++) {
                float x = values[i];
                x = x > 10.0f ? 10.0f : (x < -10.0f ? -10.0f : x);
                values[i] = 0.5f * (1.0f + x / (1.0f + (x < 0 ? -x : x)));
            }
            break;

        case NN_ACT_TANH:
            for (int i = 0; i + 8 <= size; i += 8) {
                __m256 v = _mm256_loadu_ps(&values[i]);
                v = nn_avx_tanh(v);
                _mm256_storeu_ps(&values[i], v);
            }
            for (int i = (size / 8) * 8; i < size; i++) {
                float x = values[i];
                x = x > 5.0f ? 5.0f : (x < -5.0f ? -5.0f : x);
                float x2 = x * x;
                values[i] = x * (27.0f + x2) / (27.0f + 9.0f * x2);
            }
            break;

        case NN_ACT_SOFTMAX:
            nn_softmax(values, size);
            break;

        case NN_ACT_LINEAR:
        default:
            /* No activation */
            break;
    }
}

/* ==========================================================================
 * LAYER OPERATIONS
 * ========================================================================== */

/* Initialize a layer with random weights (simple Gaussian-ish) */
static inline void nn_layer_init(NNLayer* layer, int input_size, int output_size,
                                  NNActivation activation, uint32_t* rng_state) {
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;

    /* Xavier-ish initialization */
    float scale = 2.0f / (input_size + output_size);

    for (int o = 0; o < output_size; o++) {
        for (int i = 0; i < input_size; i++) {
            /* Simple LCG for random */
            *rng_state = *rng_state * 1103515245 + 12345;
            float r = (float)(*rng_state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
            r = (r - 0.5f) * 2.0f;  /* -1 to 1 */
            layer->weights[o][i] = r * scale;
        }
        layer->biases[o] = 0.0f;
    }
}

/* Forward pass through a layer */
static inline void nn_layer_forward(NNLayer* layer, const float* input) {
    nn_avx_matvec(
        (const float*)layer->weights,
        input,
        layer->biases,
        layer->pre_activation,
        layer->output_size,
        layer->input_size
    );

    memcpy(layer->output, layer->pre_activation,
           layer->output_size * sizeof(float));

    nn_avx_activate(layer->output, layer->output_size, layer->activation);
}

/* ==========================================================================
 * NETWORK OPERATIONS
 * ========================================================================== */

/* Initialize a 3-layer network: input -> hidden -> output */
static inline void nn_init_3layer(NeuralNet* net, int input_size, int hidden_size,
                                   int output_size, uint32_t* rng_state) {
    net->layer_count = 2;

    nn_layer_init(&net->layers[0], input_size, hidden_size, NN_ACT_RELU, rng_state);
    nn_layer_init(&net->layers[1], hidden_size, output_size, NN_ACT_SOFTMAX, rng_state);
}

/* Initialize a 4-layer network: input -> hidden1 -> hidden2 -> output */
static inline void nn_init_4layer(NeuralNet* net, int input_size, int hidden1,
                                   int hidden2, int output_size, uint32_t* rng_state) {
    net->layer_count = 3;

    nn_layer_init(&net->layers[0], input_size, hidden1, NN_ACT_RELU, rng_state);
    nn_layer_init(&net->layers[1], hidden1, hidden2, NN_ACT_RELU, rng_state);
    nn_layer_init(&net->layers[2], hidden2, output_size, NN_ACT_SOFTMAX, rng_state);
}

/* Forward pass through entire network */
static inline void nn_forward(NeuralNet* net, const float* input) {
    memcpy(net->input, input, net->layers[0].input_size * sizeof(float));

    /* First layer */
    nn_layer_forward(&net->layers[0], net->input);

    /* Hidden layers */
    for (int i = 1; i < net->layer_count; i++) {
        nn_layer_forward(&net->layers[i], net->layers[i-1].output);
    }

    /* Copy final output */
    memcpy(net->output, net->layers[net->layer_count - 1].output,
           net->layers[net->layer_count - 1].output_size * sizeof(float));
}

/* Get action with highest probability */
static inline int nn_get_best_action(const NeuralNet* net) {
    int best = 0;
    float best_val = net->output[0];

    int output_size = net->layers[net->layer_count - 1].output_size;
    for (int i = 1; i < output_size; i++) {
        if (net->output[i] > best_val) {
            best_val = net->output[i];
            best = i;
        }
    }
    return best;
}

/* Get action by sampling from probability distribution */
static inline int nn_sample_action(const NeuralNet* net, uint32_t* rng_state) {
    *rng_state = *rng_state * 1103515245 + 12345;
    float r = (float)(*rng_state & 0x7FFFFFFF) / (float)0x7FFFFFFF;

    float cumsum = 0.0f;
    int output_size = net->layers[net->layer_count - 1].output_size;

    for (int i = 0; i < output_size; i++) {
        cumsum += net->output[i];
        if (r <= cumsum) {
            return i;
        }
    }
    return output_size - 1;
}

/* ==========================================================================
 * AVX BATCH FORWARD PASS - Process 8 networks in parallel
 * ========================================================================== */

typedef struct __attribute__((aligned(32))) {
    float inputs[8][NN_MAX_NEURONS_PER_LAYER];
    float outputs[8][NN_MAX_NEURONS_PER_LAYER];
    float hidden[8][NN_MAX_NEURONS_PER_LAYER];
} NNBatchState;

/* Batch forward for first layer - process 8 inputs through same weights */
static inline void nn_avx_batch_layer_forward(
    const NNLayer* layer,
    const NNBatchState* batch_in,
    NNBatchState* batch_out,
    int batch_size
) {
    /* For each output neuron */
    for (int o = 0; o < layer->output_size; o++) {
        __m256 results = _mm256_setzero_ps();

        /* Compute dot product for all 8 batch elements */
        for (int i = 0; i < layer->input_size; i++) {
            __m256 w = _mm256_set1_ps(layer->weights[o][i]);

            /* Load input[batch][i] for all 8 batches */
            __m256 x = _mm256_set_ps(
                batch_in->inputs[7][i], batch_in->inputs[6][i],
                batch_in->inputs[5][i], batch_in->inputs[4][i],
                batch_in->inputs[3][i], batch_in->inputs[2][i],
                batch_in->inputs[1][i], batch_in->inputs[0][i]
            );

            results = fmadd_ps(w, x, results);
        }

        /* Add bias */
        results = _mm256_add_ps(results, _mm256_set1_ps(layer->biases[o]));

        /* Apply activation */
        switch (layer->activation) {
            case NN_ACT_RELU:
                results = nn_avx_relu(results);
                break;
            case NN_ACT_SIGMOID:
                results = nn_avx_sigmoid(results);
                break;
            case NN_ACT_TANH:
                results = nn_avx_tanh(results);
                break;
            default:
                break;
        }

        /* Store results - scatter to batch_out->outputs[batch][o] */
        float __attribute__((aligned(32))) temp[8];
        _mm256_storeu_ps(temp, results);
        for (int b = 0; b < batch_size; b++) {
            batch_out->outputs[b][o] = temp[b];
        }
    }
}

/* ==========================================================================
 * AI INPUT FEATURE EXTRACTION
 * ========================================================================== */

typedef struct __attribute__((aligned(32))) {
    /* Self state (8 features) */
    float health_ratio;
    float ammo_ratio;
    float stamina_ratio;
    float medpen_count;
    float in_cover;
    float is_crouching;
    float is_running;
    float weapon_range;

    /* Target state (8 features) */
    float has_target;
    float target_visible;
    float target_distance;
    float target_health;
    float target_in_range;
    float target_facing_us;
    float time_visible;
    float target_velocity;

    /* Combat state (8 features) */
    float threat_count;
    float nearest_threat_dist;
    float being_suppressed;
    float frames_in_combat;
    float damage_recently;
    float stalemate_time;
    float ally_nearby;
    float squad_health;
} NNInputFeatures;

/* Extract features from entity state - all using AVX where possible */
static inline void nn_extract_features(
    const Entity* e,
    const GameState* game,
    float* out_features
) {
    NNInputFeatures features;
    memset(&features, 0, sizeof(features));

    /* Self state */
    features.health_ratio = e->health / e->max_health;
    WeaponStats ws = weapon_get_stats(&e->weapon);
    features.ammo_ratio = (float)e->weapon.mag_current / ws.mag_size;
    features.stamina_ratio = e->stamina / PLAYER_MAX_STAMINA;
    features.medpen_count = (float)e->medpens / MEDPEN_MAX;
    features.in_cover = e->has_cover_nearby ? 1.0f : 0.0f;
    features.is_crouching = e->is_crouching ? 1.0f : 0.0f;
    features.is_running = e->is_running ? 1.0f : 0.0f;
    features.weapon_range = ws.range / 50.0f;  /* Normalize */

    /* Target state */
    if (e->primary_threat >= 0 && e->primary_threat < e->threat_count) {
        const ThreatInfo* threat = &e->threats[e->primary_threat];
        const Entity* target = &game->entities[threat->entity_id];

        features.has_target = 1.0f;
        features.target_visible = threat->is_visible ? 1.0f : 0.0f;
        features.target_distance = threat->distance / 50.0f;  /* Normalize */
        features.target_health = target->health / target->max_health;
        features.target_in_range = (threat->distance < ws.range) ? 1.0f : 0.0f;
        features.target_facing_us = threat->is_aware_of_us ? 1.0f : 0.0f;
        features.time_visible = (float)threat->frames_visible / 60.0f;

        float vx = target->vx, vy = target->vy;
        features.target_velocity = sqrtf(vx*vx + vy*vy) / PLAYER_RUN_SPEED;
    }

    /* Combat state */
    features.threat_count = (float)e->threat_count / MAX_TRACKED_THREATS;

    /* Find nearest threat */
    float nearest = 100.0f;
    for (int i = 0; i < e->threat_count; i++) {
        if (e->threats[i].distance < nearest) {
            nearest = e->threats[i].distance;
        }
    }
    features.nearest_threat_dist = nearest / 50.0f;

    features.being_suppressed = 0.0f;  /* TODO: track suppression */
    features.frames_in_combat = (e->state == STATE_COMBAT) ?
        (float)e->stalemate_timer / 120.0f : 0.0f;
    features.damage_recently = (e->damage_react_timer > 0) ? 1.0f : 0.0f;
    features.stalemate_time = (float)e->stalemate_timer / 90.0f;

    /* Squad state using AVX to count nearby allies */
    if (e->squad_id >= 0) {
        const SquadInfo* squad = &game->squads.squads[e->squad_id];
        float ally_health_sum = 0;
        int ally_count = 0;

        for (int i = 0; i < squad->member_count; i++) {
            int ally_id = squad->member_ids[i];
            if (ally_id != e->id && game->entities[ally_id].alive) {
                ally_count++;
                ally_health_sum += game->entities[ally_id].health /
                                   game->entities[ally_id].max_health;
            }
        }

        features.ally_nearby = (ally_count > 0) ? 1.0f : 0.0f;
        features.squad_health = (ally_count > 0) ?
            (ally_health_sum / ally_count) : 0.0f;
    }

    /* Copy to output array */
    memcpy(out_features, &features, sizeof(features));
}

/* ==========================================================================
 * AI OUTPUT ACTION MAPPING
 * ========================================================================== */

typedef enum {
    NN_ACTION_ATTACK = 0,
    NN_ACTION_DEFEND,
    NN_ACTION_RETREAT,
    NN_ACTION_FLANK,
    NN_ACTION_HEAL,
    NN_ACTION_RELOAD,
    NN_ACTION_PATROL,
    NN_ACTION_SUPPORT,
    NN_ACTION_COUNT
} NNAction;

/* Map neural network output to game state */
static inline int nn_action_to_state(NNAction action) {
    switch (action) {
        case NN_ACTION_ATTACK:  return STATE_COMBAT;
        case NN_ACTION_DEFEND:  return STATE_HIDING;
        case NN_ACTION_RETREAT: return STATE_RETREATING;
        case NN_ACTION_FLANK:   return STATE_FLANKING;
        case NN_ACTION_HEAL:    return STATE_HEALING;
        case NN_ACTION_RELOAD:  return STATE_RELOAD;
        case NN_ACTION_PATROL:  return STATE_PATROL;
        case NN_ACTION_SUPPORT: return STATE_SUPPORTING;
        default:                return STATE_PATROL;
    }
}

/* ==========================================================================
 * BATCH INFERENCE FOR MULTIPLE ENTITIES
 * ========================================================================== */

/* Process 8 entities through same network in parallel */
static inline void nn_batch_inference_8(
    const NeuralNet* net,
    float inputs[8][NN_INPUT_SIZE],
    int* out_actions
) {
    /* This version processes each entity sequentially but uses AVX within */
    /* A more advanced version could parallelize across the batch dimension */

    for (int b = 0; b < 8; b++) {
        /* Copy input */
        float __attribute__((aligned(32))) input_buf[NN_MAX_NEURONS_PER_LAYER];
        memset(input_buf, 0, sizeof(input_buf));
        memcpy(input_buf, inputs[b], NN_INPUT_SIZE * sizeof(float));

        /* Forward through layers */
        float __attribute__((aligned(32))) hidden[NN_MAX_NEURONS_PER_LAYER];

        /* Layer 0 */
        nn_avx_matvec(
            (const float*)net->layers[0].weights,
            input_buf,
            net->layers[0].biases,
            hidden,
            net->layers[0].output_size,
            net->layers[0].input_size
        );
        nn_avx_activate(hidden, net->layers[0].output_size, net->layers[0].activation);

        /* Remaining layers */
        float __attribute__((aligned(32))) temp[NN_MAX_NEURONS_PER_LAYER];
        float* current = hidden;
        float* next = temp;

        for (int l = 1; l < net->layer_count; l++) {
            nn_avx_matvec(
                (const float*)net->layers[l].weights,
                current,
                net->layers[l].biases,
                next,
                net->layers[l].output_size,
                net->layers[l].input_size
            );
            nn_avx_activate(next, net->layers[l].output_size, net->layers[l].activation);

            /* Swap buffers */
            float* tmp = current;
            current = next;
            next = tmp;
        }

        /* Find best action using AVX */
        int output_size = net->layers[net->layer_count - 1].output_size;
        __m256 best_val = _mm256_set1_ps(-1e30f);
        __m256i best_idx = _mm256_setzero_si256();

        if (output_size == 8) {
            __m256 vals = _mm256_loadu_ps(current);
            __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

            /* Find max - need to reduce */
            float __attribute__((aligned(32))) val_arr[8];
            _mm256_storeu_ps(val_arr, vals);

            int best = 0;
            float bv = val_arr[0];
            for (int i = 1; i < 8; i++) {
                if (val_arr[i] > bv) {
                    bv = val_arr[i];
                    best = i;
                }
            }
            out_actions[b] = best;
        } else {
            /* Fallback for non-8 output sizes */
            int best = 0;
            float bv = current[0];
            for (int i = 1; i < output_size; i++) {
                if (current[i] > bv) {
                    bv = current[i];
                    best = i;
                }
            }
            out_actions[b] = best;
        }
    }
}

#endif /* SHOOTER_NEURAL_NET_H */

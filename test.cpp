#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#define INIT_A 0x43b0d7e5
#define MULT_A 0x931e8875
#define INIT_B 0x8b51f9dd
#define MULT_B 0x58f38ded
#define MIX_MULT_L 0xca01f9dd
#define MIX_MULT_R 0x4973f715
#define XSHIFT 16  // For 32-bit words
#define MASK32 0xFFFFFFFF

// Helper: Mix two 32-bit values
static inline uint32_t mix(uint32_t x, uint32_t y) {
    uint32_t result = (MIX_MULT_L * x - MIX_MULT_R * y);
    return result ^ (result >> XSHIFT);
}

// Hash function used in entropy mixing
static inline uint32_t hashmix(uint32_t value, uint32_t *hash_const) {
    value ^= *hash_const;
    *hash_const *= MULT_A;
    value *= *hash_const;
    return value ^ (value >> XSHIFT);
}

// Convert a seed (integer or array) into a uint32 array
void seed_to_uint32_array(uint64_t seed, uint32_t **out, size_t *out_len) {
    // For simplicity, treat seed as a single 64-bit value split into 2x32-bit
    *out_len = 2;
    *out = (uint32_t*)malloc(*out_len * sizeof(uint32_t));
    (*out)[0] = (uint32_t)(seed & MASK32);
    (*out)[1] = (uint32_t)((seed >> 32) & MASK32);
}

// Core entropy mixing (like SeedSequence.mix_entropy)
void mix_entropy(uint32_t *pool, size_t pool_size, uint32_t *entropy, size_t entropy_len) {
    uint32_t hash_const = INIT_A;

    // Step 1: Fill the pool with hashed entropy
    for (size_t i = 0; i < pool_size; i++) {
        pool[i] = (i < entropy_len) 
            ? hashmix(entropy[i], &hash_const)
            : hashmix(0, &hash_const);
    }

    // Step 2: Mix all pool entries
    for (size_t i_src = 0; i_src < pool_size; i_src++) {
        for (size_t i_dst = 0; i_dst < pool_size; i_dst++) {
            if (i_src != i_dst) {
                pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src], &hash_const));
            }
        }
    }

    // Step 3: Mix remaining entropy (if any)
    for (size_t i_src = pool_size; i_src < entropy_len; i_src++) {
        for (size_t i_dst = 0; i_dst < pool_size; i_dst++) {
            pool[i_dst] = mix(pool[i_dst], hashmix(entropy[i_src], &hash_const));
        }
    }
}
// Generate initial RNG state from a seed
void generate_rng_state(uint64_t seed, uint32_t *state, size_t state_size) {
    uint32_t *entropy = NULL;
    size_t entropy_len = 0;
    seed_to_uint32_array(seed, &entropy, &entropy_len);

    // Default pool size (like NumPy's DEFAULT_POOL_SIZE = 4)
    const size_t pool_size = 4;
    uint32_t pool[pool_size];
    mix_entropy(pool, pool_size, entropy, entropy_len);

    // Step 3: Expand pool into RNG state (similar to generate_state)
    uint32_t hash_const = INIT_B;
    for (size_t i = 0; i < state_size; i++) {
        uint32_t val = pool[i % pool_size];
        val ^= hash_const;
        hash_const *= MULT_B;
        val *= hash_const;
        val ^= (val >> XSHIFT);
        state[i] = val;
    }

    free(entropy);
}
int main() {
    uint64_t seed = 12345;  // Input seed
    size_t state_size = 4;  // Desired state size (e.g., 4 for PCG)
    uint32_t state[state_size];

    generate_rng_state(seed, state, state_size);

    // Print the generated state
    for (size_t i = 0; i < state_size; i++) {
        printf("state[%zu] = 0x%08x\n", i, state[i]);
    }

    return 0;
}

#ifndef SEED_SEQUENCE_H
#define SEED_SEQUENCE_H

#include <stdint.h>
#include <vector>

#define INIT_A 0x43b0d7e5
#define MULT_A 0x931e8875
#define INIT_B 0x8b51f9dd
#define MULT_B 0x58f38ded
#define XSHIFT 16 // 32-bit word size divided by 2
#define MASK32 0xFFFFFFFF
#define MIX_MULT_L 0xca01f9dd
#define MIX_MULT_R 0x4973f715
#define DEFAULT_POOL_SIZE 4
typedef struct {
    uint64_t state_high;
    uint64_t state_low;
    uint64_t inc_high;
    uint64_t inc_low;
    std::vector<uint32_t> pool; // Pool used for entropy mixing
    uint64_t seed;
} SeedState;

// Function to initialize the SeedState using entropy
extern void initialize_seed_state(uint64_t entropy, SeedState *seed_state);
extern void init_seed_state(uint64_t seed, SeedState *seed_state);

// Function to generate the state array for PRNG seeding
extern void generate_state(SeedState &seed_state, std::vector<uint64_t> &state, int n_words);



#endif // SEED_SEQUENCE_H

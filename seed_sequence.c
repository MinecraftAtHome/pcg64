#include "seed_sequence.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <cstdio>

std::vector<uint64_t> reinterpretAs64bit(const std::vector<uint32_t>& input) {
    // Calculate the output size (half the input size, rounded down)
    size_t outputSize = input.size() / 2;
    
    // Create output vector
    std::vector<uint64_t> output(outputSize);
    
    // Copy the memory directly, reinterpreting the bit pattern
    memcpy(output.data(), input.data(), outputSize * sizeof(uint64_t));

    return output;
}
std::vector<uint32_t> _coerce_to_uint32_array(uint64_t value){
    std::vector<uint32_t> vec;    
    if( value == 0){
        vec.push_back((uint32_t)value);
    }
    while(value > 0){
        vec.push_back((uint32_t)(value & MASK32));
        value = floor(value/pow(2, 32));
    }
    return vec;
}
std::vector<uint32_t> get_assembled_entropy(uint64_t seed){

    return _coerce_to_uint32_array(seed);
}
uint32_t hashmix(uint32_t value, uint32_t* hash_const){
    //# We are modifying the multiplier as we go along, so it is input-output
    value ^= hash_const[0];
    hash_const[0] *= MULT_A;
    value *=  hash_const[0];
    value ^= value >> XSHIFT;
    return value;
}
uint32_t mix(uint32_t x, uint32_t y){
    uint32_t result = (MIX_MULT_L * x - MIX_MULT_R * y);
    result ^= result >> XSHIFT;
    return result;
}

void mix_entropy(std::vector<uint32_t> &pool, std::vector<uint32_t> assembled_entropy){
    uint32_t hash_const[1];
    hash_const[0] = INIT_A;
    uint32_t ass_ent_len = assembled_entropy.size();

    for(int i = 0; i < DEFAULT_POOL_SIZE; i++){
        if(i < ass_ent_len){
            pool[i] = hashmix(assembled_entropy[i], hash_const);
        }
        else{

            pool[i] = hashmix(0, hash_const);
        }
    }
    for(int i_src = 0; i_src < DEFAULT_POOL_SIZE; i_src++){
        //Mix all bits together so late bits can affect earlier bits.
        for(int i_dst = 0; i_dst < DEFAULT_POOL_SIZE; i_dst++){
            if(i_src != i_dst){
                pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src], hash_const));

            }
        }
    }

    for(int i_src = DEFAULT_POOL_SIZE; i_src < ass_ent_len; i_src++){
        for(int i_dst = 0; i_dst < DEFAULT_POOL_SIZE; i_dst++){
            pool[i_dst] = mix(pool[i_dst], hashmix(assembled_entropy[i_src], hash_const));
        }
    }
}
void init_seed_state(uint64_t seed, SeedState *seed_state){

    seed_state->seed = seed;
    for(int i = 0; i < DEFAULT_POOL_SIZE; i++){
        seed_state->pool.push_back(0);
    }
    mix_entropy(seed_state->pool, get_assembled_entropy(seed_state->seed));
}
int lastCycleIndex = 0;
uint32_t cycle(std::vector<uint32_t> src){
    uint32_t ret_val = src[lastCycleIndex];
    printf("%u\n",ret_val);
    lastCycleIndex = (lastCycleIndex + 1) % src.size();
    return ret_val;
}
void generate_state(SeedState &seed_state, std::vector<uint64_t> &state, int n_words) {
    uint32_t hash_const = INIT_B;
    uint32_t data_val;
    n_words *= 2;

    // Initialize the state array
    printf("Init state array\n");
    std::vector<uint32_t> temp_state;
    for (int i = 0; i < n_words; i++) {
        temp_state.push_back(0);
    }
    for(int i_dst = 0; i_dst < n_words; i_dst++){
        data_val = cycle(seed_state.pool);
        data_val ^= hash_const;
        hash_const *= MULT_B;
        data_val *= hash_const;
        data_val ^= data_val >> XSHIFT;
        temp_state[i_dst] = data_val;
    }
    state = reinterpretAs64bit(temp_state);

}

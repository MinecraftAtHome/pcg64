#define PCG_FORCE_EMULATED_128BIT_MATH
#include "pcg64.h"
#include "seed_sequence.h"
#include <cstring>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

#include <string>
#include <sstream>
#include <random>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <stdexcept>
#include <memory>




int main() {
	pcg64_state rng;
	rng.pcg_state = (pcg64_random_t*)malloc(sizeof(pcg64_random_t));
	memset(rng.pcg_state, 0, sizeof(pcg64_random_t));

	
	uint64_t entropy = 0; // Example entropy value
	SeedState seed_state;
	init_seed_state(entropy, &seed_state);

	std::vector<uint64_t> state_array;
	
	generate_state(seed_state, state_array, 4);
	printf("state.0 = %llu\n", state_array[0]);
	printf("state.1 = %llu\n", state_array[1]);
	printf("state.2 = %llu\n", state_array[2]);
	printf("state.3 = %llu\n", state_array[3]);

	pcg64_set_seed(&rng, &state_array[0], &state_array[2]);
	uint64_t stlow = rng.pcg_state->state.low;
	uint64_t sthigh = rng.pcg_state->state.high;
	uint64_t inclow = rng.pcg_state->inc.low;
	uint64_t inchigh = rng.pcg_state->inc.high;
	uint64_t *arr = (uint64_t*) malloc(sizeof(uint64_t)*4);
	arr[0] = sthigh;
	arr[1] = stlow;
	arr[2] = inchigh;
	arr[3] = inclow;
	pcg64_set_state(&rng, arr, 0, 0);
	rng.has_uint32 = 0;
	rng.uinteger = 0;

	printf("state.high = %llu\n", (unsigned long long)rng.pcg_state->state.high);
	printf("state.low  = %llu\n", (unsigned long long)rng.pcg_state->state.low);
	printf("inc.high   = %llu\n", (unsigned long long)rng.pcg_state->inc.high);
	printf("inc.low    = %llu\n", (unsigned long long)rng.pcg_state->inc.low);
	for (int i = 0; i < 10; ++i) {
		printf("%llu\n", (unsigned long long)pcg64_next64(&rng));
	}

	free(rng.pcg_state);
	return 0;
}
	
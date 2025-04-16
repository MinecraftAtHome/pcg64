import numpy as np

rng = np.random.default_rng(0)
state = rng.bit_generator.state

# For PCG64, the state dict looks like this:
# {
#   'bit_generator': 'PCG64',
#   'state': {
#       'state': <int>,
#       'inc': <int>
#   },
#   ...
# }

state_int = state['state']['state']
inc_int = state['state']['inc']

# Extract high and low 64 bits
def split_uint128(x):
    low = x & ((1 << 64) - 1)
    high = x >> 64
    return high, low

state_high, state_low = split_uint128(state_int)
inc_high, inc_low = split_uint128(inc_int)

print(f"state.high = {state_high}")
print(f"state.low  = {state_low}")
print(f"inc.high   = {inc_high}")
print(f"inc.low    = {inc_low}")

for i in range(10):
    print(rng.integers(0, 2**64, dtype=np.uint64))
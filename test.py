import numpy as np

def debug_generate_state(pool, n_words, dtype=np.uint32):
    print("Debug generate_state:")
    print(f"n_words = {n_words}, dtype = {dtype}")
    print(f"Pool = {[hex(x) for x in pool]}")
    
    # Constants
    INIT_B = 0x931e8766
    MULT_B = 0x74dcd1d9
    XSHIFT = 13
    
    # Create state array
    if dtype == np.uint64:
        n_words *= 2
    state = np.zeros(n_words, dtype=np.uint32)
    
    # Fill state array
    hash_const = INIT_B
    for i_dst in range(n_words):
        data_val = pool[i_dst % len(pool)]
        print(f"Step {i_dst}:")
        print(f"  data_val = {hex(data_val)}")
        print(f"  hash_const = {hex(hash_const)}")
        
        data_val ^= hash_const
        print(f"  data_val ^= hash_const = {hex(data_val)}")
        
        hash_const *= MULT_B
        print(f"  hash_const *= MULT_B = {hex(hash_const & 0xFFFFFFFF)}")
        
        data_val *= hash_const
        print(f"  data_val *= hash_const = {hex(data_val & 0xFFFFFFFF)}")
        
        data_val ^= data_val >> XSHIFT
        print(f"  data_val ^= data_val >> XSHIFT = {hex(data_val & 0xFFFFFFFF)}")
        
        state[i_dst] = data_val
        print(f"  state[{i_dst}] = {hex(data_val & 0xFFFFFFFF)}")
    
    # Convert to uint64 if needed
    if dtype == np.uint64:
        state = state.view(np.uint64)
    
    print(f"Final state = {state}")
    return state

print(np.dtype(np.uint32).itemsize * 8 // 2)
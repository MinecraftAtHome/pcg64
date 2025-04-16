import numpy as np
import os.path
import sys
import native

def gf2_rank_subset(I, J, M):
    """
    Single-pass GF(2) rank + dependency detection with further micro-optimizations:
      * Use argmax to find pivot.
      * Minimize overhead inside the for-loop.
      * Keep the single large matrix approach (top=selected, bottom=candidates).

    Parameters
    ----------
    I, J : boolean arrays of shape (n,)
        Indicators for which rows/columns to include.
    M    : boolean array of shape (n, n)
        Matrix over GF(2).

    Returns
    -------
    rank_val : int
        Rank of the subspace formed by selected rows.
    depI : boolean array of shape (n,)
        For each index k not in I, depI[k] = True if that row is in the span of the selected subspace.
    depJ : boolean array of shape (n,)
        For each index k not in J, depJ[k] = True if that column is in the span of the selected subspace.
    """
    # Ensure boolean
    I = np.asarray(I, dtype=bool)
    J = np.asarray(J, dtype=bool)
    M = np.asarray(M, dtype=bool)
    n = M.shape[0]

    sel_I_idx = np.flatnonzero(I)
    sel_J_idx = np.flatnonzero(J)
    num_I = sel_I_idx.size
    num_J = sel_J_idx.size

    cand_I_idx = np.flatnonzero(~I)
    cand_J_idx = np.flatnonzero(~J)
    num_cand_I = cand_I_idx.size
    num_cand_J = cand_J_idx.size

    # If no rows were selected, rank=0, trivially no dependence
    if num_I == 0 and num_J == 0:
        return 0, np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    # -----------------------
    # Build TOP block (A):
    # -----------------------
    # 1) Identity for sel_I_idx
    top_I = np.zeros((num_I, n), dtype=bool)
    if num_I > 0:
        top_I[np.arange(num_I), sel_I_idx] = True

    # 2) M cols (transposed) for sel_J_idx
    top_J = M[:, sel_J_idx].T if num_J > 0 else np.zeros((0, n), dtype=bool)

    A = np.vstack((top_I, top_J))  # shape = (num_I + num_J, n)
    top_count = A.shape[0]

    # -----------------------
    # Build BOTTOM block (C):
    # -----------------------
    # 1) Identity for cand_I_idx
    bot_I = np.zeros((num_cand_I, n), dtype=bool)
    if num_cand_I > 0:
        bot_I[np.arange(num_cand_I), cand_I_idx] = True

    # 2) M cols (transposed) for cand_J_idx
    bot_J = M[:, cand_J_idx].T if num_cand_J > 0 else np.zeros((0, n), dtype=bool)

    C = np.vstack((bot_I, bot_J))  # shape = (num_cand_I + num_cand_J, n)
    bottom_count = C.shape[0]

    # -----------------------
    # Build single big matrix
    # -----------------------
    B = np.vstack((A, C))  # shape = (top_count + bottom_count, n)
    row_count, col_count = B.shape

    # Single elimination loop
    pivot_row = 0
    for col in range(col_count):
        # If we've used all pivot rows in the top block, we can stop
        if pivot_row >= top_count:
            break

        # Instead of np.nonzero(...), use argmax to find the first True in pivot region
        # The slice is [pivot_row:top_count, col]
        subcol = B[pivot_row:top_count, col]
        pivot_candidate_offset = subcol.argmax()  # returns first index of max in subcol
        # Check if that pivot is actually 1
        if not subcol[pivot_candidate_offset]:
            continue  # no pivot in this column in the top region

        # Absolute row index of the pivot
        pivot_candidate = pivot_row + pivot_candidate_offset

        # Swap if needed
        if pivot_candidate != pivot_row:
            # In-place row swap
            B[[pivot_row, pivot_candidate]] = B[[pivot_candidate, pivot_row]]

        # XOR-eliminate all rows below pivot_row (including bottom block)
        if pivot_row < row_count - 1:
            col_slice = B[pivot_row + 1:, col]  # This is the col below the pivot
            # Find rows that have a 1 in this col
            # (We do this once: it's typically faster than repeated any() calls.)
            ones_below = np.flatnonzero(col_slice)
            if ones_below.size > 0:
                # These row indices are relative to pivot_row+1
                rows_to_xor = ones_below + (pivot_row + 1)
                B[rows_to_xor] ^= B[pivot_row]

        pivot_row += 1

    # pivot_row is the rank in the top portion
    rank_val = pivot_row

    # -------------------------------------------
    # The bottom block is automatically reduced.
    # Check for zero rows => dependent
    # -------------------------------------------
    depI = np.zeros(n, dtype=bool)
    depJ = np.zeros(n, dtype=bool)

    if bottom_count > 0 and rank_val > 0:
        # The bottom block in B is B[top_count : top_count + bottom_count, :]
        candidate_block = B[top_count:]
        # 1) The first num_cand_I belong to cand_I_idx
        # 2) The next num_cand_J belong to cand_J_idx
        if num_cand_I > 0:
            cand_I_part = candidate_block[:num_cand_I]
            dep_mask_I = ~cand_I_part.any(axis=1)
            depI[cand_I_idx[dep_mask_I]] = True

        if num_cand_J > 0:
            cand_J_part = candidate_block[num_cand_I:]
            dep_mask_J = ~cand_J_part.any(axis=1)
            depJ[cand_J_idx[dep_mask_J]] = True

    return rank_val, depI, depJ

def expandWithinRank(I,J,stateprereqs):
    # rank,depI,depJ = gf2_rank_subset(I,J,stateprereqs)
    rank,depI,depJ = native.gf2_rank_subset(np.asarray(I, dtype=np.bool),np.asarray(J, dtype=np.bool),np.asarray(stateprereqs, dtype=np.bool))
    return rank,np.maximum(I,depI),np.maximum(J,depJ)

def scoreIJ(IJ):
    I,J = IJ[:128],IJ[128:]
    rank,fullI,fullJ = expandWithinRank(I,J,state_matrix)
    Iscores = stats_table_A[np.arange(64),fullI@index_map_A]
    Jscores = stats_table_B[np.arange(64),fullJ@index_map_B]
    fullIJ = np.concatenate([fullI,fullJ])
    return Iscores.sum() + Jscores.sum(), fullIJ, rank

def generate_random_binary(n_ones, inner_bound, outer_bound, rng):
    """
    Generate a random 1D binary numpy array (0s and 1s) with the following constraints:
      - The output has length L (automatically determined from the bounds arrays).
      - Exactly `n_ones` ones are present in the output.
      - For every index i where inner_bound[i] == 1, output[i] must be 1.
      - For every index i where outer_bound[i] == 0, output[i] must be 0.
      
    Parameters:
        n_ones (int): The exact number of ones required in the output.
        inner_bound (array-like): An L-length binary array. Each 1 indicates that position must be 1.
        outer_bound (array-like): An L-length binary array. Each 0 indicates that position must be 0.
        rng (np.random.Generator): A seeded numpy PRNG object.
    
    Returns:
        np.ndarray: The resulting binary array.
    
    Raises:
        AssertionError: If the bounds arrays are of unequal length, have conflicting requirements,
                        or if `n_ones` is not possible given the forced ones and zeros.
    """
    inner_bound = np.asarray(inner_bound)
    outer_bound = np.asarray(outer_bound)
    
    # Check that the two bounds arrays have equal length.
    L = inner_bound.shape[0]
    assert L == outer_bound.shape[0], "Bound arrays must have the same length."
    
    # Check for conflicting requirements: inner_bound forces a 1 where outer_bound forces a 0.
    conflict = (inner_bound == 1) & (outer_bound == 0)
    assert not np.any(conflict), "Conflicting requirements: inner_bound forces one where outer_bound forces zero."
    
    # Count forced ones and forced zeros.
    forced_ones = np.sum(inner_bound == 1)
    forced_zeros = np.sum(outer_bound == 0)
    
    # Check that the requested n_ones is possible.
    # The maximum number of ones possible is L minus the forced zeros.
    max_possible_ones = L - forced_zeros
    assert forced_ones <= n_ones <= max_possible_ones, (
        "n_ones not possible given the forced ones and zeros."
    )
    
    # Initialize the output array.
    output = np.empty(L, dtype=int)
    
    # Fill forced positions.
    forced_1_positions = (inner_bound == 1)
    forced_0_positions = (outer_bound == 0)
    output[forced_1_positions] = 1
    output[forced_0_positions] = 0
    
    # Determine free positions (not forced to 1 or 0).
    free_positions = ~(forced_1_positions | forced_0_positions)
    free_indices = np.where(free_positions)[0]
    
    # Calculate how many additional ones need to be placed.
    additional_ones = n_ones - forced_ones
    
    # Randomly choose free positions to set to 1.
    if additional_ones > 0:
        selected_indices = rng.choice(free_indices, size=additional_ones, replace=False)
        output[selected_indices] = 1
    
    # Set remaining free positions to 0.
    free_indices_set = set(free_indices)
    if additional_ones > 0:
        selected_set = set(selected_indices)
    else:
        selected_set = set()
    remaining_free = np.array(list(free_indices_set - selected_set))
    output[remaining_free] = 0
    
    return output

def check_bounds(IJ,inner_bound,outer_bound):
    return (np.minimum(IJ,inner_bound) == inner_bound).all() and (np.maximum(IJ,outer_bound) == outer_bound).all()

def _solve_u(transition_progress: float, tol: float = 1e-8, max_iter: int = 100) -> float:
    """
    Solve for u in: u/(1 - exp(-u)) = 1/transition_progress.
    Since u is expected to be small, we use a simple binary search.
    """
    target = 1 / transition_progress
    lo = 1e-8
    hi = 1.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        value = mid / (1 - np.exp(-mid))
        if abs(value - target) < tol:
            return mid
        if value < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

def estimate_progress(
    outer_iter_running_total: int,
    inner_iter_running_total: int,
    inner_iter: int,
    inner_iter_max: int,
    transition_progress: float = 0.90  # You can set this to 0.97, for example.
) -> float:
    """
    Estimate progress as a float in [0.0, 1.0), strictly increasing with work done.
    
    The cumulative work is defined as:
        x = inner_iter_running_total + inner_iter + epsilon * outer_iter_running_total
    where epsilon is a very small constant to ensure that increases in outer iterations
    always nudge the progress upward.
    
    Let T = inner_iter_max.
    
    The function is defined piecewise:
    
      For x <= T:
          f(x) = transition_progress * [1 - exp(-a*x)] / [1 - exp(-a*T)]
      For x > T:
          f(x) = transition_progress + (1 - transition_progress) * [1 - exp(-b*(x-T))]
    
    The parameter a is defined as:
          a = u/T
    where u is the unique positive solution to:
          u/(1 - exp(-u)) = 1/transition_progress.
          
    The parameter b is chosen to ensure continuity of the derivative at x = T:
          b = exp(-u) / (T * (1 - transition_progress))
    
    This design ensures:
      - For small x (i.e. x << T), f(x) â‰ˆ x/T.
      - At x = T, f(x) equals transition_progress.
      - For x > T, f(x) increases smoothly toward 1, but remains strictly less than 1.
      
    The function is strictly monotonically increasing with respect to any increase in
    outer_iter_running_total, inner_iter_running_total, or inner_iter.
    """
    epsilon = 1e-6  # tiny contribution from outer iterations
    # Cumulative work done.
    x = inner_iter_running_total + inner_iter + epsilon * outer_iter_running_total
    T = inner_iter_max

    # Solve for u given the desired transition_progress.
    u = _solve_u(transition_progress)
    a = u / T

    if x <= T:
        progress = transition_progress * (1 - np.exp(-a * x)) / (1 - np.exp(-a * T))
    else:
        # Compute b so that the derivative at x=T is continuous.
        b = np.exp(-u) / (T * (1 - transition_progress))
        progress = transition_progress + (1 - transition_progress) * (1 - np.exp(-b * (x - T)))
    return min(progress,0.99999)

def write_progress_bar(value):
    with open('boinc_frac', 'w', newline='\n') as file:
        file.write(f"{value:.5f}")
        file.flush()

def greedy_optimize(IJ: np.ndarray, scoreIJ) -> (np.ndarray, float):
    """
    Greedy optimization of a binary vector IJ, which must have a fixed number of 1s.
    At each step, every possible swap of one 1 with one 0 is tested. The swap that 
    results in the highest score (as returned by scoreIJ) is accepted. The process stops 
    when no swap improves the score.

    Parameters:
    - IJ (np.ndarray): A binary numpy array.
    - scoreIJ (callable): A function that accepts IJ and returns a score (float).

    Returns:
    - best_IJ (np.ndarray): The optimized binary vector.
    - best_score (float): The score corresponding to best_IJ.
    """
    # Ensure we work on a copy of IJ
    current_IJ = IJ.copy()
    best_score,_,_ = scoreIJ(current_IJ)
    
    improved = True
    itercount = 0
    inner_iter_count = 0
    progress = estimate_progress(outer_iter_running_total, inner_iter_running_total, inner_iter_count, max_inner_iter)
    while improved:
        print(itercount,f'{best_score:.6f}',f'{progress:.5f}')
        itercount += 1
        improved = False
        # Get the indices of 1s and 0s
        _,fullIJ,_ = scoreIJ(current_IJ)
        # the 1s we are allowed to swap
        ones_idx = np.flatnonzero(np.minimum(current_IJ, 1-inner_bound))
        # the 0s we are allowed to swap
        zeros_idx = np.flatnonzero(np.minimum(1-fullIJ, outer_bound))
        
        # Variables to track the best candidate swap in this iteration
        candidate_score = best_score
        best_swap = None
        
        # Try every swap: swap a 1 at index i with a 0 at index j.
        for i in ones_idx:
            for j in zeros_idx:
                # Swap in place
                current_IJ[i] = 0
                current_IJ[j] = 1
                assert check_bounds(current_IJ,inner_bound,outer_bound)
                new_score,_,_ = scoreIJ(current_IJ)
                
                # Revert the swap immediately to avoid unnecessary copy overhead
                current_IJ[i] = 1
                current_IJ[j] = 0
                
                if new_score > candidate_score:
                    candidate_score = new_score
                    best_swap = (i, j)

                if inner_iter_count % 5000 == 0:
                    progress = estimate_progress(outer_iter_running_total, inner_iter_running_total, inner_iter_count, max_inner_iter)
                    write_progress_bar(progress)
                    # print('progress bar updated')
                
                inner_iter_count += 1
        
        # If a beneficial swap was found, apply it and update best_score
        if best_swap is not None:
            i, j = best_swap
            current_IJ[i] = 0
            current_IJ[j] = 1
            best_score = candidate_score
            improved = True
            # Continue the loop with the updated IJ vector.
    
    return current_IJ, best_score, inner_iter_count

def binary_to_hex_chunked(binary_str):
    # Pad the binary string from the left so its length is a multiple of 4
    padded = binary_str.zfill((len(binary_str) + 3) // 4 * 4)
    
    hex_str = ''
    for i in range(0, len(padded), 4):
        chunk = padded[i:i+4]
        hex_digit = hex(int(chunk, 2))[2:]  # Convert to hex and remove '0x'
        hex_str += hex_digit.lower()        # Use .upper() if uppercase is preferred
    
    return hex_str

def write_output(best_score_overall,best_IJ_overall,
                 config_file,max_inner_iter,rng_seed,outer_iter_running_total,inner_iter_running_total):
    with open('output.txt', 'w', newline='\n') as file:
        file.write(f'{best_score_overall:.6f}')
        file.write('\n')
        file.write(binary_to_hex_chunked(''.join([str(int(i)) for i in best_IJ_overall])))
        file.write('\n')
        file.write(f'{config_file} {max_inner_iter} {rng_seed}\n')
        file.write(f'{outer_iter_running_total} {inner_iter_running_total}\n')
        file.flush()

if __name__ == '__main__':
    print(sys.argv)
    config_file = sys.argv[1] # 'config-001-hixorlo-rank100.npz'
    max_inner_iter = int(sys.argv[2]) # 500000
    rng_seed = int(sys.argv[3]) # 123
    local_config_file = None
    try:
        local_config_file = sys.argv[4] # input.npz
    except:
        pass
        
    if local_config_file:
        config = np.load(local_config_file)
    else:
        config = np.load(config_file)
    
    state_matrix = config['state_matrix']
    indexing_grid_A = config['indexing_grid']
    stats_table_A = config['stats_table'].astype('float64')
    max_rank = config['max_rank']
    inner_bound = config['inner_bound']
    outer_bound = config['outer_bound']

    indexing_grid_B = indexing_grid_A
    if 'indexing_grid_B' in config:
        indexing_grid_B = config['indexing_grid_B']

    stats_table_B = stats_table_A
    if 'stats_table_B' in config:
        stats_table_B = config['stats_table_B'].astype('float64')
    
    assert indexing_grid_A.sum(0).std() == 0
    n_statebits = indexing_grid_A.sum(0)[0]
    index_map_A = indexing_grid_A[:,:64].copy()
    for col in range(64):
        index_map_A[np.where(indexing_grid_A[:,col])[0],col] = 2**np.flip(np.arange(n_statebits))
    
    assert indexing_grid_B.sum(0).std() == 0
    n_statebits = indexing_grid_B.sum(0)[0]
    index_map_B = indexing_grid_B[:,:64].copy()
    for col in range(64):
        index_map_B[np.where(indexing_grid_B[:,col])[0],col] = 2**np.flip(np.arange(n_statebits))
    
    inner_iter_running_total = 0
    outer_iter_running_total = 0
    best_IJ_overall = np.nan
    best_score_overall = np.nan
    rng_id = "my_rng"
    print("Made it here")
    native.init_rng_state(rng_seed, rng_id)
    print("Here?")
    # if os.path.isfile('checkpoint.npz'):
    #     checkpoint = np.load('checkpoint.npz', allow_pickle=True)
    #     inner_iter_running_total = checkpoint['inner_iter_running_total']
    #     outer_iter_running_total = checkpoint['outer_iter_running_total']
    #     best_IJ_overall = checkpoint['best_IJ_overall']
    #     best_score_overall = checkpoint['best_score_overall']
    #     print(type(checkpoint['rng_state']))
    #     print(checkpoint['rng_state'])
    #     native.set_rng_state(rng_id, checkpoint['rng_state'])
    # else:
    #     np.savez_compressed(
    #         'checkpoint.npz',
    #         inner_iter_running_total = inner_iter_running_total,
    #         outer_iter_running_total = outer_iter_running_total,
    #         best_IJ_overall = best_IJ_overall,
    #         best_score_overall = best_score_overall,
    #         rng_state = native.get_rng_state(rng_id)
    #     )
    IJ = np.array([], dtype=np.int64)
    while inner_iter_running_total < max_inner_iter:
    
        IJ = native.generate_random_binary(max_rank, inner_bound, outer_bound, rng_id)
        optimized_IJ, optimized_score, inner_iter_count = greedy_optimize(IJ,scoreIJ)
        
        if not np.isfinite(best_score_overall) or optimized_score > best_score_overall:
            best_score_overall = optimized_score
            best_IJ_overall = optimized_IJ
        
        inner_iter_running_total += inner_iter_count
        outer_iter_running_total += 1
        # checkpoint_rng_array = np.array(native.get_rng_state(rng_id), dtype=object)
        # np.savez_compressed(
        #     'checkpoint.npz',
        #     inner_iter_running_total = inner_iter_running_total,
        #     outer_iter_running_total = outer_iter_running_total,
        #     best_IJ_overall = best_IJ_overall,
        #     best_score_overall = best_score_overall,
        #     rng_state = checkpoint_rng_array
        # )
    
    write_output(best_score_overall,best_IJ_overall,config_file,max_inner_iter,rng_seed,outer_iter_running_total,inner_iter_running_total)

    with open('boinc_finish_called', 'w', newline='\n') as file:
        file.write('finished\n')
        file.flush()

    write_progress_bar(1.0)
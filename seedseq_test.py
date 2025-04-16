import numpy as np

sq1 = np.random.SeedSequence(0)
print(sq1.pool)
print(sq1.generate_state(4, np.uint64))
print(sq1.generate_state(4, np.uint64))
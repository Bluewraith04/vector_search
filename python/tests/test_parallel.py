import vector_search
import time
import random

dim = 64
count = 20000
print(f"Generating {count} vectors...")
data = [[random.random() for _ in range(dim)] for _ in range(count)]

# --- Adding vectors in sequence ---
print("\n--- Testing Sequential Add ---")
index_seq = vector_search.VectorIndex(dim)
start = time.time()
for vec in data:
    index_seq.add(vec)
print(f"Sequential Time: {time.time() - start:.4f}s")

# --- Adding vectors in parallel ---
print("\n--- Testing Parallel Add ---")
index_par = vector_search.VectorIndex(dim)
start = time.time()
index_par.add_parallel(data)
print(f"Parallel Time: {time.time() - start:.4f}s")
import vector_search
import random
import time
import math

def generate_vector(dim):
    return [random.random() for _ in range(dim)]

def main():
    dim = 64
    num_vectors = 1000
    
    print(f"1. Initializing Index (Dim={dim})...")
    index = vector_search.VectorIndex(dim)

    # 2. Build the Graph
    # We add vectors one by one. The `add` logic connects them.
    vectors = []
    print(f"2. Adding {num_vectors} vectors...")
    start = time.time()
    for i in range(num_vectors):
        vec = generate_vector(dim)
        vectors.append(vec)
        index.add(vec)
    print(f"   -> Added in {time.time() - start:.4f}s")

    # 3. Create a Query
    query = generate_vector(dim)

    # 4. Compare Results
    print("\n3. Comparing Search Methods:")
    
    # Method A: Brute Force (The slow, perfect way)
    # (Assuming you still have the `search` method exposed)
    start = time.time()
    results_brute = index.search(query, k=1) 
    t_brute = time.time() - start
    best_idx_brute = results_brute[0][0]
    
    # Method B: Graph Search (HNSW)
    start = time.time()
    # Note: Our search_graph currently returns (idx, dist) tuple, not a list
    best_idx_graph, dist_graph = index.search_graph(query, 30) 
    t_graph = time.time() - start

    print(f"   [Brute Force] Found Index: {best_idx_brute} | Time: {t_brute*1000:.4f} ms")
    print(f"   [Graph HNSW] Found Index: {best_idx_graph} | Time: {t_graph*1000:.4f} ms")

    if best_idx_brute == best_idx_graph:
        print("\n✅ SUCCESS: Graph found the exact same vector!")
    else:
        print("\n⚠️  Note: Graph found a different vector.")
        print("    This is normal for 'Approximate' search.")
        print("    It usually means the graph got stuck in a local minimum.")

if __name__ == "__main__":
    main()
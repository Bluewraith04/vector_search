import vector_search
import random

dim = 4
index = vector_search.VectorIndex(dim)

index.add([0.0, 0.0, 0.0, 0.0])
index.add([1.0, 1.0, 1.0, 1.0])
index.add([0.0, 0.0, 0.0, 0.1])

query = [0.0, 0.0, 0.0, 0.05]
results = index.search(query, k=3)

print("Results (Index, Squared Distance)")
for idx, dist in results:
    print(f"Index: {idx}, Distance: {dist}")
    
    



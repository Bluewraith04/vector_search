import vector_search
import os

index = vector_search.VectorIndex(64)
index.add([1.0] * 64)

print("Saving Index...")
index.save("test_db.bin")

print("Loading Index...")
loaded_index = vector_search.VectorIndex.load("test_db.bin")

result = loaded_index.search_graph([1.0] * 64, ef=10)
print(f"Found on loaded index: {result}")

os.remove("test_db.bin")
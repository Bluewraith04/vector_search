from .vector_search import VectorIndex as _RustIndex
import math
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
    
def normalize(vec: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x*x for x in vec))
    if magnitude == 0: return vec
    return [x / magnitude for x in vec]

class VectorIndex(_RustIndex):
    def add_parallel(self, vectors, chunk_size = 5000):
        total = len(vectors)
        
        for i in tqdm(range(0, total, chunk_size), desc="Indexing"):
            chunk = vectors[i : i + chunk_size]
            super().add_parallel(chunk)


__all__ = ["VectorIndex"]

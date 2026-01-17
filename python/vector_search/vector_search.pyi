class VectorIndex:
    def __init__(self, dim: int) -> None: ...
    
    def add(self, vector: list[float]) -> None:
        """
        Adds a new vector to the database
        """
        ...
    
    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        """
        Returns a list of (index, distance) tuples.
        """
        ...
        
    def save(self, path:str) -> None: ...
    
    @staticmethod
    def load(path: str) -> "VectorIndex": ...
    
    def add_parallel(self, vectors: list[list[float]], chunk_size: int = ...) -> None:
        """
        Adds a list of vectors in parallel using all CPU cores.
        Use this for bulk loading data.
        """
        ...
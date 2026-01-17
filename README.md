
# Vector Search 

[![Rust](https://img.shields.io/badge/built_with-Rust-dca282.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance, embedded Vector Database built in Rust, exposing a seamless Python API.

It combines **SIMD hardware acceleration**, **HNSW Graph algorithms**, and **Memory Arenas** to achieve search speeds orders of magnitude faster than naive implementations.

## 🚀 Performance

Benchmarks run on 20,000 vectors (64-dim) on an 8-core machine:

| Operation | Method | Time | Speedup |
| :--- | :--- | :--- | :--- |
| **Indexing** | Sequential `add()` | 51.78s | 1x |
| | **Parallel `add_parallel()`** | **9.95s** | **~5x** ⚡ |
| **Search** | Brute Force `search()` | 77.3ms | 1x |
| | **Graph Search `search_graph()`** | **7.2ms** | **~11x** ⚡ |

## ✨ Key Features

* **Architecture:** Hybrid Rust Core / Python Shell using `PyO3`.
* **Algorithms:**
    * **HNSW (Hierarchical Navigable Small World):** For $O(\log N)$ approximate nearest neighbor search.
    * **Beam Search:** Configurable `ef` parameter to balance speed vs. accuracy.
* **Systems Engineering:**
    * **SIMD Acceleration:** Hand-written AVX2 intrinsics for Euclidean distance.
    * **Memory Arena:** Contiguous memory layout to minimize cache misses.
    * **Parallelism:** Multithreaded bulk indexing using `Rayon`.
* **Usability:**
    * Built-in persistence (`save` / `load`) using binary serialization.
    * Native Python progress bars (`tqdm`) for long-running tasks.
    * Fully type-hinted for VS Code / PyCharm autocomplete.

## 📦 Installation

Ensure you have Rust installed (`cargo`).

```bash
# Clone the repository
git clone [https://github.com/your-username/vector_search.git](https://github.com/your-username/vector_search.git)
cd vector_search

# Install directly into your active Python environment
pip install .

```

*Note: The build process automatically compiles the Rust core using `maturin` and installs the Python wrapper.*

## ⚡ Quick Start

```python
import vector_search
import random

# 1. Initialize the Index
# Dimension must match your data (e.g., 768 for OpenAI embeddings)
dim = 64
index = vector_search.VectorIndex(dim)

# 2. Generate Dummy Data
data = [[random.random() for _ in range(dim)] for _ in range(10000)]
query = [random.random() for _ in range(dim)]

# 3. Add Data (Uses all CPU cores)
print("Indexing data...")
index.add_parallel(data) 

# 4. Search
# ef=30 means we explore 30 candidates in the graph. 
# Higher ef = more accurate, but slower.
idx, dist = index.search_graph(query, ef=30)

print(f"Found nearest neighbor at index {idx} with distance {dist:.4f}")

# 5. Save & Load
index.save("my_index.bin")
loaded_index = vector_search.VectorIndex.load("my_index.bin")

```

## 🛠️ Development

If you want to modify the Rust core:

```bash
# Install development tools
pipx install maturin

# Compile and install in "editable" mode
maturin develop

```

Run the test suite:

```bash
python python/tests/test_parallel.py

```

## 🧠 Architecture

The project is structured to separate high-performance code from user-facing API code:

```text
vector_search/
├── src/                  # RUST CORE (The Engine)
│   ├── core/
│   │   ├── metrics.rs    # AVX2 SIMD Math
│   │   ├── node.rs       # Memory Layout
│   │   ├── heap.rs       # Priority Queue
│   │   └── index.rs      # HNSW Graph Logic
│   └── lib.rs            # PyO3 Bindings
|
├── python/               #  PYTHON SHELL (The API)
│   └── vector_search/
│       ├── __init__.py   # Wrapper with tqdm & utilities
│       └── vector_search.pyi # Type hints

```

## 📜 License

MIT License. Feel free to use this in your own projects.



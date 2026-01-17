use pyo3::prelude::*;
use std::collections::BinaryHeap;

use super::node::Node;
use super::heap::Candidate;
use super::metrics::compute_distance;

use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

use rayon::prelude::*;

#[pyclass(subclass)]
#[derive(Serialize, Deserialize)]
pub struct VectorIndex {
    pub arena: Vec<Node>,
    pub dimension: usize,
    pub entry_point: Option<usize>, // The index of the entry point for hnsw graph search
}


// --- Rust-Only Methods ---
impl VectorIndex {
    pub fn search_greedy(&self, query: &[f32], start_node: usize) -> usize {
        let mut current_node = start_node;
        let mut current_dist = compute_distance(query, &self.arena[current_node].vector);

        loop {
            let mut changed = false;

            for &neighbor_idx in &self.arena[current_node].neighbors {
                let dist = compute_distance(query, &self.arena[neighbor_idx].vector);

                if dist < current_dist {
                    current_dist = dist;
                    current_node = neighbor_idx;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        current_node
    }

    pub fn search_beam(&self, query: &[f32], start_node: usize, ef: usize) -> usize {
        let mut candidates = BinaryHeap::new();
        let start_dist = compute_distance(query, &self.arena[start_node].vector);

        candidates.push(Candidate {distance: start_dist, index: start_node});

        let mut visited = vec![false; self.arena.len()];
        visited[start_node] = true;

        let mut nearest_idx = start_node;
        let mut nearest_dist = start_dist;

        while let Some(current) = candidates.pop() {
            for &neighbor_idx in &self.arena[current.index].neighbors {
                if visited[neighbor_idx] { continue; }
                visited[neighbor_idx] = true;

                let dist = compute_distance(query, &self.arena[neighbor_idx].vector);
                if dist < nearest_dist {
                    nearest_dist = dist;
                    nearest_idx = neighbor_idx;
                    candidates.push(Candidate { distance: dist, index: neighbor_idx});
                } else if candidates.len() < ef {
                    candidates.push(Candidate {distance: dist, index: neighbor_idx});
                }
            }
        }
        nearest_idx
    }
}


// --- Python API methods ---
#[pymethods]
impl VectorIndex {
    #[new]
    fn new(dim: usize) -> Self {
        VectorIndex {
            arena: Vec::new(),
            dimension: dim,
            entry_point: None
        }
    }

    fn add(&mut self, vector: Vec<f32>) -> PyResult<()> {
        if vector.len() != self.dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Vector dimension mismatch. Expected {}, got {}", self.dimension, vector.len())
            ));
        }

        let new_idx = self.arena.len();

        let mut neighbors = Vec::new();

        if let Some(entry_idx) = self.entry_point {
            let nearest_idx = self.search_beam(&vector, entry_idx, 10);

            neighbors.push(nearest_idx);

            self.arena[nearest_idx].neighbors.push(new_idx);
        } else {
            self.entry_point = Some(new_idx);
        }

        let new_node = Node {
            vector,
            neighbors
        };

        self.arena.push(new_node);

        Ok(())
    }

    fn add_parallel(&mut self, vectors: Vec<Vec<f32>>) -> PyResult<()> {
        let chunk_size = 1000;
        let mut current_entry_point = self.entry_point;

        for chunk in vectors.chunks(chunk_size) {
            let nearest_neighbors: Vec<usize> = chunk.par_iter().map(
                |vector| {
                    if let Some(entry_idx) = current_entry_point {
                        self.search_beam(vector, entry_idx, 10)
                    } else {
                        0
                    }
                }
            ).collect();

            for (i, vector) in chunk.iter().enumerate() {
                let vector = vector.clone();
                let nearest_idx = nearest_neighbors[i];
                let new_idx = self.arena.len();

                let mut neighbors = Vec::new();

                if self.arena.is_empty() {
                    self.entry_point = Some(new_idx);
                    current_entry_point = Some(new_idx);
                } else {
                    neighbors.push(nearest_idx);
                    self.arena[nearest_idx].neighbors.push(new_idx);
                }
                
                self.arena.push(Node {vector, neighbors});
            }
        }

        Ok(())
    }

    fn search(&self, query: Vec<f32>, k:usize) -> PyResult<Vec<(usize, f32)>> {
        if query.len() != self.dimension {
            return Err(pyo3::exceptions::PyValueError::new_err("Dimension Mismatch"));
        }

        let mut scores: Vec<(usize, f32)> = self.arena.iter().enumerate().map(|(index, node)| {
            let dist = compute_distance(&query, &node.vector); 
            (index, dist)}).collect();

        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        scores.truncate(k);

        Ok(scores)
    }

    fn search_graph(&self, query: Vec<f32>, ef: usize) -> PyResult<(usize, f32)> {
        if let Some(entry_point) = self.entry_point { 
            let best_idx = self.search_beam(&query, entry_point, ef);
            let dist = compute_distance(&query, &self.arena[best_idx].vector);

            Ok((best_idx, dist))
        } else {
            Ok((0, f32::MAX))
        }
    }

    fn save(&self, path: String) -> PyResult<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        bincode::serialize_into(writer, self).map_err(
            |e| pyo3::exceptions::PyIOError::new_err(e.to_string())
        )?;

        Ok(())
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let index: VectorIndex = bincode::deserialize_from(reader).map_err(
            |e| pyo3::exceptions::PyIOError::new_err(e.to_string())
        )?;

        Ok(index)
    }
}

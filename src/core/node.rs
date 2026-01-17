use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub vector: Vec<f32>,
    pub neighbors: Vec<usize> // list of connections to other nodes
}

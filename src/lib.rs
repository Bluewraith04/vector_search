use pyo3::prelude::*;

mod core;

use core::index::VectorIndex;

#[pymodule]
fn vector_search(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VectorIndex>()?;
    Ok(())
}
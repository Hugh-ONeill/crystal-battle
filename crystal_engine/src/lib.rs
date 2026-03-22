// Crystal Battle engine -- Rust implementation with PyO3 bindings

pub mod actions;
pub mod batch;
pub mod batch_nn;
pub mod battle;
pub mod damage;
pub mod data;
pub mod events;
pub mod moves;
pub mod obs;
pub mod player;
pub mod pokemon;
pub mod pybridge;
pub mod search;
pub mod stat_stages;
pub mod status;
pub mod turn_engine;
pub mod types;

use pyo3::prelude::*;

#[pymodule]
fn crystal_engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    pybridge::register(m)?;
    Ok(())
}

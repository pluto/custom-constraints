#![doc = include_str!("../README.md")]
#![warn(missing_docs, clippy::missing_docs_in_private_items)]

//! Custom Constraints provides an implementation of Customizable Constraint Systems (CCS),
//! a framework for zero-knowledge proof systems.
//!
//! This crate provides tools for:
//! - Building arithmetic circuits with degree bounds
//! - Converting circuits to CCS form
//! - Optimizing circuit representations
//! - Working with sparse matrix operations
//!
//! The core components are:
//! - [`Circuit`](circuit::Circuit): For constructing and manipulating arithmetic circuits
//! - [`CCS`](ccs::CCS): The customizable constraint system representation
//! - [`SparseMatrix`](matrix::SparseMatrix): Efficient sparse matrix operations

use std::fmt::{self, Display, Formatter};

use ark_ff::Field;
#[cfg(test)] use mock::F17;
#[cfg(all(target_arch = "wasm32", test))]
use wasm_bindgen_test::wasm_bindgen_test;

pub mod ccs;
pub mod circuit;
pub mod matrix;

#[cfg(test)]
mod mock {
  //! Test utilities including a simple finite field implementation.
  use ark_ff::{Fp, MontBackend, MontConfig};

  #[allow(unexpected_cfgs)]
  #[derive(MontConfig)]
  #[modulus = "17"]
  #[generator = "3"]
  pub struct F17Config;
  /// A finite field of order 17 used for testing.
  pub type F17 = Fp<MontBackend<F17Config, 1>, 1>;
}

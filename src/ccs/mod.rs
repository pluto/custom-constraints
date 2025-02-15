//! Implements the Customizable Constraint System (CCS) format.
//!
//! A CCS represents arithmetic constraints through a combination of matrices
//! and multisets, allowing efficient verification of arithmetic computations.
//!
//! The system consists of:
//! - A set of sparse matrices representing linear combinations
//! - Multisets defining which matrices participate in each constraint
//! - Constants applied to each constraint term

use std::marker::PhantomData;

use matrix::SparseMatrix;

use super::*;

pub mod generic;
pub mod plonkish;

pub trait CCSType<F> {
  type Selectors: Default;
}

#[derive(Clone, Debug, Default)]
pub struct Generic<F>(PhantomData<F>);
impl<F: Default> CCSType<F> for Generic<F> {
  type Selectors = F;
}

/// A Customizable Constraint System over a field F.
#[derive(Debug, Default)]
pub struct CCS<C: CCSType<F>, F: Field> {
  /// Constants for each constraint term
  pub selectors: Vec<C::Selectors>,
  /// Sets of matrix indices for Hadamard products
  pub multisets: Vec<Vec<usize>>,
  /// Constraint matrices
  pub matrices: Vec<SparseMatrix<F>>,
}

impl<C: CCSType<F> + Default, F: Field> CCS<C, F> {
  /// Creates a new empty CCS.
  pub fn new() -> Self {
    Self::default()
  }
}

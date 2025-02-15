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

/// A trait for configuring different types of Customizable Constraint Systems (CCS).
///
/// This trait allows different CCS variants to specify their selector types.
/// Different CCS designs can use different types of selectors:
/// - Generic CCS uses single field elements as selectors
/// - Plonkish CCS uses vectors of field elements for multi-constraint support
/// - Other variants might use matrices or more complex structures
///
/// The selector type must implement Default to provide a zero/empty value
/// when initializing a new CCS.
pub trait CCSType<F> {
  /// The type of selectors used in this CCS variant.
  type Selectors: Default;
}

/// A type marker for the standard/generic CCS format with scalar constants as "selectors".
#[derive(Clone, Debug, Default)]
pub struct Generic<F>(PhantomData<F>);

impl<F: Default> CCSType<F> for Generic<F> {
  /// For Generic CCS, selectors are just single field elements
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
  pub matrices:  Vec<SparseMatrix<F>>,
}

impl<C: CCSType<F> + Default, F: Field> CCS<C, F> {
  /// Creates a new empty CCS.
  pub fn new() -> Self { Self::default() }
}

use matrix::SparseMatrix;

use super::*;

pub struct CCS<F: Field> {
  /// `m` in the paper
  rows:                usize,
  /// `n` in the paper
  cols:                usize,
  /// `N` in the paper
  nonzero_entries:     usize,
  /// `t` in the paper
  number_of_matrices:  usize,
  /// `q` in the paper
  number_of_multisets: usize,
  matrices:            Vec<SparseMatrix<F>>,
}

#[cfg(test)]
mod tests {
  use super::*;
}

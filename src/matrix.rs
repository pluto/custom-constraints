use ark_ff::Field;

use super::*;

pub struct SparseMatrix<F> {
  row_offsets: Vec<usize>,
  col_indices: Vec<usize>,
  values:      Vec<F>,
  num_cols:    usize,
}

impl<F: Field> SparseMatrix<F> {
  pub fn new(
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<F>,
    num_cols: usize,
  ) -> Self {
    // Add validation here
    Self { row_offsets, col_indices, values, num_cols }
  }

  pub fn mul_vector(&self, rhs: &[F]) -> Vec<F> {
    assert_eq!(rhs.len(), self.num_cols, "Invalid vector length");
    let mut result = vec![F::ZERO; self.row_offsets.len() - 1];

    for row in 0..self.row_offsets.len() - 1 {
      let start = self.row_offsets[row];
      let end = self.row_offsets[row + 1];

      result[row] = self.values[start..end]
        .iter()
        .zip(&self.col_indices[start..end])
        .map(|(v, &c)| *v * rhs[c])
        .sum();
    }

    result
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_sparse_matrix_multiplication() {
    // Let's create this matrix:
    // [2 0 1]
    // [0 3 0]
    // [4 0 5]

    let row_offsets = vec![0, 2, 3, 5]; // Points to starts of rows
    let col_indices = vec![0, 2, 1, 0, 2]; // Column indices for non-zero elements
    let values = vec![
      F17::from(2),
      F17::from(1), // First row: 2 and 1
      F17::from(3), // Second row: just 3
      F17::from(4),
      F17::from(5), // Third row: 4 and 5
    ];

    let matrix = SparseMatrix::new(row_offsets, col_indices, values, 3);

    // Create input vector [1, 2, 3]
    let input = vec![F17::from(1), F17::from(2), F17::from(3)];

    // Expected result:
    // [2*1 + 0*2 + 1*3] = [5]
    // [0*1 + 3*2 + 0*3] = [6]
    // [4*1 + 0*2 + 5*3] = [19 ≡ 2 mod 17]
    let result = matrix.mul_vector(&input);

    assert_eq!(result[0], F17::from(5));
    assert_eq!(result[1], F17::from(6));
    assert_eq!(result[2], F17::from(2)); // 19 mod 17 = 2
  }
}

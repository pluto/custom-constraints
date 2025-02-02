use std::ops::Mul;

use super::*;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
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

  pub fn new_rows_cols(num_rows: usize, num_cols: usize) -> SparseMatrix<F> {
    Self { row_offsets: vec![0; num_rows + 1], col_indices: vec![], values: vec![], num_cols }
  }

  pub fn write(mut self, row: usize, col: usize, val: F) -> Self {
    // Check bounds
    assert!(row < self.row_offsets.len() - 1, "Row index out of bounds");
    assert!(col < self.num_cols, "Column index out of bounds");
    assert_ne!(val, F::ZERO, "Trying to add a zero element into the `SparseMatrix`!");

    // Get the range of indices for the current row
    let start = self.row_offsets[row];
    let end = self.row_offsets[row + 1];

    // Search for the column index in the current row
    let pos = self.col_indices[start..end]
      .binary_search(&col)
      .map(|i| start + i)
      .unwrap_or_else(|i| start + i);

    if pos < end && self.col_indices[pos] == col {
      // Overwrite existing value
      self.values[pos] = val;
    } else {
      // Insert new value
      self.col_indices.insert(pos, col);
      self.values.insert(pos, val);

      // Update row offsets for subsequent rows
      for i in row + 1..self.row_offsets.len() {
        self.row_offsets[i] += 1;
      }
    }

    self
  }

  fn remove(mut self, row: usize, col: usize) -> Self {
    // Get the range of indices for the current row
    let start = self.row_offsets[row];
    let end = self.row_offsets[row + 1];

    // Search for the column index in the current row
    if let Ok(pos) = self.col_indices[start..end].binary_search(&col) {
      let pos = start + pos;

      // Remove the element
      self.col_indices.remove(pos);
      self.values.remove(pos);

      // Update row offsets for subsequent rows
      for i in row + 1..self.row_offsets.len() {
        self.row_offsets[i] -= 1;
      }
    }

    self
  }
}

impl<F: Field> Mul<&Vec<F>> for &SparseMatrix<F> {
  type Output = Vec<F>;

  fn mul(self, rhs: &Vec<F>) -> Self::Output {
    // TODO: Make error
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

impl<F: Field> Mul<&SparseMatrix<F>> for &SparseMatrix<F> {
  type Output = SparseMatrix<F>;

  fn mul(self, rhs: &SparseMatrix<F>) -> Self::Output {
    // We'll implement elementwise multiplication but first check dimensions match
    assert_eq!(self.num_cols, rhs.num_cols, "Matrices must have same dimensions");

    // For the Hadamard product, we'll only have non-zero elements where both matrices
    // have non-zero elements at the same position
    let mut result_values = Vec::new();
    let mut result_col_indices = Vec::new();
    let mut result_row_offsets = vec![0];

    // Process each row
    for row in 0..self.row_offsets.len() - 1 {
      // Get the ranges for non-zero elements in this row for both matrices
      let self_start = self.row_offsets[row];
      let self_end = self.row_offsets[row + 1];
      let rhs_start = rhs.row_offsets[row];
      let rhs_end = rhs.row_offsets[row + 1];

      // Create iterators over the non-zero elements in this row
      let mut self_iter = (self_start..self_end).map(|i| (self.col_indices[i], self.values[i]));
      let mut rhs_iter = (rhs_start..rhs_end).map(|i| (rhs.col_indices[i], rhs.values[i]));

      // Keep track of our position in each iterator
      let mut self_next = self_iter.next();
      let mut rhs_next = rhs_iter.next();

      // Merge the non-zero elements
      while let (Some((self_col, self_val)), Some((rhs_col, rhs_val))) = (self_next, rhs_next) {
        match self_col.cmp(&rhs_col) {
          std::cmp::Ordering::Equal => {
            // When columns match, multiply the values
            result_values.push(self_val * rhs_val);
            result_col_indices.push(self_col);
            self_next = self_iter.next();
            rhs_next = rhs_iter.next();
          },
          std::cmp::Ordering::Less => {
            // Skip elements only in self
            self_next = self_iter.next();
          },
          std::cmp::Ordering::Greater => {
            // Skip elements only in rhs
            rhs_next = rhs_iter.next();
          },
        }
      }

      // Record where this row ends
      result_row_offsets.push(result_values.len());
    }

    SparseMatrix::new(result_row_offsets, result_col_indices, result_values, self.num_cols)
  }
}

#[cfg(test)]
mod tests {

  use super::*;

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_write() {
    // Create this matrix:
    // [2 0 1]
    // [0 3 0]
    // [4 0 5]

    let row_offsets = vec![0, 2, 3, 5];
    let col_indices = vec![0, 2, 1, 0, 2];
    let values = vec![
      F17::from(2),
      F17::from(1), // First row: 2 and 1
      F17::from(3), // Second row: just 3
      F17::from(4),
      F17::from(5), // Third row: 4 and 5
    ];

    let matrix = SparseMatrix::new(row_offsets, col_indices, values, 3);

    let write_matrix = SparseMatrix::new_rows_cols(3, 3)
      .write(0, 0, F17::from(2))
      .write(0, 2, F17::ONE)
      .write(1, 1, F17::from(3))
      .write(2, 0, F17::from(4))
      .write(2, 2, F17::from(5));

    assert_eq!(matrix, write_matrix);
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_sparse_matrix_vector_multiplication() {
    let row_offsets = vec![0, 2, 3, 5];
    let col_indices = vec![0, 2, 1, 0, 2];
    let values = vec![F17::from(2), F17::from(1), F17::from(3), F17::from(4), F17::from(5)];

    let matrix = SparseMatrix::new(row_offsets, col_indices, values, 3);

    // Create input vector [1, 2, 3]
    let input = vec![F17::from(1), F17::from(2), F17::from(3)];

    // Expected result:
    // [2*1 + 0*2 + 1*3] = [5]
    // [0*1 + 3*2 + 0*3] = [6]
    // [4*1 + 0*2 + 5*3] = [19 ≡ 2 mod 17]
    let result = &matrix * &input;

    assert_eq!(result[0], F17::from(5));
    assert_eq!(result[1], F17::from(6));
    assert_eq!(result[2], F17::from(2)); // 19 mod 17 = 2
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_hadamard_multiplication() {
    // Create two test matrices:
    // Matrix 1:
    // [2 0 1]
    // [0 3 0]
    // [4 0 5]
    let test_matrix1 = SparseMatrix::new_rows_cols(3, 3)
      .write(0, 0, F17::from(2))
      .write(0, 2, F17::from(1))
      .write(1, 1, F17::from(3))
      .write(2, 0, F17::from(4))
      .write(2, 2, F17::from(5));

    // Matrix 2:
    // [3 0 0]
    // [0 2 1]
    // [0 0 2]
    let test_matrix2 = SparseMatrix::new_rows_cols(3, 3)
      .write(0, 0, F17::from(3))
      .write(1, 1, F17::from(2))
      .write(1, 2, F17::from(1))
      .write(2, 2, F17::from(2));

    // Perform Hadamard multiplication
    let result = &test_matrix1 * &test_matrix2;

    // The result should be:
    // [6 0 0]
    // [0 6 0]
    // [0 0 10]
    assert_eq!(result.values, [
      F17::from(6),  // 2*3 at (0,0)
      F17::from(6),  // 3*2 at (1,1)
      F17::from(10), // 5*2 at (2,2)
    ]);
    assert_eq!(result.col_indices, [0, 1, 2]);
    assert_eq!(result.row_offsets, [0, 1, 2, 3]);
  }
}

//! Provides a Compressed Sparse Row (CSR) matrix implementation optimized for efficient operations.
//!
//! The [`SparseMatrix`] type is designed to handle sparse matrices efficiently by storing only
//! non-zero elements in a compressed format. It supports matrix-vector multiplication and
//! element-wise (Hadamard) matrix multiplication.

use std::ops::Mul;

use super::*;

// TODO: Probably just combine values with their col indices
/// A sparse matrix implementation using the Compressed Sparse Row (CSR) format.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SparseMatrix<F> {
  /// Offsets into col_indices/values for the start of each row
  row_offsets: Vec<usize>,
  /// Column indices of non-zero elements
  col_indices: Vec<usize>,
  /// Values of non-zero elements
  values: Vec<F>,
  /// Number of columns in the matrix
  num_cols: usize,
}

impl<F: Field> SparseMatrix<F> {
  /// Creates a new sparse matrix from its CSR components.
  pub fn new(
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<F>,
    num_cols: usize,
  ) -> Self {
    // Add validation here
    Self { row_offsets, col_indices, values, num_cols }
  }

  /// Creates an empty sparse matrix with the specified dimensions.
  pub fn new_rows_cols(num_rows: usize, num_cols: usize) -> SparseMatrix<F> {
    Self { row_offsets: vec![0; num_rows + 1], col_indices: vec![], values: vec![], num_cols }
  }

  /// Writes a value to the specified position in the matrix.
  ///
  /// # Panics
  /// - If row or column indices are out of bounds
  /// - If attempting to write a zero value
  pub fn write(&mut self, row: usize, col: usize, val: F) {
    // Check bounds
    assert!(row < self.row_offsets.len() - 1, "Row index out of bounds");
    assert!(col < self.num_cols, "Column index out of bounds");
    assert_ne!(val, F::ZERO, "Trying to add a zero element into the `SparseMatrix`!");

    // Get the range of indices for the current row
    let start = self.row_offsets[row];
    let end = self.row_offsets[row + 1];

    // Search for the column index in the current row
    let pos =
      self.col_indices[start..end].binary_search(&col).map_or_else(|i| start + i, |i| start + i);

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
  }

  /// Writes a value to the matrix, expanding its dimensions if necessary.
  ///
  /// If the specified position is outside the current matrix dimensions,
  /// the matrix will be expanded to accommodate the new position.
  ///
  /// # Arguments
  /// * `row` - Row index for the value
  /// * `col` - Column index for the value
  /// * `val` - Value to write
  ///
  /// # Panics
  /// - If attempting to write a zero value
  pub fn write_expand(&mut self, row: usize, col: usize, val: F) {
    assert_ne!(val, F::ZERO, "Trying to add a zero element into the `SparseMatrix`!");

    // Expand the matrix if necessary
    if row >= self.row_offsets.len() - 1 {
      // Add new row offsets, copying the last offset
      let last_offset = *self.row_offsets.last().unwrap();
      self.row_offsets.resize(row + 2, last_offset);
    }
    if col >= self.num_cols {
      self.num_cols = col + 1;
    }

    // Now we can use the existing write logic
    self.write(row, col, val);
  }

  /// Returns the current dimensions of the matrix.
  ///
  /// # Returns
  /// A tuple (rows, cols) representing the matrix dimensions
  pub fn dimensions(&self) -> (usize, usize) {
    (self.row_offsets.len() - 1, self.num_cols)
  }

  /// Adds a new empty row to the matrix
  pub fn add_row(&mut self) {
    self.row_offsets.push(*self.row_offsets.last().unwrap_or(&0));
  }

  #[allow(unused)]
  /// Removes an entry from the [`SparseMatrix`]
  fn remove(&mut self, row: usize, col: usize) {
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
  }
}

impl<F: Field> Mul<&Vec<F>> for &SparseMatrix<F> {
  type Output = Vec<F>;

  /// Performs matrix-vector multiplication.
  ///
  /// # Panics
  /// If the vector length doesn't match the matrix column count.
  fn mul(self, rhs: &Vec<F>) -> Self::Output {
    // TODO: Make error
    assert_eq!(rhs.len(), self.num_cols, "Invalid vector length");
    let mut result = vec![F::ZERO; self.row_offsets.len() - 1];

    #[allow(clippy::needless_range_loop)]
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

  /// Performs element-wise (Hadamard) matrix multiplication.
  ///
  /// # Panics
  /// If matrix dimensions don't match.
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

impl<F: Field + Display> Display for SparseMatrix<F> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    // First, we'll find the maximum width needed for any number
    // This helps us align columns nicely
    let max_width = self.values.iter().map(|v| format!("{v}").len()).max().unwrap_or(1).max(1); // At least 1 character for "0"

    // For each row...
    for row in 0..self.row_offsets.len() - 1 {
      write!(f, "[")?;

      // Find the non-zero elements in this row
      let row_start = self.row_offsets[row];
      let row_end = self.row_offsets[row + 1];
      let mut current_col = 0;

      // Process each column, inserting zeros where needed
      for col in 0..self.num_cols {
        // Add spacing between elements
        if col > 0 {
          write!(f, " ")?;
        }

        // Check if we have a non-zero element at this position
        if current_col < row_end - row_start && self.col_indices[row_start + current_col] == col {
          // We found a non-zero element
          let val = &self.values[row_start + current_col];
          write!(f, "{val:>max_width$}")?;
          current_col += 1;
        } else {
          // This position is zero
          write!(f, "{:>width$}", 0, width = max_width)?;
        }
      }

      writeln!(f, "]")?;
    }
    Ok(())
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

    let mut write_matrix = SparseMatrix::new_rows_cols(3, 3);
    write_matrix.write(0, 0, F17::from(2));
    write_matrix.write(0, 2, F17::ONE);
    write_matrix.write(1, 1, F17::from(3));
    write_matrix.write(2, 0, F17::from(4));
    write_matrix.write(2, 2, F17::from(5));

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
    let mut test_matrix1 = SparseMatrix::new_rows_cols(3, 3);
    test_matrix1.write(0, 0, F17::from(2));
    test_matrix1.write(0, 2, F17::from(1));
    test_matrix1.write(1, 1, F17::from(3));
    test_matrix1.write(2, 0, F17::from(4));
    test_matrix1.write(2, 2, F17::from(5));

    // Matrix 2:
    // [3 0 0]
    // [0 2 1]
    // [0 0 2]
    let mut test_matrix2 = SparseMatrix::new_rows_cols(3, 3);
    test_matrix2.write(0, 0, F17::from(3));
    test_matrix2.write(1, 1, F17::from(2));
    test_matrix2.write(1, 2, F17::from(1));
    test_matrix2.write(2, 2, F17::from(2));

    // Perform Hadamard multiplication
    let result = &test_matrix1 * &test_matrix2;

    // The result should be:
    // [6 0 0]
    // [0 6 0]
    // [0 0 10]
    assert_eq!(
      result.values,
      [
        F17::from(6),  // 2*3 at (0,0)
        F17::from(6),  // 3*2 at (1,1)
        F17::from(10), // 5*2 at (2,2)
      ]
    );
    assert_eq!(result.col_indices, [0, 1, 2]);
    assert_eq!(result.row_offsets, [0, 1, 2, 3]);
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_write_expand() {
    // Create a 2x2 matrix
    let mut matrix = SparseMatrix::new_rows_cols(2, 2);

    // Write within bounds
    matrix.write_expand(0, 0, F17::from(1));
    matrix.write_expand(1, 1, F17::from(2));

    // Write beyond current dimensions
    matrix.write_expand(3, 4, F17::from(3));

    // Check dimensions
    let (rows, cols) = matrix.dimensions();
    assert_eq!(rows, 4);
    assert_eq!(cols, 5);

    // Verify values
    let expected_values = vec![(0, 0, F17::from(1)), (1, 1, F17::from(2)), (3, 4, F17::from(3))];

    for (row, col, expected_val) in expected_values {
      // Find the value in the sparse representation
      let row_start = matrix.row_offsets[row];
      let row_end = matrix.row_offsets[row + 1];
      let pos = matrix.col_indices[row_start..row_end].iter().position(|&c| c == col);

      match pos {
        Some(idx) => {
          assert_eq!(
            matrix.values[row_start + idx],
            expected_val,
            "Value mismatch at position ({}, {})",
            row,
            col
          );
        },
        None => panic!("Expected value not found at position ({}, {})", row, col),
      }
    }
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_write_expand_multiple() {
    let mut matrix = SparseMatrix::new_rows_cols(2, 2);

    // Write values in various orders to test expansion
    matrix.write_expand(5, 3, F17::from(1));
    matrix.write_expand(2, 6, F17::from(2));
    matrix.write_expand(4, 1, F17::from(3));

    let (rows, cols) = matrix.dimensions();
    assert_eq!(rows, 6);
    assert_eq!(cols, 7);

    // Test that row offsets are properly maintained
    assert_eq!(matrix.row_offsets.len(), rows + 1);

    // Verify that all rows between have valid offsets
    for i in 0..rows {
      assert!(
        matrix.row_offsets[i] <= matrix.row_offsets[i + 1],
        "Row offset invariant violated at row {}",
        i
      );
    }
  }

  #[test]
  #[should_panic(expected = "Trying to add a zero element")]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_write_expand_zero() {
    let mut matrix = SparseMatrix::new_rows_cols(2, 2);
    matrix.write_expand(3, 3, F17::from(0));
  }
}

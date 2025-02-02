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

pub struct TestSparseMatrix<
  F,
  const NUM_NONZERO: usize,
  const NUM_ROWS: usize,
  const NUM_COLUMNS: usize,
> where [(); NUM_ROWS + 1]: {
  // row_offsets is now based on NUM_ROWS instead of NUM_NONZERO
  row_offsets: [usize; NUM_ROWS + 1],
  col_indices: [usize; NUM_NONZERO],
  values:      [F; NUM_NONZERO],
}

impl<F: Field, const NUM_NONZERO: usize, const NUM_ROWS: usize, const NUM_COLUMNS: usize>
  TestSparseMatrix<F, NUM_NONZERO, NUM_ROWS, NUM_COLUMNS>
where [(); NUM_ROWS + 1]:
{
  pub const fn new() -> Self {
    Self {
      // Initialize with zeros - one entry per row plus the extra end marker
      row_offsets: [0; NUM_ROWS + 1],
      col_indices: [0; NUM_NONZERO],
      values:      [F::ZERO; NUM_NONZERO],
    }
  }

  // TODO: having this as `const` is possible, but breaks the compiler
  pub fn push_val<const ROW: usize, const COL: usize>(
    self,
    val: F,
  ) -> TestSparseMatrix<
    F,
    { NUM_NONZERO + 1 },
    { (ROW >= NUM_ROWS) as usize * (ROW + 1) + (ROW < NUM_ROWS) as usize * NUM_ROWS },
    NUM_COLUMNS,
  >
  where
    [(); ((ROW >= NUM_ROWS) as usize * (ROW + 1) + (ROW < NUM_ROWS) as usize * NUM_ROWS) + 1]:,
    [(); (NUM_COLUMNS > COL) as usize - 1]:,
  {
    // Initialize new arrays. For row_offsets, we use our conditional size calculation
    // that expands the matrix only if needed
    let mut new_values = [F::ZERO; NUM_NONZERO + 1];
    let mut new_col_indices = [0; NUM_NONZERO + 1];
    let mut new_row_offsets =
      [0; (ROW >= NUM_ROWS) as usize * (ROW + 1) + (ROW < NUM_ROWS) as usize * NUM_ROWS + 1];

    // Copy existing values and column indices
    new_values[..NUM_NONZERO].copy_from_slice(&self.values);
    new_col_indices[..NUM_NONZERO].copy_from_slice(&self.col_indices);

    // Copy existing row offsets
    new_row_offsets[..NUM_ROWS + 1].copy_from_slice(&self.row_offsets);

    // Initialize new row offsets if we're expanding
    if ROW >= NUM_ROWS {
      let last_offset = self.row_offsets[NUM_ROWS];
      for i in NUM_ROWS + 1..=ROW + 1 {
        new_row_offsets[i] = last_offset;
      }
    }

    // Add new value
    new_values[NUM_NONZERO] = val;
    new_col_indices[NUM_NONZERO] = COL;

    // Update row offsets for affected rows
    for i in
      (ROW + 1)..=((ROW >= NUM_ROWS) as usize * (ROW + 1) + (ROW < NUM_ROWS) as usize * NUM_ROWS)
    {
      new_row_offsets[i] += 1;
    }

    TestSparseMatrix {
      values:      new_values,
      col_indices: new_col_indices,
      row_offsets: new_row_offsets,
    }
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

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_const_sparse_matrix_construction() {
    // Creating a 3x3 matrix:
    // [2 0 1]
    // [0 3 0]
    // [4 0 5]

    // Start with an empty 3x3 matrix (3 rows, 3 columns, 0 non-zero elements)
    let matrix: TestSparseMatrix<F17, 0, 3, 3> = TestSparseMatrix::new();

    // Add elements one by one
    let matrix = matrix.push_val::<0, 0>(F17::from(2)); // Add 2 at (0,0)
    let matrix = matrix.push_val::<0, 2>(F17::from(1)); // Add 1 at (0,2)
    let matrix = matrix.push_val::<1, 1>(F17::from(3)); // Add 3 at (1,1)
    let matrix = matrix.push_val::<2, 0>(F17::from(4)); // Add 4 at (2,0)
    let matrix = matrix.push_val::<2, 2>(F17::from(5)); // Add 5 at (2,2)

    // Verify final state
    assert_eq!(matrix.values, [
      F17::from(2),
      F17::from(1),
      F17::from(3),
      F17::from(4),
      F17::from(5)
    ]);

    assert_eq!(matrix.col_indices, [0, 2, 1, 0, 2]);

    // Now row_offsets has a fixed size of NUM_ROWS + 1 = 4
    assert_eq!(matrix.row_offsets, [0, 2, 3, 5]);
  }

  #[test]
  fn test_conditional_row_expansion() {
    // Start with a 3x3 matrix
    let matrix: TestSparseMatrix<F17, 0, 3, 3> = TestSparseMatrix::new();

    // Add to existing rows - matrix stays 3x3
    let matrix = matrix.push_val::<2, 1>(F17::from(3));

    // Add to row 5 - matrix expands to 6x3
    let matrix = matrix.push_val::<5, 2>(F17::from(5));

    assert_eq!(matrix.col_indices, [1, 2]);
    // Row offsets will have 7 elements (6 rows + 1)
    assert_eq!(matrix.row_offsets, [0, 0, 0, 1, 1, 1, 2]);
  }
}

use super::*;

impl<F: Field> CCS<Generic<F>, F> {
  /// Checks if a witness and public input satisfy the constraint system.
  ///
  /// Forms vector z = (w, 1, x) and verifies that all constraints are satisfied.
  ///
  /// # Arguments
  /// * `w` - The witness vector
  /// * `x` - The public input vector
  ///
  /// # Returns
  /// `true` if all constraints are satisfied, `false` otherwise
  pub fn is_satisfied(&self, w: &[F], x: &[F]) -> bool {
    // Construct z = (w, 1, x)
    let mut z = Vec::with_capacity(w.len() + 1 + x.len());
    z.extend(w.iter().copied());
    z.push(F::ONE);
    z.extend(x.iter().copied());

    // Compute all matrix-vector products
    let products: Vec<Vec<F>> = self
      .matrices
      .iter()
      .enumerate()
      .map(|(i, matrix)| {
        let result = matrix * &z;
        println!("M{i} · z = {result:?}");
        result
      })
      .collect();

    // For each row in the output...
    let m = if let Some(first) = products.first() {
      first.len()
    } else {
      return true; // No constraints
    };

    // For each output coordinate...
    for row in 0..m {
      let mut sum = F::ZERO;

      // For each constraint...
      for (i, multiset) in self.multisets.iter().enumerate() {
        let mut term = products[multiset[0]][row];

        for &idx in multiset.iter().skip(1) {
          term *= products[idx][row];
        }

        let contribution = self.selectors[i] * term;
        sum += contribution;
      }

      if sum != F::ZERO {
        return false;
      }
    }

    true
  }

  /// Creates a new CCS configured for constraints up to the given degree.
  ///
  /// # Arguments
  /// * `d` - Maximum degree of constraints
  ///
  /// # Panics
  /// If d < 2
  pub fn new_degree(d: usize) -> Self {
    assert!(d >= 2, "Degree must be positive");

    let mut ccs = Self { selectors: Vec::new(), multisets: Vec::new(), matrices: Vec::new() };

    // We'll create terms starting from highest degree down to degree 1
    // For a degree d CCS, we need terms of all degrees from d down to 1
    let mut next_matrix_index = 0;

    // Handle each degree from d down to 1
    for degree in (1..=d).rev() {
      // For a term of degree k, we need k matrices Hadamard multiplied
      let matrix_indices: Vec<usize> = (0..degree).map(|i| next_matrix_index + i).collect();

      // Add this term's multiset and its coefficient
      ccs.multisets.push(matrix_indices);
      ccs.selectors.push(F::ONE);

      // Update our tracking of matrix indices
      next_matrix_index += degree;
    }

    // Calculate total number of matrices needed:
    // For degree d, we need d + (d-1) + ... + 1 matrices
    // This is the triangular number formula: n(n+1)/2
    let total_matrices = (d * (d + 1)) / 2;

    // Initialize empty matrices - their content will be filled during conversion
    for _ in 0..total_matrices {
      ccs.matrices.push(SparseMatrix::new_rows_cols(1, 0));
    }

    ccs
  }
}

impl<F: Field + Display> Display for CCS<Generic<F>, F> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    writeln!(f, "Customizable Constraint System:")?;

    // First, display all matrices with their indices
    writeln!(f, "\nMatrices:")?;
    for (i, matrix) in self.matrices.iter().enumerate() {
      writeln!(f, "M{i} =")?;
      writeln!(f, "{matrix}")?;
    }

    // Show how constraints are formed from multisets and constants
    writeln!(f, "\nConstraints:")?;

    // We expect multisets to come in pairs, each pair forming one constraint
    for i in 0..self.multisets.len() {
      // Write the constant for the first multiset
      write!(f, "{}·(", self.selectors[i])?;

      // Write the Hadamard product for the first multiset
      if let Some(first_idx) = self.multisets[i].first() {
        write!(f, "M{first_idx}")?;
        for &idx in &self.multisets[i][1..] {
          write!(f, "∘M{idx}")?;
        }
      }
      write!(f, ")")?;

      // Sum up the expressions to the last one
      if i < self.multisets.len() - 1 {
        write!(f, " + ")?;
      }
    }
    writeln!(f, " = 0")?;
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::mock::F17;

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_ccs_satisfaction() {
    println!("\nSetting up CCS for constraint x * y = z");

    // For z = (y, z, 1, x), create matrices:
    let mut m1 = SparseMatrix::new_rows_cols(1, 4);
    m1.write(0, 3, F17::ONE); // Select x
    let mut m2 = SparseMatrix::new_rows_cols(1, 4);
    m2.write(0, 0, F17::ONE); // Select y
    let mut m3 = SparseMatrix::new_rows_cols(1, 4);
    m3.write(0, 1, F17::ONE); // Select z

    println!("Created matrices:");
    println!("M1 (selects x): {m1:?}");
    println!("M2 (selects y): {m2:?}");
    println!("M3 (selects z): {m3:?}");

    let mut ccs = CCS::<Generic<_>, _>::new();
    ccs.matrices = vec![m1, m2, m3];
    // Encode x * y - z = 0
    ccs.multisets = vec![vec![0, 1], vec![2]];
    ccs.selectors = vec![F17::ONE, F17::from(-1)];

    println!("\nTesting valid case: x=2, y=3, z=6");
    let x = vec![F17::from(2)]; // public input x = 2
    let w = vec![F17::from(3), F17::from(6)]; // witness y = 3, z = 6
    assert!(ccs.is_satisfied(&w, &x));

    println!("\nTesting invalid case: x=2, y=3, z=7");
    let w_invalid = vec![F17::from(3), F17::from(7)]; // witness y = 3, z = 7 (invalid)
    assert!(!ccs.is_satisfied(&w_invalid, &x));
  }
}

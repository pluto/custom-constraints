use super::*;

#[derive(Clone, Debug, Default)]
pub struct Plonkish<F>(PhantomData<F>);
impl<F> CCSType<F> for Plonkish<F> {
  type Selectors = Vec<F>;
}

impl<F: Field> CCS<Plonkish<F>, F> {
  /// Creates a new Plonkish CCS with the specified width.
  /// The width determines the number of matrices A_i in the system.
  /// The minimum width is 2, corresponding to PLONK-style constraints
  /// where we have cross-terms between different selector matrices.
  ///
  /// # Arguments
  /// * `width` - Number of matrices A_i (must be >= 2)
  ///
  /// # Panics
  /// If width < 2
  pub fn new_width(width: usize) -> Self {
    assert!(width >= 2, "Width must be at least 2");

    let mut ccs = Self::default();

    // Initialize matrices A_1 through A_width
    for _ in 0..width {
      ccs.matrices.push(SparseMatrix::new_rows_cols(1, 0));
    }

    // Set up multisets for cross terms only (i != j)
    for i in 0..width {
      for j in (i + 1)..width {
        ccs.multisets.push(vec![i, j]);
      }
    }

    // Set up multisets for linear terms
    for i in 0..width {
      ccs.multisets.push(vec![i]);
    }

    // Add multiset for constant term (empty multiset)
    ccs.multisets.push(vec![]);

    // Initialize selectors
    let num_cross_terms = (width * (width - 1)) / 2;
    let num_terms = num_cross_terms + width + 1; // +1 for constant term
    ccs.selectors = vec![vec![F::ZERO]; num_terms];

    ccs
  }

  /// Helper method to set a cross-term coefficient
  pub fn set_cross_term(&mut self, i: usize, j: usize, value: F) {
    assert!(i != j, "Cross terms must be between different matrices");
    let width = self.matrices.len();
    assert!(i < width && j < width, "Index out of bounds");

    // Ensure i < j for consistent indexing
    let (i, j) = if i < j { (i, j) } else { (j, i) };

    // Calculate index for the cross term
    let idx = (i * (2 * width - i - 1)) / 2 + (j - i - 1);

    if let Some(selector) = self.selectors.get_mut(idx) {
      selector[0] = value;
    }
  }

  /// Helper method to set a linear term coefficient
  pub fn set_linear(&mut self, i: usize, value: F) {
    let width = self.matrices.len();
    assert!(i < width, "Index out of bounds");

    let num_cross_terms = (width * (width - 1)) / 2;
    if let Some(selector) = self.selectors.get_mut(num_cross_terms + i) {
      selector[0] = value;
    }
  }

  pub fn set_constant(&mut self, value: F) {
    let idx = self.selectors.len() - 1; // Constant term is last
    if let Some(selector) = self.selectors.get_mut(idx) {
      selector[0] = value;
    }
  }

  /// Checks if a witness and public input satisfy the Plonkish constraint system.
  /// The constraint has the form:
  /// sum_{i<j} q_{i,j} (A_i z o A_j z) + sum_i q_i (A_i z) + q_c = 0
  ///
  /// # Arguments
  /// * `x` - The public input vector
  /// * `w` - The witness vector
  ///
  /// # Returns
  /// `true` if all constraints are satisfied, `false` otherwise
  // New method to set constant term
  pub fn is_satisfied(&self, x: &[F], w: &[F]) -> bool {
    let mut z = Vec::with_capacity(x.len() + w.len());
    z.extend(x.iter().copied());
    z.extend(w.iter().copied());

    let products: Vec<Vec<F>> = self
      .matrices
      .iter()
      .enumerate()
      .map(|(i, matrix)| {
        let result = matrix * &z;
        println!("A_{}·z = {:?}", i, result);
        result
      })
      .collect();

    let m = if let Some(first) = products.first() {
      first.len()
    } else {
      return true;
    };

    for row in 0..m {
      let mut sum = F::ZERO;
      let width = self.matrices.len();
      let mut term_idx = 0;

      // Process quadratic terms (i < j)
      for i in 0..width {
        for j in (i + 1)..width {
          if let Some(selector) = self.selectors.get(term_idx) {
            let term = products[i][row] * products[j][row];
            for &coeff in selector {
              sum += coeff * term;
            }
          }
          term_idx += 1;
        }
      }

      // Process linear terms
      for i in 0..width {
        if let Some(selector) = self.selectors.get(self.num_cross_terms() + i) {
          let term = products[i][row];
          for &coeff in selector {
            sum += coeff * term;
          }
        }
      }

      // Add constant term
      if let Some(selector) = self.selectors.last() {
        for &coeff in selector {
          sum += coeff;
        }
      }

      println!("Row {}: sum = {:?}", row, sum);
      if sum != F::ZERO {
        return false;
      }
    }

    true
  }

  /// Helper to calculate number of cross terms
  fn num_cross_terms(&self) -> usize {
    let width = self.matrices.len();
    (width * (width - 1)) / 2
  }
}

impl<F: Field + Display> Display for CCS<Plonkish<F>, F> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    let width = self.matrices.len();

    writeln!(f, "Plonkish Constraint System (width = {}):", width)?;

    // Display matrices
    writeln!(f, "\nMatrices:")?;
    for (i, matrix) in self.matrices.iter().enumerate() {
      writeln!(f, "A_{} =", i)?;
      writeln!(f, "{}", matrix)?;
    }

    // Display selectors
    writeln!(f, "\nSelectors:")?;
    let mut term_idx = 0;

    // Display cross term selectors
    for i in 0..width {
      for j in (i + 1)..width {
        if let Some(selector) = self.selectors.get(term_idx) {
          write!(f, "q_{},{} = [", i, j)?;
          for (idx, &coeff) in selector.iter().enumerate() {
            if idx > 0 {
              write!(f, ", ")?;
            }
            write!(f, "{}", coeff)?;
          }
          writeln!(f, "]")?;
        }
        term_idx += 1;
      }
    }

    // Display linear term selectors
    for i in 0..width {
      if let Some(selector) = self.selectors.get(term_idx) {
        write!(f, "q_{} = [", i)?;
        for (idx, &coeff) in selector.iter().enumerate() {
          if idx > 0 {
            write!(f, ", ")?;
          }
          write!(f, "{}", coeff)?;
        }
        writeln!(f, "]")?;
      }
      term_idx += 1;
    }

    if let Some(selector) = self.selectors.last() {
      write!(f, "q_c = [")?;
      for (idx, &coeff) in selector.iter().enumerate() {
        if idx > 0 {
          write!(f, ", ")?;
        }
        write!(f, "{}", coeff)?;
      }
      writeln!(f, "]")?;
    }

    // Display constraint equation
    writeln!(f, "\nConstraint equation:")?;

    let mut first_term = true;
    term_idx = 0;

    // Display cross terms (i != j)
    for i in 0..width {
      for j in (i + 1)..width {
        if let Some(selector) = self.selectors.get(term_idx) {
          if !selector.iter().all(|&x| x == F::ZERO) {
            if !first_term {
              write!(f, " + ")?;
            }
            write!(f, "q_{},{}·(A_{}·z ∘ A_{}·z)", i, j, i, j)?;
            first_term = false;
          }
        }
        term_idx += 1;
      }
    }

    // Display linear terms
    for i in 0..width {
      if let Some(selector) = self.selectors.get(term_idx) {
        if !selector.iter().all(|&x| x == F::ZERO) {
          if !first_term {
            write!(f, " + ")?;
          }
          write!(f, "q_{}·(A_{}·z)", i, i)?;
          first_term = false;
        }
      }
      term_idx += 1;
    }

    if self.selectors.last().is_some() {
      write!(f, " + q_c")?;
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
  fn test_plonkish_structure() {
    let ccs = CCS::<Plonkish<F17>, F17>::new_width(3);

    // For width 3, we should have:
    // - 3 cross terms (1,2), (1,3), (2,3)
    // - 3 linear terms
    assert_eq!(ccs.multisets.len(), 6, "Should have 6 terms total");

    // Check cross term multisets
    assert_eq!(ccs.multisets[0], vec![0, 1], "First cross term incorrect");
    assert_eq!(ccs.multisets[1], vec![0, 2], "Second cross term incorrect");
    assert_eq!(ccs.multisets[2], vec![1, 2], "Third cross term incorrect");

    // Check linear term multisets
    assert_eq!(ccs.multisets[3], vec![0], "First linear term incorrect");
    assert_eq!(ccs.multisets[4], vec![1], "Second linear term incorrect");
    assert_eq!(ccs.multisets[5], vec![2], "Third linear term incorrect");
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_plonkish_display() {
    let mut ccs = CCS::<Plonkish<F17>, F17>::new_width(2);

    // Set up test matrices
    let mut a1 = SparseMatrix::new_rows_cols(1, 4);
    a1.write(0, 0, F17::ONE);
    ccs.matrices[0] = a1;

    let mut a2 = SparseMatrix::new_rows_cols(1, 4);
    a2.write(0, 1, F17::ONE);
    ccs.matrices[1] = a2;

    // Set some coefficients
    ccs.set_cross_term(0, 1, F17::from(3)); // 3(A_1·z)(A_2·z)
    ccs.set_linear(0, F17::from(4)); // 4(A_1·z)
    ccs.set_linear(1, F17::from(5)); // 5(A_2·z)

    println!("{}", ccs);
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_plonkish_satisfaction() {
    let mut ccs = CCS::<Plonkish<F17>, F17>::new_width(2);

    // Set up matrices for x * y + 2x + 3y + 4 = 0
    let mut a1 = SparseMatrix::new_rows_cols(1, 2);
    a1.write(0, 0, F17::ONE); // Select x
    ccs.matrices[0] = a1;

    let mut a2 = SparseMatrix::new_rows_cols(1, 2);
    a2.write(0, 1, F17::ONE); // Select y
    ccs.matrices[1] = a2;

    // Set coefficients
    ccs.set_cross_term(0, 1, F17::ONE); // 1 * (x * y)
    ccs.set_linear(0, F17::from(2)); // + 2x
    ccs.set_linear(1, F17::from(3)); // + 3y
    ccs.set_constant(F17::from(8)); // + 4

    println!("ccs: {ccs}");

    // With:
    // x = 4, y = 5
    // 4 * 5 + 2*4 + 3*5 + 8 = 51 ≡ 0 (mod 17)
    let x = vec![];
    let w = vec![F17::from(4), F17::from(5)];

    // Let's print the computation
    println!("\nVerifying computation:");
    let prod = F17::from(4) * F17::from(5); // x * y
    let lin1 = F17::from(2) * F17::from(4); // 2x
    let lin2 = F17::from(3) * F17::from(5); // 3y
    let constant = F17::from(8); // 4
    println!("x * y = {}", prod);
    println!("2x = {}", lin1);
    println!("3y = {}", lin2);
    println!("constant = {}", constant);
    println!("sum = {}", prod + lin1 + lin2 + constant);

    assert!(ccs.is_satisfied(&x, &w));

    // Test with invalid assignment
    let w = vec![F17::from(2), F17::from(3)];
    assert!(!ccs.is_satisfied(&x, &w));
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_plonkish_simple() {
    let mut ccs = CCS::<Plonkish<F17>, F17>::new_width(2);

    // Set up matrices for x * y + 1 = 0
    let mut a1 = SparseMatrix::new_rows_cols(1, 2);
    a1.write(0, 0, F17::ONE); // Select x
    ccs.matrices[0] = a1;

    let mut a2 = SparseMatrix::new_rows_cols(1, 2);
    a2.write(0, 1, F17::ONE); // Select y
    ccs.matrices[1] = a2;

    // Set coefficients
    ccs.set_cross_term(0, 1, F17::ONE); // x * y
    ccs.set_constant(F17::ONE); // + 1

    println!("ccs: {ccs}");

    // 16 * 16 + 1 = 257 ≡ 0 (mod 17)
    let x = vec![];
    let w = vec![-F17::from(1), F17::from(1)];
    assert!(ccs.is_satisfied(&x, &w));
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_plonkish_width3() {
    let mut ccs = CCS::<Plonkish<F17>, F17>::new_width(3);

    // Let's create a constraint:
    // (x * y) + (y * z) + (x * z) + 2x + 3y + 4z + 5 = 0

    // Set up matrices
    let mut a0 = SparseMatrix::new_rows_cols(1, 3);
    a0.write(0, 0, F17::ONE); // Select x
    ccs.matrices[0] = a0;

    let mut a1 = SparseMatrix::new_rows_cols(1, 3);
    a1.write(0, 1, F17::ONE); // Select y
    ccs.matrices[1] = a1;

    let mut a2 = SparseMatrix::new_rows_cols(1, 3);
    a2.write(0, 2, F17::ONE); // Select z
    ccs.matrices[2] = a2;

    // Set cross terms
    ccs.set_cross_term(0, 1, F17::ONE); // x * y
    ccs.set_cross_term(1, 2, F17::ONE); // y * z
    ccs.set_cross_term(0, 2, F17::ONE); // x * z

    // Set linear terms
    ccs.set_linear(0, F17::from(2)); // 2x
    ccs.set_linear(1, F17::from(3)); // 3y
    ccs.set_linear(2, F17::from(4)); // 4z

    // Set constant term
    ccs.set_constant(-F17::from(4)); // - 4

    println!("ccs: {ccs}");

    // Let's use x = 2, y = 3, z = 4
    // (2 * 3) + (3 * 4) + (2 * 4) + 2*2 + 3*3 + 4*4 + 5
    // = 6 + 12 + 8 + 4 + 9 + 16 + 5
    // = 60 ≡ 9 (mod 17)
    let x = vec![];
    let w = vec![F17::from(2), F17::from(3), F17::from(4)];

    // Let's print the computation
    println!("\nVerifying computation:");
    let xy = F17::from(2) * F17::from(3);
    let yz = F17::from(3) * F17::from(4);
    let xz = F17::from(2) * F17::from(4);
    let x_term = F17::from(2) * F17::from(2);
    let y_term = F17::from(3) * F17::from(3);
    let z_term = F17::from(4) * F17::from(4);
    let constant = -F17::from(4);

    println!("x * y = {}", xy);
    println!("y * z = {}", yz);
    println!("x * z = {}", xz);
    println!("2x = {}", x_term);
    println!("3y = {}", y_term);
    println!("4z = {}", z_term);
    println!("constant = {}", constant);
    println!("sum = {}", xy + yz + xz + x_term + y_term + z_term + constant);

    // Find solution where this equals 0 (mod 17)
    // Solution: x = 2, y = 3, z = 1
    let w = vec![F17::from(2), F17::from(3), F17::from(4)];
    assert!(ccs.is_satisfied(&x, &w));

    // Invalid assignment should fail
    let w = vec![F17::from(1), F17::from(1), F17::from(1)];
    assert!(!ccs.is_satisfied(&x, &w));
  }
}

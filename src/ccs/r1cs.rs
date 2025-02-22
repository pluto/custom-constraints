use super::*;

use super::*;

/// A type marker for PLONK-style constraint systems.
///
/// This type configures a CCS to use vector-valued selectors suitable for
/// PLONK-style constraints where each selector holds coefficients for multiple
/// constraints.
#[derive(Clone, Debug, Default)]
pub struct R1CS<F>(PhantomData<F>);
impl<F: Default> CCSType<F> for R1CS<F> {
  type Selectors = F;
}

impl<F: Field> CCS<R1CS<F>, F> {
  /// Creates a new basic PLONK-style constraint system.
  ///
  /// The system will have:
  /// - Three selector matrices A, B, C for linear terms
  /// - One multiplication term between A and B matrices
  /// - Linear terms for each selector matrix
  /// - A constant term
  pub fn new_r1cs() -> Self {
    let mut ccs = Self { matrices: Vec::new(), multisets: Vec::new(), selectors: Vec::new() };

    // Initialize 3 empty matrices (A, B, C)
    for _ in 0..3 {
      ccs.matrices.push(SparseMatrix::new_rows_cols(0, 0));
    }

    // Create multisets for all terms
    ccs.multisets.push(vec![0, 1]); // Multiplication term (A,B)
    ccs.multisets.push(vec![2]); // Linear term C

    // Initialize selector vectors
    ccs.selectors = vec![F::ONE; 3];

    ccs
  }

  /// Adds a new variable to the system.
  /// Returns the index of the new variable.
  pub fn add_variable(&mut self) -> usize {
    let var_idx = self.matrices[0].dimensions().1;

    // Add a new column to each matrix
    for matrix in &mut self.matrices {
      matrix.add_column();
    }

    var_idx
  }

  /// Adds a new constraint to the system.
  ///
  /// This method:
  /// 1. Adds a new row to each selector matrix
  /// 2. Extends each selector vector with a zero coefficient
  ///
  /// # Returns
  /// The index of the new constraint
  /// Adds a new constraint to the system.
  pub fn add_constraint(&mut self) -> usize {
    let constraint_idx = self.matrices[0].dimensions().0;

    // Add a new row to each selector matrix
    for matrix in &mut self.matrices {
      matrix.add_row();
    }

    constraint_idx
  }

  /// Sets a multiplication term A[i]·z * B[j]·z in a constraint
  ///
  /// # Arguments
  /// * `constraint_idx` - Which constraint to modify
  /// * `value` - Coefficient value
  /// * `var_a` - Variable index for matrix A
  /// * `var_b` - Variable index for matrix B
  pub fn set_multiplication(
    &mut self,
    constraint_idx: usize,
    value: F,
    var_a: usize,
    var_b: usize,
  ) {
    // Update matrix A
    self.matrices[0].write(constraint_idx, var_a, F::ONE);

    // Update matrix B
    self.matrices[1].write(constraint_idx, var_b, F::ONE);
  }

  /// Sets a linear term for a specific matrix (A, B, or C) in a constraint
  ///
  /// # Arguments
  /// * `matrix` - Which matrix (0=A, 1=B, 2=C)
  /// * `constraint_idx` - Which constraint to modify
  /// * `value` - Coefficient value
  /// * `var` - Variable index to select
  pub fn set_linear(&mut self, matrix: usize, constraint_idx: usize, value: F, var: usize) {
    assert!(matrix < 3, "Matrix index must be 0 (A), 1 (B), or 2 (C)");

    // Update matrix entry
    self.matrices[matrix].write(constraint_idx, var, F::ONE);
  }

  /// Sets the constant term for a specific constraint.
  pub fn set_constant(&mut self, constraint_idx: usize, value: F) {}

  /// Checks if a witness and public input satisfy all constraints.
  pub fn is_satisfied(&self, x: &[F], w: &[F]) -> bool {
    let mut z = Vec::with_capacity(x.len() + w.len());
    z.extend(x.iter().copied());
    z.extend(w.iter().copied());

    // Calculate matrix-vector products
    let az = &self.matrices[0] * &z;
    let bz = &self.matrices[1] * &z;
    let cz = &self.matrices[2] * &z;

    let num_constraints = az.len();
    if num_constraints == 0 {
      return true;
    }

    // Check each constraint
    for row in 0..num_constraints {
      let mut sum = F::ZERO;

      // Multiplication term qm×(Az×Bz)
      let mul_term = self.selectors[0] * az[row] * bz[row];
      sum += mul_term;

      // Linear terms
      let lin_c = self.selectors[3] * cz[row]; // qo×Cz
      sum += lin_c;

      if sum != F::ZERO {
        return false;
      }
    }

    true
  }
}

impl<F: Field + Display> Display for CCS<R1CS<F>, F> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    writeln!(f, "Plonkish Constraint System:\n")?;

    // Display matrices
    writeln!(f, "Matrices:")?;
    writeln!(f, "A =")?;
    writeln!(f, "{}", self.matrices[0])?;
    writeln!(f, "B =")?;
    writeln!(f, "{}", self.matrices[1])?;
    writeln!(f, "C =")?;
    writeln!(f, "{}", self.matrices[2])?;

    // Display selectors
    writeln!(f, "\nSelectors:")?;
    writeln!(f, "qm = {:?}", self.selectors[0])?; // multiplication term
    writeln!(f, "ql = {:?}", self.selectors[1])?; // linear term for A
    writeln!(f, "qr = {:?}", self.selectors[2])?; // linear term for B
    writeln!(f, "qo = {:?}", self.selectors[3])?; // linear term for C
    writeln!(f, "qc = {:?}", self.selectors[4])?; // constant term

    // Display constraint equation
    writeln!(f, "\nConstraint equation:")?;
    let mut terms = Vec::new();

    // Add non-zero terms to equation

    terms.push("Az·Bz");

    terms.push("Cz");

    // Write equation
    if terms.is_empty() {
      write!(f, "0")?;
    } else {
      write!(f, "{}", terms.join(" + "))?;
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
  fn test_r1cs_structure() {
    let ccs = CCS::<R1CS<F17>, F17>::new_r1cs();

    // For width 3, we should have:
    // - 1 cross terms (0,1)
    // - 3 linear terms
    // - 1 constant term
    assert_eq!(ccs.multisets.len(), 2, "Should have 5 terms total");

    // Check cross term multisets
    assert_eq!(ccs.multisets[0], vec![0, 1], "First cross term incorrect");

    // Check linear term multisets
    assert_eq!(ccs.multisets[1], vec![2], "First linear term incorrect");
  }

  //   #[test]
  //   #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  //   fn test_plonkish_display() {
  //     let mut ccs = CCS::<Plonkish<F17>, F17>::new_plonkish();

  //     // Add variables
  //     let x = ccs.add_variable();
  //     let y = ccs.add_variable();
  //     let z = ccs.add_variable();
  //     let w = ccs.add_variable();

  //     // Set up display for one constraint
  //     let c1 = ccs.add_constraint();

  //     // Set coefficients for: 3(x·y) + 4x + 5y + 6z + 7 = 0
  //     ccs.set_multiplication(c1, F17::from(3), x, y); // 3(x·y)
  //     ccs.set_linear(0, c1, F17::from(4), x); // + 4x
  //     ccs.set_linear(1, c1, F17::from(5), y); // + 5y
  //     ccs.set_linear(2, c1, F17::from(6), z); // + 6z
  //     ccs.set_constant(c1, F17::from(7)); // + 7

  //     println!("{ccs}");
  //   }

  //   #[test]
  //   #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  //   fn test_plonkish_satisfaction() {
  //     let mut ccs = CCS::new_plonkish();

  //     // Add variables for x and y
  //     let x = ccs.add_variable();
  //     let y = ccs.add_variable();

  //     // Add constraint
  //     let c1 = ccs.add_constraint();

  //     // Set up constraint: x * y + 2x + 3y + 8 = 0
  //     ccs.set_multiplication(c1, F17::ONE, x, y); // x * y term
  //     ccs.set_linear(0, c1, F17::from(2), x); // 2x term
  //     ccs.set_linear(1, c1, F17::from(3), y); // 3y term
  //     ccs.set_linear(2, c1, F17::from(0), x); // no C term (using x as dummy var)
  //     ccs.set_constant(c1, F17::from(8)); // constant term

  //     println!("{ccs}");

  //     // With:
  //     // x = 4, y = 5
  //     // 4 * 5 + 2*4 + 3*5 + 8 = 51 ≡ 0 (mod 17)
  //     let x = vec![];
  //     let w = vec![F17::from(4), F17::from(5)];

  //     assert!(ccs.is_satisfied(&x, &w));

  //     // Test with invalid assignment
  //     let w = vec![F17::from(2), F17::from(3)];
  //     assert!(!ccs.is_satisfied(&x, &w));
  //   }

  //   #[test]
  //   #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  //   fn test_plonkish_simple() {
  //     let mut ccs = CCS::new_plonkish();

  //     // Add variables
  //     let x = ccs.add_variable();
  //     let y = ccs.add_variable();

  //     // Add constraint for x * y + 1 = 0
  //     let c1 = ccs.add_constraint();

  //     // Set up constraint using the new API
  //     ccs.set_multiplication(c1, F17::ONE, x, y); // x * y
  //     ccs.set_constant(c1, F17::ONE); // + 1

  //     println!("{ccs}");

  //     // 16 * 16 + 1 = 257 ≡ 0 (mod 17)
  //     let x = vec![];
  //     let w = vec![-F17::from(1), F17::from(1)];
  //     assert!(ccs.is_satisfied(&x, &w));
  //   }

  //   #[test]
  //   #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  //   fn test_multiple_constraints() {
  //     let mut ccs = CCS::new_plonkish();

  //     // Add variables
  //     let x = ccs.add_variable();
  //     let y = ccs.add_variable();
  //     let z = ccs.add_variable();

  //     // First constraint: x * y + z + 12 = 0
  //     let c1 = ccs.add_constraint();
  //     ccs.set_multiplication(c1, F17::ONE, x, y); // x * y
  //     ccs.set_linear(2, c1, F17::ONE, z); // + z (using matrix C)
  //     ccs.set_constant(c1, F17::from(12)); // + 12

  //     // Second constraint: y * z + x + 10 = 0
  //     let c2 = ccs.add_constraint();
  //     ccs.set_multiplication(c2, F17::ONE, y, z); // y * z
  //     ccs.set_linear(2, c2, F17::ONE, x); // + x (using matrix C instead of A)
  //     ccs.set_constant(c2, F17::from(10)); // + 10

  //     println!("{ccs}");

  //     // Test with valid assignment: (1,2,3)
  //     let x = vec![];
  //     let w = vec![F17::from(1), F17::from(2), F17::from(3)];

  //     // Manual verification
  //     // First constraint: 1 * 2 + 3 + 12 = 17 ≡ 0 (mod 17)
  //     // Second constraint: 2 * 3 + 1 + 10 = 17 ≡ 0 (mod 17)
  //     assert!(ccs.is_satisfied(&x, &w), "Valid assignment (1,2,3) should satisfy the constraints");

  //     // Test with invalid assignment: (1,1,1)
  //     let w_invalid = vec![F17::from(1), F17::from(1), F17::from(1)];
  //     assert!(
  //       !ccs.is_satisfied(&x, &w_invalid),
  //       "Invalid assignment (1,1,1) should not satisfy the constraints"
  //     );
  //   }
}

//! PLONK-style Customizable Constraint Systems (CCS).
//!
//! This module implements a specialized variant of CCS that builds on the PLONK
//! (Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge)
//! design pattern. The system represents arithmetic constraints using selector matrices
//! and coefficient vectors that work together to create polynomial equations.
//!
//! The constraint system has the form:
//!
//! ```text
//! sum_{i≤j} q_{i,j} (A_i z ∘ A_j z) + sum_i q_i (A_i z) + q_c = 0
//! ```
//!
//! where:
//! - A_i are selector matrices that each extract specific variables from the input vector z
//! - q_{i,j} are coefficient vectors for multiplication terms between any two selector matrices
//! - q_i are coefficient vectors for linear terms from each selector matrix
//! - q_c is a coefficient vector for constant terms
//! - z is the combined input vector containing both public inputs and witness values
//! - ∘ denotes the Hadamard (element-wise) product between vectors
//!
//! Each selector matrix A_i determines which variables participate in the constraint system.
//! The multiplication terms q_{i,j} allow creating products between any two selected variables,
//! while the linear terms q_i allow direct use of selected variables. The constant term q_c
//! completes the polynomial.
//!
//! For example, with a width of 4, we can create constraints involving:
//! - Multiplication terms: Any product x_i * x_j where i ≤ j
//! - Linear terms: Any variable x_i by itself
//! - Constants: Added directly to the equation
//!
//! # Features
//! - Flexible constraint creation through selector matrices
//! - Support for arbitrary width constraint systems
//! - Multiplication terms between any pair of selected variables
//! - Linear terms for direct variable use
//! - Constant terms for each constraint
//! - Multiple constraints sharing the same structure
//!
//! # Example
//! Here's how to create a system for the constraint x * y + z = 0:
//!
//! ```rust
//! use custom_constraints::{
//!   ccs::{plonkish::Plonkish, CCS},
//!   matrix::SparseMatrix,
//! };
//! # use ark_ff::{Field, Fp, MontBackend, MontConfig};
//! # #[derive(MontConfig)]
//! # #[modulus = "17"]
//! # #[generator = "3"]
//! # struct FConfig;
//! # type F = Fp<MontBackend<FConfig, 1>, 1>;
//!
//! // Create a width-3 system (allowing up to 3 variables per constraint)
//! let mut ccs = CCS::<Plonkish<F>, F>::new_width(3);
//! let c = ccs.add_constraint();
//!
//! // Set up matrices to select variables
//! let mut a1 = SparseMatrix::new_rows_cols(1, 3);
//! a1.write(0, 0, F::ONE); // Select x
//! ccs.matrices[0] = a1;
//!
//! let mut a2 = SparseMatrix::new_rows_cols(1, 3);
//! a2.write(0, 1, F::ONE); // Select y
//! ccs.matrices[1] = a2;
//!
//! let mut a3 = SparseMatrix::new_rows_cols(1, 3);
//! a3.write(0, 2, F::ONE); // Select z
//! ccs.matrices[2] = a3;
//!
//! // Set coefficients to create x * y + z = 0
//! ccs.set_multiplication_coefficient(0, 1, c, F::ONE); // x * y term
//! ccs.set_linear(2, c, F::ONE); // z term
//! ```
//!
//! This creates a system where:
//! 1. A₁ selects the x variable
//! 2. A₂ selects the y variable
//! 3. A₃ selects the z variable
//! 4. The coefficients combine these to form x * y + z = 0

use super::*;

/// A type marker for PLONK-style constraint systems.
///
/// This type configures a CCS to use vector-valued selectors suitable for
/// PLONK-style constraints where each selector holds coefficients for multiple
/// constraints.
#[derive(Clone, Debug, Default)]
pub struct Plonkish<F>(PhantomData<F>);
impl<F> CCSType<F> for Plonkish<F> {
  type Selectors = Vec<F>;
}

impl<F: Field> CCS<Plonkish<F>, F> {
  /// Creates a new basic PLONK-style constraint system.
  ///
  /// The system will have:
  /// - Three selector matrices A, B, C for linear terms
  /// - One multiplication term between A and B matrices
  /// - Linear terms for each selector matrix
  /// - A constant term
  pub fn new_plonkish() -> Self {
    let mut ccs = Self { matrices: Vec::new(), multisets: Vec::new(), selectors: Vec::new() };

    // Initialize 3 empty matrices (A, B, C)
    for _ in 0..3 {
      ccs.matrices.push(SparseMatrix::new_rows_cols(0, 0));
    }

    // Create multisets for all terms
    ccs.multisets.push(vec![0, 1]); // Multiplication term (A,B)
    ccs.multisets.push(vec![0]); // Linear term A
    ccs.multisets.push(vec![1]); // Linear term B
    ccs.multisets.push(vec![2]); // Linear term C
    ccs.multisets.push(vec![]); // Constant term

    // Initialize selector vectors
    ccs.selectors = vec![vec![]; 5]; // 1 mul + 3 linear + 1 const

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

    // Initialize new constraint coefficients to zero
    for selector in &mut self.selectors {
      selector.push(F::ZERO);
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
    // Set coefficient
    if let Some(selector) = self.selectors.get_mut(0) {
      if let Some(coeff) = selector.get_mut(constraint_idx) {
        *coeff = value;
      }
    }

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

    // Set coefficient
    if let Some(selector) = self.selectors.get_mut(matrix + 1) {
      if let Some(coeff) = selector.get_mut(constraint_idx) {
        *coeff = value;
      }
    }

    // Update matrix entry
    self.matrices[matrix].write(constraint_idx, var, F::ONE);
  }

  /// Sets the constant term for a specific constraint.
  pub fn set_constant(&mut self, constraint_idx: usize, value: F) {
    if let Some(selector) = self.selectors.last_mut() {
      if let Some(coeff) = selector.get_mut(constraint_idx) {
        *coeff = value;
      }
    }
  }

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
      let mul_term = self.selectors[0][row] * az[row] * bz[row];
      sum += mul_term;

      // Linear terms
      let lin_a = self.selectors[1][row] * az[row]; // ql×Az
      let lin_b = self.selectors[2][row] * bz[row]; // qr×Bz
      let lin_c = self.selectors[3][row] * cz[row]; // qo×Cz
      sum += lin_a + lin_b + lin_c;

      // Constant term
      sum += self.selectors[4][row];

      if sum != F::ZERO {
        return false;
      }
    }

    true
  }
}

impl<F: Field + Display> Display for CCS<Plonkish<F>, F> {
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
    if !self.selectors[0].iter().all(|&x| x == F::ZERO) {
      terms.push("qm·(Az·Bz)");
    }
    if !self.selectors[1].iter().all(|&x| x == F::ZERO) {
      terms.push("ql·Az");
    }
    if !self.selectors[2].iter().all(|&x| x == F::ZERO) {
      terms.push("qr·Bz");
    }
    if !self.selectors[3].iter().all(|&x| x == F::ZERO) {
      terms.push("qo·Cz");
    }
    if !self.selectors[4].iter().all(|&x| x == F::ZERO) {
      terms.push("qc");
    }

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
  fn test_plonkish_structure() {
    let ccs = CCS::<Plonkish<F17>, F17>::new_plonkish();

    // For width 3, we should have:
    // - 1 cross terms (0,1)
    // - 3 linear terms
    // - 1 constant term
    assert_eq!(ccs.multisets.len(), 5, "Should have 5 terms total");

    // Check cross term multisets
    assert_eq!(ccs.multisets[0], vec![0, 1], "First cross term incorrect");

    // Check linear term multisets
    assert_eq!(ccs.multisets[1], vec![0], "First linear term incorrect");
    assert_eq!(ccs.multisets[2], vec![1], "Second linear term incorrect");
    assert_eq!(ccs.multisets[3], vec![2], "Third linear term incorrect");
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_plonkish_display() {
    let mut ccs = CCS::<Plonkish<F17>, F17>::new_plonkish();

    // Add variables
    let x = ccs.add_variable();
    let y = ccs.add_variable();
    let z = ccs.add_variable();
    let w = ccs.add_variable();

    // Set up display for one constraint
    let c1 = ccs.add_constraint();

    // Set coefficients for: 3(x·y) + 4x + 5y + 6z + 7 = 0
    ccs.set_multiplication(c1, F17::from(3), x, y); // 3(x·y)
    ccs.set_linear(0, c1, F17::from(4), x); // + 4x
    ccs.set_linear(1, c1, F17::from(5), y); // + 5y
    ccs.set_linear(2, c1, F17::from(6), z); // + 6z
    ccs.set_constant(c1, F17::from(7)); // + 7

    println!("{ccs}");
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_plonkish_satisfaction() {
    let mut ccs = CCS::new_plonkish();

    // Add variables for x and y
    let x = ccs.add_variable();
    let y = ccs.add_variable();

    // Add constraint
    let c1 = ccs.add_constraint();

    // Set up constraint: x * y + 2x + 3y + 8 = 0
    ccs.set_multiplication(c1, F17::ONE, x, y); // x * y term
    ccs.set_linear(0, c1, F17::from(2), x); // 2x term
    ccs.set_linear(1, c1, F17::from(3), y); // 3y term
    ccs.set_linear(2, c1, F17::from(0), x); // no C term (using x as dummy var)
    ccs.set_constant(c1, F17::from(8)); // constant term

    println!("{ccs}");

    // With:
    // x = 4, y = 5
    // 4 * 5 + 2*4 + 3*5 + 8 = 51 ≡ 0 (mod 17)
    let x = vec![];
    let w = vec![F17::from(4), F17::from(5)];

    assert!(ccs.is_satisfied(&x, &w));

    // Test with invalid assignment
    let w = vec![F17::from(2), F17::from(3)];
    assert!(!ccs.is_satisfied(&x, &w));
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_plonkish_simple() {
    let mut ccs = CCS::new_plonkish();

    // Add variables
    let x = ccs.add_variable();
    let y = ccs.add_variable();

    // Add constraint for x * y + 1 = 0
    let c1 = ccs.add_constraint();

    // Set up constraint using the new API
    ccs.set_multiplication(c1, F17::ONE, x, y); // x * y
    ccs.set_constant(c1, F17::ONE); // + 1

    println!("{ccs}");

    // 16 * 16 + 1 = 257 ≡ 0 (mod 17)
    let x = vec![];
    let w = vec![-F17::from(1), F17::from(1)];
    assert!(ccs.is_satisfied(&x, &w));
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_multiple_constraints() {
    let mut ccs = CCS::new_plonkish();

    // Add variables
    let x = ccs.add_variable();
    let y = ccs.add_variable();
    let z = ccs.add_variable();

    // First constraint: x * y + z + 12 = 0
    let c1 = ccs.add_constraint();
    ccs.set_multiplication(c1, F17::ONE, x, y); // x * y
    ccs.set_linear(2, c1, F17::ONE, z); // + z (using matrix C)
    ccs.set_constant(c1, F17::from(12)); // + 12

    // Second constraint: y * z + x + 10 = 0
    let c2 = ccs.add_constraint();
    ccs.set_multiplication(c2, F17::ONE, y, z); // y * z
    ccs.set_linear(2, c2, F17::ONE, x); // + x (using matrix C instead of A)
    ccs.set_constant(c2, F17::from(10)); // + 10

    println!("{ccs}");

    // Test with valid assignment: (1,2,3)
    let x = vec![];
    let w = vec![F17::from(1), F17::from(2), F17::from(3)];

    // Manual verification
    // First constraint: 1 * 2 + 3 + 12 = 17 ≡ 0 (mod 17)
    // Second constraint: 2 * 3 + 1 + 10 = 17 ≡ 0 (mod 17)
    assert!(ccs.is_satisfied(&x, &w), "Valid assignment (1,2,3) should satisfy the constraints");

    // Test with invalid assignment: (1,1,1)
    let w_invalid = vec![F17::from(1), F17::from(1), F17::from(1)];
    assert!(
      !ccs.is_satisfied(&x, &w_invalid),
      "Invalid assignment (1,1,1) should not satisfy the constraints"
    );
  }
}

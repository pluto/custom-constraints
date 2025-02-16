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
  /// Creates a new PLONK-style constraint system with the specified width.
  ///
  /// The width parameter determines how many selector matrices and corresponding
  /// terms are created in the system. For a width n, the system will have:
  /// - n selector matrices A_0 through A_{n-1}, each capable of selecting variables from the input
  ///   vector
  /// - Multiplication terms q_{i,j} for all pairs where i ≤ j, allowing products between any two
  ///   (possibly same) selected variables
  /// - Linear terms q_i for each selector matrix, allowing direct use of selected variables
  /// - A constant term q_c that adds field elements to constraints
  ///
  /// # Arguments
  /// * `width` - Number of selector matrices to create (must be ≥ 2)
  ///
  /// # Panics
  /// Panics if width < 2, as PLONK-style systems need at least two matrices
  /// for multiplication terms
  ///
  /// # Examples
  /// ```
  /// # use custom_constraints::ccs::{plonkish::Plonkish, CCS};
  /// # use ark_ff::{Field, Fp, MontBackend, MontConfig};
  /// # #[derive(MontConfig)]
  /// # #[modulus = "17"]
  /// # #[generator = "3"]
  /// # struct FConfig;
  /// # type F = Fp<MontBackend<FConfig, 1>, 1>;
  /// let ccs = CCS::<Plonkish<F>, F>::new_width(3);
  /// // Creates a system with:
  /// // - 3 selector matrices
  /// // - 6 multiplication terms (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
  /// // - 3 linear terms
  /// // - 1 constant term
  /// ```
  pub fn new_width(width: usize) -> Self {
    assert!(width >= 2, "Width must be at least 2");
    let mut ccs = Self::default();

    // Initialize selector matrices
    for _ in 0..width {
      ccs.matrices.push(SparseMatrix::new_rows_cols(0, 0));
    }

    // Create multisets for all possible terms:
    // 1. Multiplication terms (i,j) where i ≤ j
    for i in 0..width {
      for j in i..width {
        ccs.multisets.push(vec![i, j]);
      }
    }
    // 2. Linear terms
    for i in 0..width {
      ccs.multisets.push(vec![i]);
    }
    // 3. Constant term
    ccs.multisets.push(vec![]);

    // Initialize selector vectors for all terms
    let num_selectors = (width * (width + 1)) / 2 + width + 1;
    ccs.selectors = vec![vec![]; num_selectors];

    ccs
  }

  /// Adds a new constraint to the system.
  ///
  /// This method:
  /// 1. Adds a new row to each selector matrix
  /// 2. Extends each selector vector with a zero coefficient
  ///
  /// The new constraint starts with all zero coefficients and can be configured
  /// using set_multiplication_coefficient, set_linear, and set_constant.
  ///
  /// # Returns
  /// The index of the new constraint, which can be used in subsequent coefficient
  /// setting operations
  ///
  /// # Examples
  /// ```
  /// # use custom_constraints::ccs::{plonkish::Plonkish, CCS};
  /// # use ark_ff::{Field, Fp, MontBackend, MontConfig};
  /// # #[derive(MontConfig)]
  /// # #[modulus = "17"]
  /// # #[generator = "3"]
  /// # struct FConfig;
  /// # type F = Fp<MontBackend<FConfig, 1>, 1>;
  /// let mut ccs = CCS::<Plonkish<F>, F>::new_width(2);
  /// let c1 = ccs.add_constraint(); // First constraint
  /// let c2 = ccs.add_constraint(); // Second constraint
  /// ```
  pub fn add_constraint(&mut self) -> usize {
    // Get current number of constraints
    let constraint_idx = self.matrices.first().map_or(0, |first| first.dimensions().0);

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

  /// Sets a multiplication coefficient for a specific constraint.
  ///
  /// This creates a term q_{i,j} * (A_i z ∘ A_j z) in the constraint equation,
  /// where:
  /// - q_{i,j} is the coefficient being set
  /// - A_i and A_j are selector matrices
  /// - z is the input vector
  /// - ∘ denotes the Hadamard (element-wise) product
  ///
  /// # Arguments
  /// * `i` - First matrix index
  /// * `j` - Second matrix index
  /// * `constraint_idx` - Which constraint to modify
  /// * `value` - Coefficient value to set
  ///
  /// # Panics
  /// Panics if i or j are out of bounds for the system's width
  pub fn set_multiplication_coefficient(
    &mut self,
    i: usize,
    j: usize,
    constraint_idx: usize,
    value: F,
  ) {
    let width = self.matrices.len();
    assert!(i < width && j < width, "Matrix index out of bounds");

    // Ensure i ≤ j for consistent indexing
    let (i, j) = if i <= j { (i, j) } else { (j, i) };

    // Calculate index for (i,j) pair
    // For each row k, we have (width-k) terms starting with (k,k)
    let idx = (i * (2 * width - i + 1)) / 2 + (j - i);

    if let Some(selector) = self.selectors.get_mut(idx) {
      if let Some(coeff) = selector.get_mut(constraint_idx) {
        *coeff = value;
      }
    }
  }

  /// Sets a linear term coefficient for a specific constraint.
  ///
  /// This creates a term q_i * (A_i z) in the constraint equation,
  /// where:
  /// - q_i is the coefficient being set
  /// - A_i is a selector matrix
  /// - z is the input vector
  ///
  /// # Arguments
  /// * `i` - Matrix index
  /// * `constraint_idx` - Which constraint to modify
  /// * `value` - Coefficient value to set
  ///
  /// # Panics
  /// Panics if i is out of bounds for the system's width
  pub fn set_linear(&mut self, i: usize, constraint_idx: usize, value: F) {
    let width = self.matrices.len();
    assert!(i < width, "Matrix index out of bounds");

    let num_mul_terms = (width * (width + 1)) / 2;
    let idx = num_mul_terms + i;

    if let Some(selector) = self.selectors.get_mut(idx) {
      if let Some(coeff) = selector.get_mut(constraint_idx) {
        *coeff = value;
      }
    }
  }

  /// Sets the constant term for a specific constraint.
  ///
  /// The constant term q_c is added directly to the constraint equation
  /// without any variable interaction.
  ///
  /// # Arguments
  /// * `constraint_idx` - Which constraint to modify
  /// * `value` - Constant value to set
  pub fn set_constant(&mut self, constraint_idx: usize, value: F) {
    if let Some(selector) = self.selectors.last_mut() {
      if let Some(coeff) = selector.get_mut(constraint_idx) {
        *coeff = value;
      }
    }
  }

  /// Checks if a witness and public input satisfy all constraints in the system.
  ///
  /// For each constraint, evaluates the equation:
  /// ```text
  /// sum_{i≤j} q_{i,j} (A_i z ∘ A_j z) + sum_i q_i (A_i z) + q_c = 0
  /// ```
  /// where z is the concatenation of public inputs x and witness values w.
  ///
  /// # Arguments
  /// * `x` - Public input values
  /// * `w` - Witness values
  ///
  /// # Returns
  /// `true` if all constraints evaluate to zero, `false` otherwise
  pub fn is_satisfied(&self, x: &[F], w: &[F]) -> bool {
    let mut z = Vec::with_capacity(x.len() + w.len());
    z.extend(x.iter().copied());
    z.extend(w.iter().copied());

    // Calculate matrix-vector products for each selector matrix
    let products: Vec<Vec<F>> = self.matrices.iter().map(|matrix| matrix * &z).collect();

    // If no constraints, system is trivially satisfied
    let num_constraints = products.first().map_or(0, |v| v.len());
    if num_constraints == 0 {
      return true;
    }

    let width = self.matrices.len();

    // Check each constraint
    for row in 0..num_constraints {
      let mut sum = F::ZERO;
      let mut selector_idx = 0;

      // Evaluate multiplication terms (i ≤ j)
      for i in 0..width {
        for j in i..width {
          if let Some(selector) = self.selectors.get(selector_idx) {
            let term = products[i][row] * products[j][row];
            sum += selector[row] * term;
          }
          selector_idx += 1;
        }
      }

      // Evaluate linear terms
      products.iter().take(width).zip(self.selectors.iter().skip(selector_idx)).for_each(
        |(product, selector)| {
          sum += selector[row] * product[row];
        },
      );

      // Add constant term
      if let Some(selector) = self.selectors.last() {
        sum += selector[row];
      }

      if sum != F::ZERO {
        return false;
      }
    }

    true
  }
}

impl<F: Field + Display> Display for CCS<Plonkish<F>, F> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    writeln!(f, "Plonkish Constraint System (width = {}):\n", self.matrices.len())?;

    // Display matrices
    writeln!(f, "Matrices:")?;
    for (i, matrix) in self.matrices.iter().enumerate() {
      writeln!(f, "A_{i} =")?;
      writeln!(f, "{matrix}")?;
    }

    // Display selectors
    writeln!(f, "\nSelectors:")?;
    let width = self.matrices.len();
    let mut idx = 0;

    // Display multiplication term selectors
    for i in 0..width {
      for j in i..width {
        write!(f, "q_{i},{j} = [")?;
        if let Some(selector) = self.selectors.get(idx) {
          for (k, &coeff) in selector.iter().enumerate() {
            if k > 0 {
              write!(f, ", ")?;
            }
            write!(f, "{coeff}")?;
          }
        }
        writeln!(f, "]")?;
        idx += 1;
      }
    }

    // Display linear term selectors
    for i in 0..width {
      write!(f, "q_{i} = [")?;
      if let Some(selector) = self.selectors.get(idx) {
        for (k, &coeff) in selector.iter().enumerate() {
          if k > 0 {
            write!(f, ", ")?;
          }
          write!(f, "{coeff}")?;
        }
      }
      writeln!(f, "]")?;
      idx += 1;
    }

    // Display constant term
    write!(f, "q_c = [")?;
    if let Some(selector) = self.selectors.last() {
      for (k, &coeff) in selector.iter().enumerate() {
        if k > 0 {
          write!(f, ", ")?;
        }
        write!(f, "{coeff}")?;
      }
    }
    writeln!(f, "]")?;

    // Display constraint equation
    writeln!(f, "\nConstraint equation:")?;
    let mut first_term = true;

    // Write multiplication terms
    idx = 0;
    for i in 0..width {
      for j in i..width {
        if let Some(selector) = self.selectors.get(idx) {
          if !selector.iter().all(|&x| x == F::ZERO) {
            if !first_term {
              write!(f, " + ")?;
            }
            write!(f, "q_{i},{j}·(A_{i}·z ∘ A_{j}·z)")?;
            first_term = false;
          }
        }
        idx += 1;
      }
    }

    // Write linear terms
    for i in 0..width {
      if let Some(selector) = self.selectors.get(idx) {
        if !selector.iter().all(|&x| x == F::ZERO) {
          if !first_term {
            write!(f, " + ")?;
          }
          write!(f, "q_{i}·(A_{i}·z)")?;
          first_term = false;
        }
      }
      idx += 1;
    }

    // Write constant term if non-zero
    if let Some(selector) = self.selectors.last() {
      if !selector.iter().all(|&x| x == F::ZERO) {
        if !first_term {
          write!(f, " + ")?;
        }
        write!(f, "q_c")?;
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
  fn test_plonkish_structure() {
    let ccs = CCS::<Plonkish<F17>, F17>::new_width(3);

    // For width 3, we should have:
    // - 6 cross terms (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
    // - 3 linear terms
    // - 1 constant term
    assert_eq!(ccs.multisets.len(), 10, "Should have 6 terms total");

    // Check cross term multisets
    assert_eq!(ccs.multisets[0], vec![0, 0], "First cross term incorrect");
    assert_eq!(ccs.multisets[1], vec![0, 1], "Second cross term incorrect");
    assert_eq!(ccs.multisets[2], vec![0, 2], "Third cross term incorrect");
    assert_eq!(ccs.multisets[3], vec![1, 1], "First cross term incorrect");
    assert_eq!(ccs.multisets[4], vec![1, 2], "Second cross term incorrect");
    assert_eq!(ccs.multisets[5], vec![2, 2], "Third cross term incorrect");

    // Check linear term multisets
    assert_eq!(ccs.multisets[6], vec![0], "First linear term incorrect");
    assert_eq!(ccs.multisets[7], vec![1], "Second linear term incorrect");
    assert_eq!(ccs.multisets[8], vec![2], "Third linear term incorrect");
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_plonkish_display() {
    let mut ccs = CCS::<Plonkish<F17>, F17>::new_width(2);

    // Set up display for one constraint
    ccs.add_constraint();

    // Set up test matrices
    let mut a1 = SparseMatrix::new_rows_cols(1, 4);
    a1.write(0, 0, F17::ONE);
    ccs.matrices[0] = a1;

    let mut a2 = SparseMatrix::new_rows_cols(1, 4);
    a2.write(0, 1, F17::ONE);
    ccs.matrices[1] = a2;

    // Set some coefficients
    ccs.set_multiplication_coefficient(0, 1, 0, F17::from(3)); // 3(A_1·z)(A_2·z)
    ccs.set_linear(0, 0, F17::from(4)); // 4(A_1·z)
    ccs.set_linear(1, 0, F17::from(5)); // 5(A_2·z)

    println!("{ccs}");
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_plonkish_satisfaction() {
    let mut ccs = CCS::<Plonkish<F17>, F17>::new_width(2);

    // Test one constraint
    ccs.add_constraint();

    // Set up matrices for x * y + 2x + 3y + 4 = 0
    let mut a1 = SparseMatrix::new_rows_cols(1, 2);
    a1.write(0, 0, F17::ONE); // Select x
    ccs.matrices[0] = a1;

    let mut a2 = SparseMatrix::new_rows_cols(1, 2);
    a2.write(0, 1, F17::ONE); // Select y
    ccs.matrices[1] = a2;

    // Set coefficients
    ccs.set_multiplication_coefficient(0, 1, 0, F17::ONE); // 1 * (x * y)
    ccs.set_linear(0, 0, F17::from(2)); // + 2x
    ccs.set_linear(1, 0, F17::from(3)); // + 3y
    ccs.set_constant(0, F17::from(8)); // + 4

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
    println!("x * y = {prod}");
    println!("2x = {lin1}");
    println!("3y = {lin2}");
    println!("constant = {constant}");
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

    // Test one simple constraint
    ccs.add_constraint();

    // Set up matrices for x * y + 1 = 0
    let mut a1 = SparseMatrix::new_rows_cols(1, 2);
    a1.write(0, 0, F17::ONE); // Select x
    ccs.matrices[0] = a1;

    let mut a2 = SparseMatrix::new_rows_cols(1, 2);
    a2.write(0, 1, F17::ONE); // Select y
    ccs.matrices[1] = a2;

    // Set coefficients
    ccs.set_multiplication_coefficient(0, 1, 0, F17::ONE); // x * y
    ccs.set_constant(0, F17::ONE); // + 1

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
    ccs.add_constraint();

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
    ccs.set_multiplication_coefficient(0, 1, 0, F17::ONE); // x * y
    ccs.set_multiplication_coefficient(1, 2, 0, F17::ONE); // y * z
    ccs.set_multiplication_coefficient(0, 2, 0, F17::ONE); // x * z

    // Set linear terms
    ccs.set_linear(0, 0, F17::from(2)); // 2x
    ccs.set_linear(1, 0, F17::from(3)); // 3y
    ccs.set_linear(2, 0, F17::from(4)); // 4z

    // Set constant term
    ccs.set_constant(0, -F17::from(4)); // - 4

    println!("ccs: {ccs}");

    // Let's print the computation
    println!("\nVerifying computation:");
    let xy = F17::from(2) * F17::from(3);
    let yz = F17::from(3) * F17::from(4);
    let xz = F17::from(2) * F17::from(4);
    let x_term = F17::from(2) * F17::from(2);
    let y_term = F17::from(3) * F17::from(3);
    let z_term = F17::from(4) * F17::from(4);
    let constant = -F17::from(4);

    println!("x * y = {xy}");
    println!("y * z = {yz}");
    println!("x * z = {xz}");
    println!("2x = {x_term}");
    println!("3y = {y_term}");
    println!("4z = {z_term}");
    println!("constant = {constant}");
    println!("sum = {}", xy + yz + xz + x_term + y_term + z_term + constant);

    let x = vec![];

    // Find solution where this equals 0 (mod 17)
    // Solution: x = 2, y = 3, z = 1
    let w = vec![F17::from(2), F17::from(3), F17::from(4)];
    assert!(ccs.is_satisfied(&x, &w));

    // Invalid assignment should fail
    let w = vec![F17::from(1), F17::from(1), F17::from(1)];
    assert!(!ccs.is_satisfied(&x, &w));
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_multiple_constraints() {
    let mut ccs = CCS::<Plonkish<F17>, F17>::new_width(3);

    // First constraint: x * y + z = 0
    let c1 = ccs.add_constraint();

    // Second constraint: y * z + x = 0
    let c2 = ccs.add_constraint();

    // Set up matrices
    let mut a1 = SparseMatrix::new_rows_cols(2, 3);
    a1.write(0, 0, F17::ONE); // x in first constraint
    a1.write(1, 0, F17::ONE); // x in second constraint
    ccs.matrices[0] = a1;

    let mut a2 = SparseMatrix::new_rows_cols(2, 3);
    a2.write(0, 1, F17::ONE); // y in first constraint
    a2.write(1, 1, F17::ONE); // y in second constraint
    ccs.matrices[1] = a2;

    let mut a3 = SparseMatrix::new_rows_cols(2, 3);
    a3.write(0, 2, F17::ONE); // z in first constraint
    a3.write(1, 2, F17::ONE); // z in second constraint
    ccs.matrices[2] = a3;

    // Set coefficients for first constraint: x * y + z + 12 = 0
    ccs.set_multiplication_coefficient(0, 1, c1, F17::ONE); // x * y
    ccs.set_linear(2, c1, F17::ONE); // + z
    ccs.set_constant(c1, F17::from(12)); // + 12

    // Set coefficients for second constraint: y * z + x + 10 = 0
    ccs.set_multiplication_coefficient(1, 2, c2, F17::ONE); // y * z
    ccs.set_linear(0, c2, F17::ONE); // + x
    ccs.set_constant(c2, F17::from(10)); // + 10

    println!("ccs: {ccs}");

    // Test with satisfying assignment
    // For first constraint: 1 * 2 + 3 + 12 ≡ 0 (mod 17)
    // For second constraint: 2 * 3 + 1 + 10 ≡ 0 (mod 17)
    let x = vec![];
    let w = vec![F17::from(1), F17::from(2), F17::from(3)];
    assert!(ccs.is_satisfied(&x, &w));

    // Test with invalid assignment
    let w = vec![F17::from(1), F17::from(1), F17::from(1)];
    assert!(!ccs.is_satisfied(&x, &w));
  }
}

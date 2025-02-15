use super::*;

#[derive(Clone, Debug, Default)]
pub struct Plonkish<F>(PhantomData<F>);
impl<F> CCSType<F> for Plonkish<F> {
  type Selectors = Vec<F>;
}

impl<F: Field> CCS<Plonkish<F>, F> {
  /// Creates a new Plonkish CCS with the specified width.
  /// The width determines the number of matrices A_i in the system.
  /// The minimum width is 2, corresponding to the form:
  /// q_M o (A_1 z o A_2 z) + q_1 o A_1 z + q_2 o A_2 z + q_c = 0
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

    // For width 2 as an example, we need:
    // 1. q_M o (A_1 z o A_2 z) -> multiset [0, 1]
    // 2. q_1 o A_1 z -> multiset [0]
    // 3. q_2 o A_2 z -> multiset [1]
    // 4. q_c -> empty multiset

    // Set up multisets for quadratic terms (all pairs)
    for i in 0..width {
      for j in i..width {
        ccs.multisets.push(vec![i, j]);
      }
    }

    // Set up multisets for linear terms
    for i in 0..width {
      ccs.multisets.push(vec![i]);
    }

    // Add constant term (empty multiset)
    ccs.multisets.push(vec![]);

    // Initialize selectors
    // Length should be total number of terms:
    // - Number of quadratic terms: width * (width + 1) / 2
    // - Number of linear terms: width
    // - One constant term
    let num_quadratic = (width * (width + 1)) / 2;
    let num_terms = num_quadratic + width + 1;
    ccs.selectors = vec![vec![F::ZERO]; num_terms];

    ccs
  }

  /// Helper method to set a quadratic term coefficient q_M[i,j]
  pub fn set_quadratic(&mut self, i: usize, j: usize, value: F) {
    let width = self.matrices.len();
    assert!(i < width && j < width, "Index out of bounds");

    // Calculate the index in the selectors vector for this quadratic term
    // We need to find where (i,j) lands in our ordered pairs
    let mut idx = 0;
    for x in 0..width {
      for y in x..width {
        if (x == i && y == j) || (x == j && y == i) {
          // Found our term
          if let Some(selector) = self.selectors.get_mut(idx) {
            selector[0] = value;
            return;
          }
        }
        idx += 1;
      }
    }
  }

  /// Helper method to set a linear term coefficient q_i
  pub fn set_linear(&mut self, i: usize, value: F) {
    let width = self.matrices.len();
    assert!(i < width, "Index out of bounds");

    // Linear terms come after all quadratic terms
    let num_quadratic = (width * (width + 1)) / 2;
    if let Some(selector) = self.selectors.get_mut(num_quadratic + i) {
      selector[0] = value;
    }
  }

  /// Helper method to set the constant term q_c
  pub fn set_constant(&mut self, value: F) {
    // Constant term is the last selector
    if let Some(last) = self.selectors.last_mut() {
      last[0] = value;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::mock::F17;

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_plonkish_ccs_width_2() {
    let mut ccs = CCS::<Plonkish<F17>, F17>::new_width(2);

    // Check number of matrices
    assert_eq!(ccs.matrices.len(), 2, "Should have 2 matrices for width 2");

    // For width 2, we should have:
    // - 3 quadratic terms (1,1), (1,2), (2,2)
    // - 2 linear terms
    // - 1 constant term
    assert_eq!(ccs.multisets.len(), 6, "Should have 6 terms total");

    // Check multisets structure
    assert_eq!(ccs.multisets[0], vec![0, 0], "First quadratic term incorrect");
    assert_eq!(ccs.multisets[1], vec![0, 1], "Cross term incorrect");
    assert_eq!(ccs.multisets[2], vec![1, 1], "Second quadratic term incorrect");
    assert_eq!(ccs.multisets[3], vec![0], "First linear term incorrect");
    assert_eq!(ccs.multisets[4], vec![1], "Second linear term incorrect");
    assert_eq!(ccs.multisets[5], vec![], "Constant term incorrect");

    // Test setting coefficients
    ccs.set_quadratic(0, 0, F17::from(2));
    ccs.set_quadratic(0, 1, F17::from(3));
    ccs.set_linear(0, F17::from(4));
    ccs.set_constant(F17::from(5));

    // Verify coefficients
    assert_eq!(ccs.selectors[0][0], F17::from(2), "Quadratic coefficient not set correctly");
    assert_eq!(ccs.selectors[1][0], F17::from(3), "Cross term not set correctly");
    assert_eq!(ccs.selectors[3][0], F17::from(4), "Linear coefficient not set correctly");
    assert_eq!(ccs.selectors[5][0], F17::from(5), "Constant term not set correctly");
  }

  #[test]
  #[should_panic(expected = "Width must be at least 2")]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_plonkish_ccs_width_1() {
    let _ccs = CCS::<Plonkish<F17>, F17>::new_width(1);
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_plonkish_ccs_width_3() {
    let ccs = CCS::<Plonkish<F17>, F17>::new_width(3);

    // For width 3, we should have:
    // - 6 quadratic terms (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
    // - 3 linear terms
    // - 1 constant term
    assert_eq!(ccs.multisets.len(), 10, "Should have 10 terms total");

    // Check first few multisets
    assert_eq!(ccs.multisets[0], vec![0, 0], "First quadratic term incorrect");
    assert_eq!(ccs.multisets[1], vec![0, 1], "First cross term incorrect");
    assert_eq!(ccs.multisets[2], vec![0, 2], "Second cross term incorrect");
  }
}

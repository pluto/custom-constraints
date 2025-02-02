use std::fmt::{self, Display, Formatter};

use matrix::SparseMatrix;

use super::*;

#[derive(Debug, Default)]
pub struct CCS<F: Field> {
  pub constants: Vec<F>,
  pub multisets: Vec<Vec<usize>>,
  pub matrices:  Vec<SparseMatrix<F>>,
}

impl<F: Field + std::fmt::Debug> CCS<F> {
  pub fn new() -> Self { Self::default() }

  pub fn is_satisfied(&self, w: Vec<F>, x: Vec<F>) -> bool {
    println!("\nBeginning CCS satisfaction check:");

    // Construct z = (w, 1, x)
    let mut z = Vec::with_capacity(w.len() + 1 + x.len());
    z.extend(w.iter().cloned());
    z.push(F::ONE);
    z.extend(x.iter().cloned());

    println!("Constructed z vector: {:?}", z);

    // Compute all matrix-vector products
    let products: Vec<Vec<F>> = self
      .matrices
      .iter()
      .enumerate()
      .map(|(i, matrix)| {
        let result = matrix * &z;
        println!("M{} · z = {:?}", i, result);
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
      println!("\nChecking row {}", row);
      let mut sum = F::ZERO;

      // For each constraint...
      for (i, multiset) in self.multisets.iter().enumerate() {
        println!("Processing constraint {} with multiset {:?}", i, multiset);

        // Get the Hadamard product of all matrices in this multiset
        let mut term = products[multiset[0]][row];
        println!("Starting with result from M{}: {:?}", multiset[0], term);

        // Multiply element-wise with remaining vectors
        for &idx in multiset.iter().skip(1) {
          term = term * products[idx][row];
          println!("After multiplying with M{}: {:?}", idx, term);
        }

        // Multiply by constant and add to sum
        let contribution = self.constants[i] * term;
        println!("Adding c{} * term = {:?} to sum", i, contribution);
        sum = sum + contribution;
        println!("Current sum: {:?}", sum);
      }

      if sum != F::ZERO {
        println!("Row {} failed: final sum {:?} ≠ 0\n", row, sum);
        return false;
      }
      println!("Row {} satisfied: final sum = 0\n", row);
    }

    println!("All constraints satisfied!");
    true
  }
}

impl<F: Field + Display> Display for CCS<F> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    writeln!(f, "Customizable Constraint System:")?;

    // First, display all matrices with their indices
    writeln!(f, "\nMatrices:")?;
    for (i, matrix) in self.matrices.iter().enumerate() {
      writeln!(f, "M{} =", i)?;
      writeln!(f, "{}", matrix)?;
    }

    // Show how constraints are formed from multisets and constants
    writeln!(f, "\nConstraints:")?;

    // We expect multisets to come in pairs, each pair forming one constraint
    for i in (0..self.multisets.len()).step_by(2) {
      // Write the constant for the first multiset
      write!(f, "{}·(", self.constants[i])?;

      // Write the Hadamard product for the first multiset
      if let Some(first_idx) = self.multisets[i].first() {
        write!(f, "M{}", first_idx)?;
        for &idx in &self.multisets[i][1..] {
          write!(f, "∘M{}", idx)?;
        }
      }
      write!(f, ")")?;

      // If we have a second multiset in the pair
      if i + 1 < self.multisets.len() {
        // Write the constant and Hadamard product for the second multiset
        write!(f, " + {}·(", self.constants[i + 1])?;
        if let Some(first_idx) = self.multisets[i + 1].first() {
          write!(f, "M{}", first_idx)?;
          for &idx in &self.multisets[i + 1][1..] {
            write!(f, "∘M{}", idx)?;
          }
        }
        write!(f, ")")?;
      }

      // Each constraint equals zero
      writeln!(f, " = 0")?;
    }
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
    let m1 = SparseMatrix::new_rows_cols(1, 4).write(0, 3, F17::ONE); // Select x
    let m2 = SparseMatrix::new_rows_cols(1, 4).write(0, 0, F17::ONE); // Select y
    let m3 = SparseMatrix::new_rows_cols(1, 4).write(0, 1, F17::ONE); // Select z

    println!("Created matrices:");
    println!("M1 (selects x): {:?}", m1);
    println!("M2 (selects y): {:?}", m2);
    println!("M3 (selects z): {:?}", m3);

    let mut ccs = CCS::new();
    ccs.matrices = vec![m1, m2, m3];
    // Encode x * y - z = 0
    ccs.multisets = vec![vec![0, 1], vec![2]];
    ccs.constants = vec![F17::ONE, F17::from(-1)];

    println!("\nTesting valid case: x=2, y=3, z=6");
    let x = vec![F17::from(2)]; // public input x = 2
    let w = vec![F17::from(3), F17::from(6)]; // witness y = 3, z = 6
    assert!(ccs.is_satisfied(w, x.clone()));

    println!("\nTesting invalid case: x=2, y=3, z=7");
    let w_invalid = vec![F17::from(3), F17::from(7)]; // witness y = 3, z = 7 (invalid)
    assert!(!ccs.is_satisfied(w_invalid, x));
  }
}

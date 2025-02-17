//! Integration with Noir circuit compilation and ACIR representation.
//!
//! This module provides functionality to convert Noir circuits into our CCS representation.
//! It handles the translation from ACIR (Abstract Circuit Intermediate Representation)
//! to a PLONK-style constraint system.
//!
//! # Overview
//! The conversion process involves:
//! 1. Reading a serialized Noir program
//! 2. Converting ACIR gates into our circuit representation
//! 3. Generating corresponding constraint systems
//!
//! Noir uses the bn254 field by default, so this module is primarily designed to work
//! with that field. While other fields may work, they are not officially supported
//! and may lead to incorrect coefficient conversions.
//!
//! # Example
//! ```ignore
//! use custom_constraints::noir::NoirProgram;
//! use ark_bn254::Fr;
//!
//! // Read compiled Noir program
//! let bin = std::fs::read("program.json").unwrap();
//! let program = NoirProgram::<Fr>::new(&bin);
//!
//! // Generate constraint system
//! let ccs = program.generate_constraints();
//! ```

// NOTE: This is required for Noir to work right basically.
use ark_bn254::Fr;

use acvm::acir::{
  self,
  acir_field::GenericFieldElement,
  circuit::{brillig::BrilligBytecode, Opcode, Program},
  native_types::WitnessMap,
};
use ark_ff::{AdditiveGroup, PrimeField};
use serde::{Deserialize, Serialize};

use super::*;
use crate::{
  ccs::{plonkish::Plonkish, CCS},
  matrix::SparseMatrix,
};

/// Represents a compiled Noir program with its bytecode.
///
/// This structure holds the serialized ACIR representation of a Noir program
/// and provides methods to convert it into our circuit and constraint system
/// representations.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NoirProgram {
  /// Raw bytecode for the Noir circuit
  #[serde(
    serialize_with = "Program::serialize_program_base64",
    deserialize_with = "Program::deserialize_program_base64"
  )]
  pub bytecode: Program<GenericFieldElement<Fr>>,
}

impl NoirProgram {
  /// Creates a new [`NoirProgram`] from serialized bytecode.
  ///
  /// # Arguments
  /// * `bin` - Serialized program bytes (typically from a .json file)
  pub fn new(bin: &[u8]) -> Self {
    serde_json::from_slice(bin).unwrap()
  }

  /// Returns the main circuit from the program.
  ///
  /// Noir programs can contain multiple functions, but we're primarily
  /// interested in the main circuit (functions[0]).
  pub fn circuit(&self) -> &acir::circuit::Circuit<GenericFieldElement<Fr>> {
    &self.bytecode.functions[0]
  }

  /// Returns any unconstrained functions in the program.
  pub fn unconstrained_functions(&self) -> &Vec<BrilligBytecode<GenericFieldElement<Fr>>> {
    &self.bytecode.unconstrained_functions
  }

  /// Generates a PLONK-style constraint system from the Noir program.
  ///
  /// This function converts ACIR gates directly into a constraint system where:
  /// 1. Each selector matrix identifies specific variables
  /// 2. Multiplication terms allow quadratic constraints
  /// 3. Linear terms capture direct variable usage
  /// 4. Constant terms complete the constraints
  pub fn generate_constraints(&self) -> CCS<Plonkish<Fr>, Fr> {
    let (mut ccs, width) = match self.circuit().expression_width {
      acir::circuit::ExpressionWidth::Unbounded => panic!("Unbounded width not supported"),
      acir::circuit::ExpressionWidth::Bounded { width } => (CCS::new_width(width), width),
    };

    // Initialize matrices for witness selection
    for i in 0..width {
      ccs.matrices[i] = SparseMatrix::new_rows_cols(0, self.circuit().num_vars() as usize);
    }

    // Process ACIR gates into constraints
    for opcode in &self.circuit().opcodes {
      if let Opcode::AssertZero(gate) = opcode {
        let constraint_idx = ccs.add_constraint();

        // Handle multiplication terms
        for (q_ij, wi, wj) in &gate.mul_terms {
          let i = wi.as_usize();
          let j = wj.as_usize();

          // Set up selector matrices
          ccs.matrices[i].write_expand(constraint_idx, i, Fr::ONE);
          ccs.matrices[j].write_expand(constraint_idx, j, Fr::ONE);

          // Set multiplication coefficient
          ccs.set_multiplication_coefficient(i, j, constraint_idx, q_ij.into_repr());
        }

        // Handle linear terms
        for (q_i, wi) in &gate.linear_combinations {
          let i = wi.as_usize();
          ccs.matrices[i].write_expand(constraint_idx, i, Fr::ONE);
          ccs.set_linear(i, constraint_idx, q_i.into_repr());
        }

        // Set constant term
        ccs.set_constant(constraint_idx, gate.q_c.into_repr());
      }
    }
    ccs
  }

  pub fn solve(
    &self,
    public_inputs: Vec<Fr>,
    private_inputs: Vec<Fr>,
  ) -> WitnessMap<GenericFieldElement<Fr>> {
    let mut acvm = acvm::pwg::ACVM::new(
      &acvm::blackbox_solver::StubbedBlackBoxSolver(false),
      &self.circuit().opcodes,
      acir::native_types::WitnessMap::new(),
      self.unconstrained_functions(),
      &[],
    );

    dbg!(self.circuit().public_parameters.0.len());
    dbg!(self.circuit().private_parameters.len());

    self.circuit().public_parameters.0.iter().for_each(|witness| {
      let f = GenericFieldElement::<Fr>::from_repr(public_inputs[witness.as_usize()]);
      acvm.overwrite_witness(*witness, f);
    });

    // write witness values for external_inputs
    self.circuit().private_parameters.iter().for_each(|witness| {
      let idx = witness.as_usize() - public_inputs.len();

      let f = GenericFieldElement::<Fr>::from_repr(private_inputs[idx]);
      acvm.overwrite_witness(*witness, f);
    });
    let _status = acvm.solve();
    acvm.finalize()
  }

  pub fn is_satisfied(&self, public_inputs: Vec<Fr>, private_inputs: Vec<Fr>) -> bool {
    let ccs = self.generate_constraints();
    let witness = self.solve(public_inputs, private_inputs);
    let witness = witness_map_as_vec(witness);
    ccs.is_satisfied(&[], &witness)
  }
}

fn witness_map_as_vec(witness_map: WitnessMap<GenericFieldElement<Fr>>) -> Vec<Fr> {
  // Find the maximum witness index to determine vector size
  let pairs: Vec<_> = witness_map.into_iter().collect();
  for (i, (witness, _)) in pairs.iter().enumerate() {
    if witness.as_usize() != i {
      panic!();
    }
  }

  // Create and fill our result vector
  pairs.into_iter().map(|(_, value)| value.into_repr()).collect()
}

#[cfg(test)]
mod tests {
  use std::path::Path;

  use super::*;
  // TODO: Can refactor
  fn program() -> NoirProgram {
    let json_path = Path::new("./tests/fixtures/basic.json");
    let bin = std::fs::read(json_path).unwrap();
    NoirProgram::new(&bin)
  }

  fn program_hash() -> NoirProgram {
    let json_path = Path::new("./tests/fixtures/hash.json");
    let bin = std::fs::read(json_path).unwrap();
    NoirProgram::new(&bin)
  }

  /// Tests conversion of a Noir program to our constraint system
  /// This test uses the example circuit:
  /// ```ignore
  /// pub fn main(x0: pub Field, w: [Field; 2]) -> pub Field {
  ///     1 * x0 * x0 + 2 * x0 * w[0] + 3 * x0 * w[1] +
  ///     4 * w[0] * w[0] + 5 * w[0] * w[1] + 6 * w[1] * w[1] +
  ///     7 * x0 + 8 * w[0] + 9 * w[1] + 10
  /// }
  /// ```
  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_generate_constraints() {
    let program = program();
    let ccs = program.generate_constraints();

    // Verify matrix dimensions and structure
    assert_eq!(ccs.matrices.len(), 4, "Should have 4 selector matrices");

    // Check matrix structure
    for (i, matrix) in ccs.matrices.iter().enumerate() {
      let (rows, cols) = matrix.dimensions();
      assert!(rows > 0, "Matrix {} should have rows", i);
      assert!(cols >= 4, "Matrix {} should have at least 4 columns", i);

      // Verify each matrix is properly selecting its variable
      // A_0 should select x0, A_1 should select w[0], etc.
      for _ in 0..rows {
        assert_eq!(
          matrix.dimensions().1,
          4,
          "Matrix should have exactly 4 columns (space for x0, w[0], w[1], output)"
        );

        // Each matrix should have exactly one 1 in its corresponding column
        assert_eq!(
          matrix * &vec![Fr::from(1), Fr::from(1), Fr::from(1), Fr::from(1)],
          vec![Fr::from(1); rows],
          "Matrix {} should select exactly one variable",
          i
        );
      }
    }

    // Now let's verify every coefficient from our polynomial
    let selectors = &ccs.selectors;

    // First, verify the quadratic terms
    // x0 * x0 term should have coefficient 1
    assert_eq!(
      selectors[0][0], // q_0,0 coefficient
      -Fr::from(1),
      "x0^2 term should have coefficient -1"
    );

    // x0 * w[0] term should have coefficient 2
    assert_eq!(
      selectors[1][0], // q_0,1 coefficient
      -Fr::from(2),
      "x0*w[0] term should have coefficient -2"
    );

    // x0 * w[1] term should have coefficient 3
    assert_eq!(
      selectors[2][0], // q_0,2 coefficient
      -Fr::from(3),
      "x0*w[1] term should have coefficient -3"
    );

    // w[0] * w[0] term should have coefficient 4
    assert_eq!(
      selectors[4][0], // q_1,1 coefficient
      -Fr::from(4),
      "w[0]^2 term should have coefficient -4"
    );

    // w[0] * w[1] term should have coefficient 5
    assert_eq!(
      selectors[5][0], // q_1,2 coefficient
      -Fr::from(5),
      "w[0]*w[1] term should have coefficient -5"
    );

    // w[1] * w[1] term should have coefficient 6
    assert_eq!(
      selectors[7][0], // q_2,2 coefficient
      -Fr::from(6),
      "w[1]^2 term should have coefficient -6"
    );

    // Verify linear terms
    let num_quad_terms = (4 * 5) / 2; // Number of quadratic terms

    // x0 term should have coefficient 7
    assert_eq!(selectors[num_quad_terms][0], -Fr::from(7), "x0 term should have coefficient -7");

    // w[0] term should have coefficient 8
    assert_eq!(
      selectors[num_quad_terms + 1][0],
      -Fr::from(8),
      "w[0] term should have coefficient -8"
    );

    // w[1] term should have coefficient 9
    assert_eq!(
      selectors[num_quad_terms + 2][0],
      -Fr::from(9),
      "w[1] term should have coefficient -9"
    );

    // Verify constant term
    assert_eq!(selectors.last().unwrap()[0], -Fr::from(10), "Constant term should be -10");

    // Verify the output variable's coefficient is 1
    assert_eq!(
      selectors[num_quad_terms + 3][0],
      Fr::from(1),
      "Output variable should have coefficient 1"
    );
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_solve_basic() {
    let program = program();
    let witness = program.solve(vec![Fr::from(1)], vec![Fr::from(2), Fr::from(3)]);
    dbg!(witness);
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_satisfied_basic() {
    let program = program();
    assert!(program.is_satisfied(vec![Fr::from(1)], vec![Fr::from(2), Fr::from(3)]));
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_solve_hash() {
    let program = program_hash();
    let witness = program.solve(vec![Fr::from(1); 32], vec![Fr::from(2); 16]);
    dbg!(witness);
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_satisfied_hash() {
    let program = program_hash();
    program.is_satisfied(vec![Fr::from(1); 32], vec![Fr::from(2); 16]);
  }
}

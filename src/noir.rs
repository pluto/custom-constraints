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
use std::collections::HashMap;

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
    let mut ccs = CCS::<Plonkish<Fr>, Fr>::new_plonkish();

    // First, add all variables
    let mut max_witness = 0;
    for opcode in &self.circuit().opcodes {
      if let Opcode::AssertZero(gate) = opcode {
        for (_, wi, wj) in &gate.mul_terms {
          max_witness = max_witness.max(wi.as_usize()).max(wj.as_usize());
        }
        for (_, wi) in &gate.linear_combinations {
          max_witness = max_witness.max(wi.as_usize());
        }
      }
    }

    println!("\nInitializing CCS with {} variables", max_witness + 1);

    // Add variables
    for _ in 0..=max_witness {
      ccs.add_variable();
    }

    // Process each ACIR gate
    for opcode in &self.circuit().opcodes {
      if let Opcode::AssertZero(gate) = opcode {
        let c = ccs.add_constraint();

        println!("\nConstructing constraint {}:", c);

        // First pass: Handle squared terms (w0^2, w1^2, w2^2)
        println!("\nSquared terms:");
        for (q_ij, wi, wj) in &gate.mul_terms {
          if wi.as_usize() == wj.as_usize() {
            let idx = wi.as_usize();
            let coeff = -q_ij.into_repr(); // Negate ACIR coefficient

            // For term like 4w0^2, we can write sqrt(4) to both A and B
            let sqrt_coeff = coeff.sqrt().unwrap_or(coeff);
            println!("  {} * w{} * w{} (sqrt = {})", coeff, idx, idx, sqrt_coeff);

            ccs.matrices[0].write(c, idx, sqrt_coeff);
            ccs.matrices[1].write(c, idx, sqrt_coeff);
          }
        }

        // Second pass: Handle cross terms (w0*w1, w1*w2, etc)
        println!("\nCross terms:");
        for (q_ij, wi, wj) in &gate.mul_terms {
          let wi_idx = wi.as_usize();
          let wj_idx = wj.as_usize();
          if wi_idx != wj_idx {
            let coeff = -q_ij.into_repr(); // Negate ACIR coefficient
            println!("  {} * w{} * w{}", coeff, wi_idx, wj_idx);

            // For cross terms, we can put the coefficient in B
            ccs.matrices[0].write(c, wi_idx, Fr::ONE);
            ccs.matrices[1].write(c, wj_idx, coeff);
          }
        }

        ccs.selectors[0][c] = Fr::ONE; // qm

        println!("\nLinear terms:");
        for (q_i, wi) in &gate.linear_combinations {
          let wi_idx = wi.as_usize();
          let coeff = q_i.into_repr();
          println!("  {} * w{}", coeff, wi_idx);
          ccs.matrices[2].write(c, wi_idx, coeff);
        }
        ccs.selectors[3][c] = Fr::ONE; // qo

        // Handle constant term
        ccs.selectors[4][c] = gate.q_c.into_repr();

        // Debug output
        println!("\nMatrix values:");
        println!("Matrix A: {:?}", (0..4).map(|i| ccs.matrices[0].get(c, i)).collect::<Vec<_>>());
        println!("Matrix B: {:?}", (0..4).map(|i| ccs.matrices[1].get(c, i)).collect::<Vec<_>>());
        println!("Matrix C: {:?}", (0..4).map(|i| ccs.matrices[2].get(c, i)).collect::<Vec<_>>());

        // Print expected terms when multiplied
        println!("\nExpected terms when multiplied:");
        for i in 0..4 {
          let a_val = ccs.matrices[0].get(c, i);
          for j in 0..4 {
            let b_val = ccs.matrices[1].get(c, j);
            if a_val != Fr::ZERO && b_val != Fr::ZERO {
              println!(
                "  ({} * w{}) * ({} * w{}) = {} * w{} * w{}",
                a_val,
                i,
                b_val,
                j,
                a_val * b_val,
                i,
                j
              );
            }
          }
        }
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

  // pub fn is_satisfied(&self, public_inputs: Vec<Fr>, private_inputs: Vec<Fr>) -> bool {
  //   let ccs = self.generate_constraints();
  //   let witness = self.solve(public_inputs, private_inputs);
  //   let witness = witness_map_as_vec(witness);
  //   ccs.is_satisfied(&[], &witness)
  // }
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

    // Verify basic structure
    assert_eq!(ccs.matrices.len(), 3, "Should have 3 matrices (A, B, C)");
    assert_eq!(ccs.selectors.len(), 5, "Should have 5 selectors (qm, ql, qr, qo, qc)");

    // The polynomial has one constraint:
    // 1*x0*x0 + 2*x0*w[0] + 3*x0*w[1] + 4*w[0]*w[0] + 5*w[0]*w[1] + 6*w[1]*w[1] +
    // 7*x0 + 8*w[0] + 9*w[1] + 10 = 0

    // Check dimensions
    let (rows, cols) = ccs.matrices[0].dimensions();
    assert!(rows > 0, "Should have at least one constraint");
    assert_eq!(cols, 4, "Should have space for x0, w[0], w[1], output");

    // Check multiplication terms (using matrices A and B with qm selector)
    let qm = &ccs.selectors[0]; // multiplication selector

    // Print matrices and selectors for debugging
    println!("Matrix A:\n{}", ccs.matrices[0]);
    println!("Matrix B:\n{}", ccs.matrices[1]);
    println!("Matrix C:\n{}", ccs.matrices[2]);
    println!("qm: {:?}", qm);
    println!("ql: {:?}", ccs.selectors[1]);
    println!("qr: {:?}", ccs.selectors[2]);
    println!("qo: {:?}", ccs.selectors[3]);
    println!("qc: {:?}", ccs.selectors[4]);

    assert!(ccs.is_satisfied(&[], &[Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(175)]));

    // // Check linear terms (using matrix C with qo selector)
    // let qo = &ccs.selectors[3]; // output selector

    // // Check that x0 has coefficient 7 in linear terms
    // let mut found_x0_term = false;
    // for row in 0..rows {
    //   if ccs.matrices[2].get(row, 0) == Some(&Fr::ONE) && qo[row] == -Fr::from(7) {
    //     found_x0_term = true;
    //     break;
    //   }
    // }
    // assert!(found_x0_term, "Should find linear term 7*x0");

    // // Check constant term
    // let qc = &ccs.selectors[4];
    // assert!(qc.contains(&-Fr::from(10)), "Should have constant term -10");

    // // Test satisfaction with valid assignment
    // let x = vec![];
    // let w = vec![Fr::from(1), Fr::from(2), Fr::from(3)]; // Example values
    // assert!(ccs.is_satisfied(&x, &w), "Valid assignment should satisfy constraints");

    // // Test with invalid assignment
    // let w_invalid = vec![Fr::from(0), Fr::from(0), Fr::from(0)];
    // assert!(!ccs.is_satisfied(&x, &w_invalid), "Invalid assignment should not satisfy constraints");
  }

  // #[test]
  // #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  // fn test_solve_basic() {
  //   let program = program();
  //   let witness = program.solve(vec![Fr::from(1)], vec![Fr::from(2), Fr::from(3)]);
  //   dbg!(witness);
  // }

  // #[test]
  // #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  // fn test_satisfied_basic() {
  //   let program = program();
  //   assert!(program.is_satisfied(vec![Fr::from(1)], vec![Fr::from(2), Fr::from(3)]));
  // }

  // #[test]
  // #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  // fn test_solve_hash() {
  //   let program = program_hash();
  //   let witness = program.solve(vec![Fr::from(1); 32], vec![Fr::from(2); 16]);
  //   dbg!(witness);
  // }

  // #[test]
  // #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  // fn test_satisfied_hash() {
  //   let program = program_hash();
  //   program.is_satisfied(vec![Fr::from(1); 32], vec![Fr::from(2); 16]);
  // }
}

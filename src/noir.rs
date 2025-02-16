use acvm::acir::{
  self,
  acir_field::GenericFieldElement,
  circuit::{brillig::BrilligBytecode, Opcode, Program},
};
use ark_ff::PrimeField;
use serde::{Deserialize, Serialize};

use super::*;
use crate::{
  ccs::{plonkish::Plonkish, CCS},
  circuit::{expression::Expression, Building, Circuit},
  matrix::SparseMatrix,
};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NoirProgram<F: Field + PrimeField> {
  #[serde(
    serialize_with = "Program::serialize_program_base64",
    deserialize_with = "Program::deserialize_program_base64"
  )]
  pub bytecode: Program<GenericFieldElement<F>>,
}

impl<F: Field + PrimeField> NoirProgram<F> {
  pub fn new(bin: &[u8]) -> Self { serde_json::from_slice(bin).unwrap() }

  pub fn circuit(&self) -> &acir::circuit::Circuit<GenericFieldElement<F>> {
    &self.bytecode.functions[0]
  }

  pub fn unconstrained_functions(&self) -> &Vec<BrilligBytecode<GenericFieldElement<F>>> {
    &self.bytecode.unconstrained_functions
  }

  pub fn generate_circuit(&self) -> Circuit<Building, F> {
    // ------------------------------------------------------------------------------------------------------------ //
    // Set up a circuit with the public and private inputs so the witness input will be (x,w,a)
    let mut circuit = Circuit::new();
    let public_inputs = self
      .circuit()
      .public_parameters
      .0
      .iter()
      .map(|x| circuit.x(x.as_usize()))
      .collect::<Vec<_>>();
    let private_inputs = self
      .circuit()
      .private_parameters
      .iter()
      .map(|w| circuit.w(w.as_usize() - circuit.pub_inputs))
      .collect::<Vec<_>>();
    let mut aux_vars = vec![];
    for _ in 0..(self.circuit().num_vars() as usize - public_inputs.len() - private_inputs.len()) {
      aux_vars.push(Expression::Variable(circuit.new_aux()));
    }
    let witnesses = [public_inputs, private_inputs, aux_vars].concat();
    dbg!(&witnesses);
    dbg!(&self.circuit().return_values);
    dbg!(self.circuit().num_vars());
    // TODO: This isn't properly marking output
    for return_values in &self.circuit().return_values.0 {
      circuit.mark_output(witnesses[return_values.as_usize()].clone());
    }

    // ------------------------------------------------------------------------------------------------------------ //

    let mut opcode_idx = 0;
    for opcode in &self.circuit().opcodes {
      println!("Opcode: {opcode_idx}");
      if let Opcode::AssertZero(gate) = opcode {
        // println!("gate: {gate}");
        dbg!(gate);
        let mut expr: Expression<F> = Circuit::constant(gate.q_c.into_repr());
        let mut mul_idx = 0;
        for (q_ij, wi, wj) in &gate.mul_terms {
          println!("Mul idx: {mul_idx}");
          expr = expr
            + Circuit::constant(q_ij.into_repr())
              * witnesses[wi.as_usize()].clone()
              * witnesses[wj.as_usize()].clone();
          mul_idx += 1;
        }

        for (q_i, wi) in &gate.linear_combinations {
          dbg!(wi);
          expr = expr + Circuit::constant(q_i.into_repr()) * witnesses[wi.as_usize()].clone();
        }
        circuit.add_internal(expr);

        opcode_idx += 1;
      }
      if let Opcode::MemoryInit { .. } | Opcode::MemoryOp { .. } = opcode {
        panic!("Memory Opcode was used! This is not currently supported.");
      }
    }
    circuit
  }

  pub fn generate_constraints(&self) -> CCS<Plonkish<F>, F> {
    let (mut ccs, width) = match self.circuit().expression_width {
      acir::circuit::ExpressionWidth::Unbounded => panic!("Can't handle unbounded right now"),
      acir::circuit::ExpressionWidth::Bounded { width } => (CCS::new_width(width), width),
    };

    // Initialize matrices for all potential witness selections
    for i in 0..width {
      ccs.matrices[i] = SparseMatrix::new_rows_cols(0, self.circuit().num_vars() as usize);
    }

    for opcode in &self.circuit().opcodes {
      if let Opcode::AssertZero(gate) = opcode {
        let constraint_idx = ccs.add_constraint();
        println!("Processing constraint {}", constraint_idx);

        // Handle multiplication terms
        for (q_ij, wi, wj) in &gate.mul_terms {
          println!("Setting multiplication term: q_{}_{} = {}", wi.as_usize(), wj.as_usize(), q_ij);
          // Write the coefficient directly - no need for cross-term conversion
          let i = wi.as_usize();
          let j = wj.as_usize();

          // Write 1 in the appropriate position in each matrix
          // Matrix A_i selects witness i
          ccs.matrices[i].write_expand(constraint_idx, i, F::ONE);
          // Matrix A_j selects witness j
          ccs.matrices[j].write_expand(constraint_idx, j, F::ONE);

          // Set the multiplication coefficient
          ccs.set_multiplication_coefficient(i, j, constraint_idx, q_ij.into_repr());
        }

        // Handle linear terms
        for (q_i, wi) in &gate.linear_combinations {
          println!("Setting linear term: q_{} = {}", wi.as_usize(), q_i);
          let i = wi.as_usize();
          // Matrix A_i selects witness i
          ccs.matrices[i].write_expand(constraint_idx, i, F::ONE);
          // Set the linear coefficient
          ccs.set_linear(i, constraint_idx, q_i.into_repr());
        }

        // Set constant term
        ccs.set_constant(constraint_idx, gate.q_c.into_repr());
      }
    }
    ccs
  }
}

#[cfg(test)]
mod tests {
  use std::path::Path;

  // TODO: I had to use this otherwise the coefficients are wrong? This is a known limitation: https://github.com/noir-lang/noir/issues/5055
  use ark_bn254::Fr;

  use super::*;
  use crate::circuit::expression::Variable;

  fn program() -> NoirProgram<Fr> {
    let json_path = Path::new("./examples/noir/target").join(format!("example.json"));
    dbg!(&json_path);
    let bin = std::fs::read(&json_path).unwrap();
    // TODO: This field might break everything
    NoirProgram::<Fr>::new(&bin)
  }

  #[test]
  fn test_generate_circuit() {
    let program = program();
    let circuit = program.generate_circuit();
    println!("\nExpanded forms:");
    for (expr, var) in circuit.expressions() {
      match var {
        Variable::Aux(idx) => println!("Auxiliary y_{} := {}", idx, expr),
        Variable::Output(idx) => println!("Output   o_{} := {}", idx, expr),
        _ => println!("Other    {} := {}", var, circuit.expand(expr)),
      }
    }
  }

  #[test]
  fn test_generate_constraints() {
    let program = program();
    let ccs = program.generate_constraints();
    println!("{ccs}")
  }
}

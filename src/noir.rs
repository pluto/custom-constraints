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

  fn generate_circuit(&self) -> Circuit<Building, F> {
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

  fn generate_constraints(&self) -> CCS<Plonkish<F>, F> {
    // let mut witness_map: HashMap<Witness, Variable> = HashMap::new();
    let (mut ccs, width) = match self.circuit().expression_width {
      acir::circuit::ExpressionWidth::Unbounded =>
        panic!("Can't handle unbounded right now -- not sure what's different though really."),
      acir::circuit::ExpressionWidth::Bounded { width } => (CCS::new_width(width), width),
    };

    // Set up the matrices to have the right number of columns
    // Admittedly, this is kinda jank and we should handle this better
    for i in 0..ccs.matrices.len() {
      ccs.matrices[i] = SparseMatrix::new_rows_cols(0, self.circuit().num_vars() as usize);
    }

    for opcode in &self.circuit().opcodes {
      if let Opcode::AssertZero(gate) = opcode {
        // Greedily add a constraint for each opcode (I think this is fine?)
        let constraint_idx = ccs.add_constraint();
        println!("Added constraint: {constraint_idx}");
        for (mul_idx, mul_term) in gate.mul_terms.iter().enumerate() {
          println!("Added mul: {mul_idx}");
          let i = mul_idx % (width - 1);
          let j = (mul_idx + 1) % (width - i) + i;
          // EX:
          // --> width = 4
          // mul_idx == 0
          // --> ( 0 % 3 = 0, (0 + 1) % (4 - 0) + 0 = 1)
          // mul_idx == 1
          // --> ( 1 % 3 = 0, (1 + 1) % (4 - 0) + 0 = 2)
          // mul_idx == 2
          // --> ( 2 % 3 = 0, (2 + 1) % (4 - 0) + 0 = 3)
          // mul_idx == 3
          // --> ( 3 % 3 = 1, (3 + 1) % (4 - 1) + 1 = 2)
          ccs.set_cross_term(i, j, constraint_idx, mul_term.0.into_repr());
          ccs.matrices[mul_idx].write_expand(constraint_idx, mul_term.1.as_usize(), F::ONE);
          ccs.matrices[mul_idx + 1].write_expand(constraint_idx, mul_term.2.as_usize(), F::ONE);
        }

        for (add_idx, add_term) in gate.linear_combinations.iter().enumerate() {
          ccs.set_linear(add_idx, constraint_idx, add_term.0.into_repr());
        }

        ccs.set_constant(constraint_idx, gate.q_c.into_repr());
      }
      if let Opcode::MemoryInit { .. } | Opcode::MemoryOp { .. } = opcode {
        panic!("Memory Opcode was used! This is not currently supported.");
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

  #[test]
  fn test_width() {
    let json_path = Path::new("./examples/noir/target").join(format!("example.json"));
    dbg!(&json_path);
    let bin = std::fs::read(&json_path).unwrap();
    // TODO: This field might break everything
    let program = NoirProgram::<Fr>::new(&bin);
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
}

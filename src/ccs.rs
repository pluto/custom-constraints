use matrix::SparseMatrix;

use super::*;

#[derive(Debug, Default)]
pub struct CCS<F: Field> {
  /// `m` in the paper, number of rows
  // rows:                usize,
  /// `n` in the paper, total number of inputs + 1
  // cols:                usize,
  /// `N` in the paper
  // nonzero_entries:     usize,
  /// `t` in the paper
  // number_of_matrices:  usize,
  /// `q` in the paper
  // number_of_multisets: usize,
  // constants: Vec<F>,
  // matrices:       Vec<SparseMatrix<F>>,
  constraints: Vec<Constraint<F>>,
  public_inputs:  usize,
  private_inputs: usize,
}

#[derive(Debug, Clone)]
pub enum Constraint<F> {
  Multiplication(Gate<F>),
  Addition(Gate<F>),
}

#[derive(Debug, Clone)]
pub struct Gate<F> {
  inputs:    Vec<Variable>,
  output:    VariableOrConstant<F>,
  constants: Vec<F>,
}

#[derive(Debug, Clone, Copy)]
pub enum Variable {
  PublicInput(usize),
  PrivateInput(usize),
  Aux(usize),
}

#[derive(Debug, Clone)]
pub enum VariableOrConstant<F> {
  Variable(Variable),
  Constant(F),
}

impl<F: Field> CCS<F> {
  pub fn new() -> Self { Self::default() }

  pub fn alloc_public_input(&mut self) { self.public_inputs += 1 }

  pub fn alloc_private_input(&mut self) { self.private_inputs += 1 }

  pub fn alloc_constraint(&mut self, constraint: Constraint<F>) {
    self.constraints.push(constraint);
  }

  pub fn is_satisfied(&self, public_inputs: Vec<F>, private_inputs: Vec<F>) -> bool {
    for constraint in &self.constraints {
      match constraint {
        Constraint::Addition(gate) => {
          if gate
            .inputs
            .iter()
            .zip(gate.constants.clone())
            .map(|(var, c)| match var {
              Variable::PublicInput(idx) => c * public_inputs[*idx],
              Variable::PrivateInput(idx) => c * private_inputs[*idx],
              Variable::Aux(_idx) => todo!(),
            })
            .sum::<F>()
            != match gate.output {
              VariableOrConstant::Constant(val) => val,
              VariableOrConstant::Variable(_) => todo!(),
            }
          {
            return false;
          }
        },
        Constraint::Multiplication(gate) => {
          if gate
            .inputs
            .iter()
            .zip(gate.constants.clone())
            .map(|(var, c)| match var {
              Variable::PublicInput(idx) => c * public_inputs[*idx],
              Variable::PrivateInput(idx) => c * private_inputs[*idx],
              Variable::Aux(_idx) => todo!(),
            })
            .product::<F>()
            != match gate.output {
              VariableOrConstant::Constant(val) => val,
              VariableOrConstant::Variable(_) => todo!(),
            }
          {
            return false;
          }
        },
      }
    }
    true
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_allocations() {
    let mut ccs = CCS::<F17>::new();
    ccs.alloc_private_input();
    ccs.alloc_public_input();

    let gate = Gate {
      inputs:    vec![Variable::PublicInput(0), Variable::PrivateInput(0)],
      constants: vec![F17::ONE, F17::ONE],
      output:    VariableOrConstant::Constant(F17::from(10)),
    };
    let constraint = Constraint::Multiplication(gate);
    ccs.alloc_constraint(constraint);
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_satisfy() {
    let mut ccs = CCS::<F17>::new();
    ccs.alloc_private_input();
    ccs.alloc_public_input();

    let gate = Gate {
      inputs:    vec![Variable::PublicInput(0), Variable::PrivateInput(0)],
      constants: vec![F17::ONE, F17::from(2)],
      output:    VariableOrConstant::Constant(F17::from(10)),
    };
    let constraint = Constraint::Multiplication(gate);
    ccs.alloc_constraint(constraint);

    assert!(ccs.is_satisfied(vec![F17::from(5)], vec![F17::from(1)]));

    assert!(!ccs.is_satisfied(vec![F17::from(1)], vec![F17::from(1)]));
  }
}

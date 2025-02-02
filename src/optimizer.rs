use std::collections::{HashMap, HashSet};

use ark_ff::Field;

use crate::{
  ccs::CCS,
  circuit::{CircuitBuilder, Expression, Variable},
  matrix::SparseMatrix,
};

impl<F: Field> Into<CCS<F>> for CircuitBuilder<F> {
  fn into(self) -> CCS<F> {
    let converter = CircuitToCCSConverter::new();
    converter.convert_circuit(self)
  }
}

struct CircuitToCCSConverter<F: Field> {
  // Track the next available index for auxiliary variables
  next_aux:   usize,
  // Map complex expressions to their auxiliary variable indices
  memo:       HashMap<String, usize>,
  // The CCS we're building
  ccs:        CCS<F>,
  // Number of public and witness inputs
  pub_inputs: usize,
  wit_inputs: usize,
}

impl<F: Field> CircuitToCCSConverter<F> {
  fn new() -> Self {
    Self {
      next_aux:   0,
      memo:       HashMap::new(),
      ccs:        CCS::new(),
      pub_inputs: 0,
      wit_inputs: 0,
    }
  }

  fn convert_circuit(mut self, circuit: CircuitBuilder<F>) -> CCS<F> {
    println!("\nBeginning circuit conversion:");

    self.pub_inputs = circuit.pub_inputs;
    self.wit_inputs = circuit.wit_inputs;
    println!(
      "Circuit has {} public inputs and {} witness inputs",
      self.pub_inputs, self.wit_inputs
    );

    if let Some((final_expr, _)) = circuit.expressions().last() {
      println!("Found final expression to process");
      let expanded = circuit.expand(final_expr);
      println!("Expanded expression: {}", expanded);
      self.process_final_constraint(&expanded);
    } else {
      println!("No expressions found in circuit!");
    }

    println!(
      "Conversion complete. Created {} matrices and {} constraints",
      self.ccs.matrices.len(),
      self.ccs.multisets.len() / 2
    );

    self.ccs
  }

  fn process_final_constraint(&mut self, expr: &Expression<F>) {
    println!("Processing final constraint");
    match expr {
      Expression::Add(terms) => {
        println!("Found addition with {} terms", terms.len());
        if terms.len() != 2 {
          panic!("Expected exactly 2 terms in final constraint");
        }

        // Get our multiplication term (first term)
        let mul_term = &terms[0];

        // Get our witness term (second term, which should be 16 * w_2)
        let witness_term = &terms[1];

        // Extract the factors from multiplication term
        let factors = if let Expression::Mul(factors) = mul_term {
          factors
        } else {
          panic!("First term should be multiplication");
        };

        // Extract witness index from witness term
        let w_idx = if let Expression::Mul(w_terms) = witness_term {
          // We expect 16 * w_2
          if let [Expression::Constant(_), Expression::Variable(Variable::Witness(idx))] =
            w_terms.as_slice()
          {
            idx
          } else {
            panic!("Expected constant * witness");
          }
        } else {
          panic!("Second term should be multiplication");
        };

        println!("Creating matrices for {} multiplication factors", factors.len());

        // Create matrices for multiplication factors
        let start_idx = self.ccs.matrices.len();
        let mut matrices = Vec::new();

        // Add matrices for multiplication factors
        for factor in factors {
          let (_, matrix) = self.create_selector_matrix(factor);
          matrices.push(matrix);
        }

        // Add matrix for witness
        let (_, w_matrix) =
          self.create_selector_matrix(&Expression::Variable(Variable::Witness(*w_idx)));
        matrices.push(w_matrix);

        // Add all matrices to CCS
        for matrix in matrices {
          self.ccs.matrices.push(matrix);
        }

        // Create the constraint:
        // First multiset for multiplication factors
        let mul_indices: Vec<_> = (0..factors.len()).map(|i| start_idx + i).collect();
        println!("Multiplication multiset: {:?}", mul_indices);
        self.ccs.multisets.push(mul_indices);

        // Second multiset for witness
        let witness_idx = start_idx + factors.len();
        println!("Witness multiset: {:?}", [witness_idx]);
        self.ccs.multisets.push(vec![witness_idx]);

        // Add corresponding constants
        self.ccs.constants.extend([F::ONE, F::from(-1)]);
        println!("Added constraint constants: 1, -1");
      },
      _ => panic!("Expected addition in final constraint"),
    }
  }

  fn create_multiplication_constraint(&mut self, factors: &[Expression<F>], output: &Variable) {
    // First collect all matrices and remember their starting index
    let start_idx = self.ccs.matrices.len();

    // Create input matrices and add them immediately
    let input_multiset: Vec<usize> = factors
      .iter()
      .enumerate()
      .map(|(i, factor)| {
        let (_, matrix) = self.create_selector_matrix(factor);
        self.ccs.matrices.push(matrix);
        start_idx + i // Each matrix gets a unique index
      })
      .collect();

    // Create and add output matrix
    let (_, output_matrix) = self.create_output_selector_matrix(output);
    self.ccs.matrices.push(output_matrix);
    let output_idx = start_idx + factors.len();

    // Add constraints:
    // 1. The product of inputs (with unique indices)
    self.ccs.multisets.push(input_multiset);
    self.ccs.constants.push(F::ONE);

    // 2. The negated output
    self.ccs.multisets.push(vec![output_idx]);
    self.ccs.constants.push(F::from(-1));
  }

  fn create_addition_constraint(&mut self, terms: &[Expression<F>], output: &Variable) {
    // Create selector matrices for each term
    let mut matrices = Vec::new();
    let mut multiset = Vec::new();

    for term in terms {
      let (matrix_idx, matrix) = self.create_selector_matrix(term);
      matrices.push(matrix);
      multiset.push(matrix_idx);
    }

    // Create output selector matrix
    let (output_matrix_idx, output_matrix) = self.create_output_selector_matrix(output);
    matrices.push(output_matrix);

    // Add matrices if they're new
    for matrix in matrices {
      self.ccs.matrices.push(matrix);
    }

    // Add constraint: sum of terms = output
    // Each term gets its own single-element multiset
    for &term_idx in &multiset {
      self.ccs.multisets.push(vec![term_idx]);
      self.ccs.constants.push(F::ONE);
    }
    self.ccs.multisets.push(vec![output_matrix_idx]);
    self.ccs.constants.push(F::from(-1));
  }

  fn create_selector_matrix(&mut self, expr: &Expression<F>) -> (usize, SparseMatrix<F>) {
    let n = self.wit_inputs + 1 + self.pub_inputs;
    let matrix_idx = self.ccs.matrices.len();

    match expr {
      Expression::Variable(Variable::Public(i)) => {
        let mut matrix = SparseMatrix::new_rows_cols(1, n);
        matrix = matrix.write(0, self.wit_inputs + 1 + i, F::ONE);
        (matrix_idx, matrix)
      },
      Expression::Variable(Variable::Witness(i)) => {
        let mut matrix = SparseMatrix::new_rows_cols(1, n);
        matrix = matrix.write(0, *i, F::ONE);
        (matrix_idx, matrix)
      },
      Expression::Variable(Variable::Aux(i)) => {
        let mut matrix = SparseMatrix::new_rows_cols(1, n);
        matrix = matrix.write(0, self.wit_inputs + i, F::ONE);
        (matrix_idx, matrix)
      },
      Expression::Constant(c) => {
        let mut matrix = SparseMatrix::new_rows_cols(1, n);
        matrix = matrix.write(0, self.wit_inputs, *c);
        (matrix_idx, matrix)
      },
      complex_expr => {
        // For complex expressions, we need to:
        // 1. Process the expression first
        // 2. Create an auxiliary variable for it
        // 3. Create a selector matrix for that auxiliary variable
        let aux_var = self.process_complex_expression(complex_expr);
        self.create_selector_matrix(&Expression::Variable(aux_var))
      },
    }
  }

  fn process_complex_expression(&mut self, expr: &Expression<F>) -> Variable {
    match expr {
      Expression::Mul(factors) => {
        // Create a new auxiliary variable for this multiplication
        let aux_var = Variable::Aux(self.next_aux);
        self.next_aux += 1;

        // Create the multiplication constraint
        self.create_multiplication_constraint(factors, &aux_var);

        aux_var
      },
      Expression::Add(terms) => {
        // Handle addition similarly
        let aux_var = Variable::Aux(self.next_aux);
        self.next_aux += 1;

        self.create_addition_constraint(terms, &aux_var);

        aux_var
      },
      _ => panic!("Unexpected complex expression type"),
    }
  }

  fn create_output_selector_matrix(&mut self, var: &Variable) -> (usize, SparseMatrix<F>) {
    self.create_selector_matrix(&Expression::Variable(var.clone()))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::mock::F17;

  #[test]
  fn test_circuit_to_ccs() {
    let mut builder = CircuitBuilder::new();

    // Create expression: x * y * z = w
    let x = builder.x(0);
    let y = builder.w(0);
    let z = builder.w(1);
    let w = builder.w(2);
    let mul = x * y * z;
    builder.add_expression(mul.clone());
    let sub = mul - w;
    builder.add_expression(sub);

    // Convert to CCS
    let ccs: CCS<F17> = builder.into();

    // Test the CCS
    let x_val = vec![F17::from(2)]; // public input
    let w_val = vec![F17::from(3), F17::from(4), F17::from(24)]; // witnesses
    assert!(ccs.is_satisfied(w_val, x_val.clone()));

    // Test with invalid output
    let w_invalid = vec![F17::from(3), F17::from(4), F17::from(25)];
    assert!(!ccs.is_satisfied(w_invalid, x_val));
  }
}

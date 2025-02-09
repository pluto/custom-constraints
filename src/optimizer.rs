// use std::collections::{HashMap, HashSet};

// use ark_ff::Field;

// use crate::{
//   ccs::CCS,
//   circuit::{CircuitBuilder, Expression, Variable},
//   matrix::SparseMatrix,
// };

// impl<F: Field> Into<CCS<F>> for CircuitBuilder<F> {
//   fn into(self) -> CCS<F> {
//     let converter = CircuitToCCSConverter::new();
//     converter.convert_circuit(self)
//   }
// }

// struct CircuitToCCSConverter<F: Field> {
//   // Track the next available index for auxiliary variables
//   next_aux:   usize,
//   // Map complex expressions to their auxiliary variable indices
//   memo:       HashMap<String, usize>,
//   // The CCS we're building
//   ccs:        CCS<F>,
//   // Number of public and witness inputs
//   pub_inputs: usize,
//   wit_inputs: usize,
// }

// impl<F: Field> CircuitToCCSConverter<F> {
//   fn new() -> Self {
//     Self {
//       next_aux:   0,
//       memo:       HashMap::new(),
//       ccs:        CCS::new(),
//       pub_inputs: 0,
//       wit_inputs: 0,
//     }
//   }

//   fn convert_circuit(mut self, circuit: CircuitBuilder<F>) -> CCS<F> {
//     // Initialize state from the circuit
//     self.pub_inputs = circuit.pub_inputs;
//     self.wit_inputs = circuit.wit_inputs;

//     // Process expressions to build constraints
//     for (expr, var) in circuit.expressions() {
//       let expanded = circuit.expand(expr);
//       self.process_expression(&expanded, *var);
//     }

//     self.ccs
//   }

//   // Create a matrix that selects a specific variable or constant
//   fn create_selector_matrix(&mut self, expr: &Expression<F>) -> (usize, SparseMatrix<F>) {
//     let n = self.wit_inputs + 1 + self.pub_inputs;
//     let matrix_idx = self.ccs.matrices.len();

//     match expr {
//       Expression::Variable(Variable::Public(i)) => {
//         let mut matrix = SparseMatrix::new_rows_cols(1, n);
//         matrix = matrix.write(0, self.wit_inputs + 1 + i, F::ONE);
//         (matrix_idx, matrix)
//       },
//       Expression::Variable(Variable::Witness(i)) => {
//         let mut matrix = SparseMatrix::new_rows_cols(1, n);
//         matrix = matrix.write(0, *i, F::ONE);
//         (matrix_idx, matrix)
//       },
//       Expression::Variable(Variable::Aux(i)) => {
//         let mut matrix = SparseMatrix::new_rows_cols(1, n);
//         matrix = matrix.write(0, self.wit_inputs + i, F::ONE);
//         (matrix_idx, matrix)
//       },
//       Expression::Constant(c) => {
//         let mut matrix = SparseMatrix::new_rows_cols(1, n);
//         matrix = matrix.write(0, self.wit_inputs, *c);
//         (matrix_idx, matrix)
//       },
//       // For complex expressions, create an aux variable and process it
//       complex_expr => {
//         // Create new auxiliary variable
//         let aux_var = Variable::Aux(self.next_aux);
//         self.next_aux += 1;

//         // Process the complex expression with this aux variable
//         self.process_expression(complex_expr, aux_var);

//         // Return selector matrix for the aux variable
//         let mut matrix = SparseMatrix::new_rows_cols(1, n);
//         matrix = matrix.write(0, self.wit_inputs + aux_var.as_aux_index(), F::ONE);
//         (matrix_idx, matrix)
//       },
//     }
//   }

//   // Process a multiplication with degree management
//   fn process_multiplication(&mut self, factors: &[Expression<F>], output: Variable) {
//     const MAX_DEGREE: usize = 3; // We can make this configurable later

//     if factors.len() <= MAX_DEGREE {
//       // Handle direct multiplication if within degree bound
//       self.create_multiplication_constraint(factors, &output);
//     } else {
//       // Split into smaller degree products
//       let (first_part, second_part) = factors.split_at(MAX_DEGREE);

//       // Create auxiliary variable for first part
//       let aux_var = Variable::Aux(self.next_aux);
//       self.next_aux += 1;

//       // Process first part
//       self.create_multiplication_constraint(first_part, &aux_var);

//       // Recursively process remaining factors
//       let mut remaining = vec![Expression::Variable(aux_var)];
//       remaining.extend(second_part.iter().cloned());
//       self.process_multiplication(&remaining, output);
//     }
//   }

//   // Create a constraint for multiplication
//   fn create_multiplication_constraint(&mut self, factors: &[Expression<F>], output: &Variable) {
//     let start_idx = self.ccs.matrices.len();

//     // Create matrices for input factors
//     let input_indices: Vec<usize> = factors
//       .iter()
//       .map(|factor| {
//         let (idx, matrix) = self.create_selector_matrix(factor);
//         self.ccs.matrices.push(matrix);
//         idx
//       })
//       .collect();

//     // Create output matrix
//     let (output_idx, output_matrix) =
// self.create_selector_matrix(&Expression::Variable(*output));     self.ccs.matrices.
// push(output_matrix);

//     // Add constraint: product of inputs = output
//     self.ccs.multisets.push(input_indices);
//     self.ccs.multisets.push(vec![output_idx]);
//     self.ccs.constants.extend([F::ONE, F::from(-1)]);
//   }

//   // Process any expression and create necessary constraints
//   fn process_expression(&mut self, expr: &Expression<F>, output: Variable) {
//     match expr {
//       Expression::Mul(factors) => {
//         self.process_multiplication(factors, output);
//       },
//       Expression::Add(terms) => {
//         // Create matrices for each term
//         let term_indices: Vec<usize> = terms
//           .iter()
//           .map(|term| {
//             let (idx, matrix) = self.create_selector_matrix(term);
//             self.ccs.matrices.push(matrix);
//             idx
//           })
//           .collect();

//         // Create output matrix
//         let (output_idx, output_matrix) =
//           self.create_selector_matrix(&Expression::Variable(output));
//         self.ccs.matrices.push(output_matrix);

//         // Add each term as its own multiset
//         for &term_idx in &term_indices {
//           self.ccs.multisets.push(vec![term_idx]);
//           self.ccs.constants.push(F::ONE);
//         }

//         // Subtract the output
//         self.ccs.multisets.push(vec![output_idx]);
//         self.ccs.constants.push(F::from(-1));
//       },
//       _ => {
//         // For simple expressions, just create an equality constraint
//         let (input_idx, input_matrix) = self.create_selector_matrix(expr);
//         let (output_idx, output_matrix) =
//           self.create_selector_matrix(&Expression::Variable(output));

//         self.ccs.matrices.push(input_matrix);
//         self.ccs.matrices.push(output_matrix);

//         self.ccs.multisets.push(vec![input_idx]);
//         self.ccs.multisets.push(vec![output_idx]);
//         self.ccs.constants.extend([F::ONE, F::from(-1)]);
//       },
//     }
//   }
// }

// impl Variable {
//   fn as_aux_index(&self) -> usize {
//     match self {
//       Variable::Aux(i) => *i,
//       _ => panic!("Expected auxiliary variable"),
//     }
//   }
// }

// #[cfg(test)]
// mod tests {
//   use super::*;
//   use crate::mock::F17;

//   #[test]
//   #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
//   fn test_circuit_to_ccs() {
//     let mut builder = CircuitBuilder::new();

//     // Create expression: x * y * z = w
//     let x = builder.x(0);
//     let y = builder.w(0);
//     let z = builder.w(1);
//     let w = builder.w(2);
//     let mul = x * y * z;
//     builder.add_expression(mul.clone());
//     let sub = mul - w;
//     builder.add_expression(sub);

//     // Convert to CCS
//     let ccs: CCS<F17> = builder.into();
//     println!("{ccs}");

//     // Test the CCS
//     let x_val = vec![F17::from(2)]; // public input
//     let w_val = vec![F17::from(3), F17::from(4), F17::from(24)]; // witnesses
//     assert!(ccs.is_satisfied(w_val, x_val.clone()));

//     // Test with invalid output
//     let w_invalid = vec![F17::from(3), F17::from(4), F17::from(25)];
//     assert!(!ccs.is_satisfied(w_invalid, x_val));
//   }

//   #[test]
//   fn test_other_circuit_to_ccs() {
//     let mut builder = CircuitBuilder::new();

//     // Create expression: x * y * z = w
//     let x = builder.x(0);
//     let y = builder.w(0);
//     let z = builder.w(1);
//     let w = builder.w(2);
//     let mul = x.clone() * y * z.clone() * z.clone() * z;
//     builder.add_expression(mul.clone());
//     let sub = mul - w * x;
//     builder.add_expression(sub);

//     // Convert to CCS
//     let ccs: CCS<F17> = builder.into();
//     println!("{ccs}");

//     // Test the CCS
//     let x_val = vec![F17::from(2)]; // public input
//     let w_val = vec![F17::from(3), F17::from(4), F17::from(24)]; // witnesses
//     assert!(ccs.is_satisfied(w_val, x_val.clone()));

//     // Test with invalid output
//     let w_invalid = vec![F17::from(3), F17::from(4), F17::from(25)];
//     assert!(!ccs.is_satisfied(w_invalid, x_val));
//   }
// }

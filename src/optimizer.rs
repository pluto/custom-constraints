// use std::collections::{HashMap, HashSet};

// use ark_ff::Field;

// use crate::{
//   ccs::CCS,
//   circuit::{CircuitBuilder, Expression, Variable},
// };

// // First, let's define a structure to represent our optimization state
// struct CCSOptimizer<F: Field> {
//   // Track expressions we've already processed to avoid duplicate constraints
//   memo:         HashMap<Expression<F>, Variable>,
//   // Track variable dependencies for topological sorting
//   dependencies: HashMap<Variable, HashSet<Variable>>,
//   // The resulting CCS we're building
//   ccs:          CCS<F>,
// }

// impl<F: Field> CCSOptimizer<F> {
//   fn new() -> Self { Self { memo: HashMap::new(), dependencies: HashMap::new(), ccs: CCS::new() }
// }

//   // Convert a circuit into CCS constraints
//   fn process_circuit(&mut self, circuit: &CircuitBuilder<F>) -> CCS<F> {
//     // First, expand all expressions to their base form
//     let expanded = circuit
//       .expressions()
//       .iter()
//       .map(|(expr, var)| (circuit.expand(expr), var))
//       .collect::<Vec<_>>();

//     // Process expressions in dependency order
//     for (expr, output_var) in expanded {
//       self.process_expression(&expr, output_var);
//     }

//     std::mem::take(&mut self.ccs)
//   }

//   // Process a single expression, potentially breaking it into multiple constraints
//   fn process_expression(&mut self, expr: &Expression<F>, output: &Variable) {
//     match expr {
//       // Base cases don't need processing
//       Expression::Variable(_) | Expression::Constant(_) => {},

//       Expression::Mul(factors) => {
//         // For multiplication, we want to maximize the degree of each constraint
//         // while respecting any implementation-specific maximum degree
//         const MAX_DEGREE: usize = 4; // Example maximum degree

//         let mut current_factors = Vec::new();
//         let mut current_var = None;

//         for factor in factors {
//           current_factors.push(factor.clone());

//           if current_factors.len() == MAX_DEGREE {
//             // Create intermediate result
//             let intermediate = self.ccs.alloc_aux_input();
//             let intermediate_var = Variable::Aux(intermediate);

//             // Create high-degree multiplication constraint
//             self.add_mul_constraint(&current_factors, &intermediate_var);

//             // Start new set of factors with this intermediate result
//             current_factors = vec![Expression::Variable(intermediate_var)];
//             current_var = Some(intermediate_var);
//           }
//         }

//         // Handle remaining factors
//         if !current_factors.is_empty() {
//           self.add_mul_constraint(&current_factors, output);
//         }
//       },

//       Expression::Add(terms) => {
//         // For addition, we want to batch as many terms as possible
//         // into a single constraint to minimize auxiliary variables
//         self.add_add_constraint(terms, output);
//       },
//     }
//   }

//   // Helper to create a multiplication constraint
//   fn add_mul_constraint(&mut self, factors: &[Expression<F>], output: &Variable) {
//     let constraint = Gate {
//       inputs:    factors.iter().map(|f| self.get_or_create_variable(f)).collect(),
//       output:    output.clone(),
//       constants: vec![F::ONE; factors.len()],
//     };

//     self.ccs.alloc_constraint(Constraint::Multiplication(constraint));
//   }

//   // Helper to create an addition constraint
//   fn add_add_constraint(&mut self, terms: &[Expression<F>], output: &Variable) {
//     let constraint = Gate {
//       inputs:    terms.iter().map(|t| self.get_or_create_variable(t)).collect(),
//       output:    output.clone(),
//       constants: vec![F::ONE; terms.len()],
//     };

//     self.ccs.alloc_constraint(Constraint::Addition(constraint));
//   }

//   // Helper to get or create a variable for an expression
//   fn get_or_create_variable(&mut self, expr: &Expression<F>) -> Variable {
//     match expr {
//       Expression::Variable(var) => var.clone(),
//       Expression::Constant(c) => {
//         // For constants, we could either create a public input
//         // or handle them specially in the constraint system
//         let idx = self.ccs.alloc_public_input();
//         Variable::Public(idx)
//       },
//       _ => {
//         // For complex expressions, we need to create an auxiliary
//         // variable and add appropriate constraints
//         let aux = self.ccs.alloc_aux_input();
//         let var = Variable::Aux(aux);
//         self.process_expression(expr, &var);
//         var
//       },
//     }
//   }
// }

// // Extension trait for CircuitBuilder to generate CCS
// impl<F: Field> CircuitBuilder<F> {
//   pub fn into_ccs(self) -> CCS<F> {
//     let mut optimizer = CCSOptimizer::new();
//     optimizer.process_circuit(&self)
//   }
// }

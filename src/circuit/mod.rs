use std::{
  collections::{HashMap, HashSet},
  fmt::{self, Display, Formatter},
  marker::PhantomData,
};

use ark_ff::Field;

use crate::{ccs::CCS, matrix::SparseMatrix};

pub mod expression;
#[cfg(test)] mod tests;

use self::expression::*;

#[derive(Debug)]
pub struct Building;

#[derive(Debug)]
pub struct DegreeConstrained<const DEGREE: usize>;

#[derive(Debug)]
pub struct Optimized<const DEGREE: usize>;

pub trait CircuitState {}

impl CircuitState for Building {}
impl<const DEGREE: usize> CircuitState for DegreeConstrained<DEGREE> {}
impl<const DEGREE: usize> CircuitState for Optimized<DEGREE> {}

#[derive(Debug, Clone)]
pub struct Circuit<S: CircuitState, F: Field> {
  pub_inputs:   usize,
  wit_inputs:   usize,
  aux_count:    usize,
  output_count: usize,
  expressions:  Vec<(Expression<F>, Variable)>,
  memo:         HashMap<String, Variable>,
  _marker:      PhantomData<S>,
}

impl<F: Field> Circuit<Building, F> {
  pub fn new() -> Self {
    Self {
      pub_inputs:   0,
      wit_inputs:   0,
      aux_count:    0,
      output_count: 0,
      expressions:  Vec::new(),
      memo:         HashMap::new(),
      _marker:      PhantomData,
    }
  }

  pub fn x(&mut self, i: usize) -> Expression<F> {
    assert!(i <= self.pub_inputs);
    self.pub_inputs = self.pub_inputs.max(i + 1);
    Expression::Variable(Variable::Public(i))
  }

  pub fn w(&mut self, i: usize) -> Expression<F> {
    assert!(i <= self.wit_inputs);
    self.wit_inputs = self.wit_inputs.max(i + 1);
    Expression::Variable(Variable::Witness(i))
  }

  pub const fn constant(c: F) -> Expression<F> { Expression::Constant(c) }

  pub fn add_internal(&mut self, expr: Expression<F>) -> Expression<F> {
    self.get_or_create_aux(&expr)
  }

  pub fn mark_output(&mut self, expr: Expression<F>) -> Expression<F> {
    match expr {
      Expression::Variable(Variable::Aux(aux_idx)) => {
        // Find and convert the specific auxiliary variable we want to change
        for (_, var) in self.expressions.iter_mut() {
          if *var == Variable::Aux(aux_idx) {
            *var = Variable::Output(self.output_count);
            break; // Found and converted the variable
          }
        }
        let output_idx = self.output_count;
        self.output_count += 1;
        self.aux_count -= 1; // Decrease aux count since we converted one
        Expression::Variable(Variable::Output(output_idx))
      },
      _ => {
        // For other expressions, create a new output variable
        let output_idx = self.output_count;
        let var = Variable::Output(output_idx);
        self.output_count += 1;
        self.expressions.push((expr, var));
        Expression::Variable(var)
      },
    }
  }

  const fn new_aux(&mut self) -> Variable {
    let var = Variable::Aux(self.aux_count);
    self.aux_count += 1;
    var
  }

  fn get_or_create_aux(&mut self, expr: &Expression<F>) -> Expression<F> {
    // Create a string representation of the expression for memoization
    let expr_key = format!("{expr}");

    if let Some(&var) = self.memo.get(&expr_key) {
      // We've seen this expression before, reuse the existing variable
      Expression::Variable(var)
    } else {
      // First time seeing this expression, create new auxiliary variable
      let var = self.new_aux();
      self.expressions.push((expr.clone(), var));
      self.memo.insert(expr_key, var);
      Expression::Variable(var)
    }
  }

  /// Reduces an expression to have maximum degree d by introducing auxiliary variables
  pub fn reduce_degree(&mut self, expr: Expression<F>, d: usize) -> Expression<F> {
    let current_degree = compute_degree(&expr);
    if current_degree <= d {
      return expr;
    }

    match expr {
      Expression::Mul(factors) => {
        let mut current_group = Vec::new();
        let mut current_group_degree = 0;
        let mut reduced_factors = Vec::new();

        for factor in factors {
          let factor_degree = compute_degree(&factor);

          if current_group_degree + factor_degree > d {
            if !current_group.is_empty() {
              let group_expr = if current_group.len() == 1 {
                current_group.pop().unwrap()
              } else {
                Expression::Mul(current_group.drain(..).collect())
              };
              // Use get_or_create_aux instead of directly adding
              reduced_factors.push(self.get_or_create_aux(&group_expr));
              current_group_degree = 0;
            }
          }

          let reduced_factor = self.reduce_degree(factor, d);
          let reduced_factor_degree = compute_degree(&reduced_factor);

          current_group.push(reduced_factor);
          current_group_degree += reduced_factor_degree;
        }

        if !current_group.is_empty() {
          let group_expr = if current_group.len() == 1 {
            current_group.pop().unwrap()
          } else {
            Expression::Mul(current_group)
          };
          reduced_factors.push(self.get_or_create_aux(&group_expr));
        }

        if reduced_factors.len() > 1 {
          self.reduce_degree(Expression::Mul(reduced_factors), d)
        } else {
          reduced_factors.pop().unwrap()
        }
      },
      Expression::Add(terms) => {
        let reduced_terms: Vec<_> =
          terms.into_iter().map(|term| self.reduce_degree(term, d)).collect();
        Expression::Add(reduced_terms)
      },
      _ => expr,
    }
  }

  // TODO: Remove clone
  // New method to transition to DegreeConstrained state
  pub fn fix_degree<const D: usize>(mut self) -> Circuit<DegreeConstrained<D>, F> {
    // First, collect all expressions we need to process
    let expressions_to_process: Vec<_> = self.expressions.clone();

    // Clear existing expressions since we'll rebuild them
    self.expressions.clear();

    // Process non-output expressions first
    for (expr, var) in expressions_to_process.iter() {
      match var {
        Variable::Output(_) => continue, // Handle outputs in second pass
        _ => {
          let reduced = self.reduce_degree(expr.clone(), D);
          self.expressions.push((reduced, *var));
        },
      }
    }

    // Now handle output expressions
    for (expr, var) in expressions_to_process.iter() {
      if let Variable::Output(_) = var {
        let reduced = self.reduce_degree(expr.clone(), D);
        self.expressions.push((reduced, *var));
      }
    }

    // Create the new degree-constrained circuit
    Circuit {
      pub_inputs:   self.pub_inputs,
      wit_inputs:   self.wit_inputs,
      aux_count:    self.aux_count,
      output_count: self.output_count,
      expressions:  self.expressions,
      memo:         self.memo,
      _marker:      PhantomData,
    }
  }
}

impl<const D: usize, F: Field> Circuit<DegreeConstrained<D>, F> {
  pub fn optimize(self) -> Circuit<Optimized<D>, F> {
    println!("\nStarting optimization process...");

    // Create a new building circuit
    let mut new_circuit = Circuit::<Building, F>::new();
    new_circuit.pub_inputs = self.pub_inputs;
    new_circuit.wit_inputs = self.wit_inputs;

    println!("\nInitial definitions:");
    let definitions: HashMap<Variable, Expression<F>> = self
      .expressions
      .iter()
      .map(|(expr, var)| {
        println!("{} := {} (degree {})", var, expr, compute_degree(expr));
        (*var, expr.clone())
      })
      .collect();

    // Map to track how we process auxiliary variables
    let mut aux_map = HashMap::new();

    fn process_expr<const D: usize, F: Field>(
      expr: &Expression<F>,
      definitions: &HashMap<Variable, Expression<F>>,
      aux_map: &mut HashMap<Variable, Expression<F>>,
      new_circuit: &mut Circuit<Building, F>,
      depth: usize, // Add depth parameter for indentation
    ) -> Expression<F> {
      let indent = "  ".repeat(depth);

      println!("{}Processing expression: {}", indent, expr);

      match expr {
        Expression::Variable(var @ Variable::Aux(_)) => {
          println!("{}Found auxiliary variable: {}", indent, var);

          // Check if we've processed this before
          if let Some(mapped_expr) = aux_map.get(var) {
            println!("{}Already processed, reusing: {}", indent, mapped_expr);
            return mapped_expr.clone();
          }

          // Get its definition
          if let Some(def) = definitions.get(var) {
            let degree = compute_degree(def);
            println!("{}Definition has degree {}: {}", indent, degree, def);

            if degree == D {
              println!("{}Creating new aux var for degree {} expression", indent, D);
              let new_expr =
                process_expr::<D, _>(def, definitions, aux_map, new_circuit, depth + 1);
              let result = new_circuit.add_internal(new_expr);
              println!("{}Created new aux var: {}", indent, result);
              aux_map.insert(*var, result.clone());
              result
            } else {
              println!("{}Using definition directly (degree < {})", indent, D);
              process_expr::<D, _>(def, definitions, aux_map, new_circuit, depth + 1)
            }
          } else {
            println!("{}No definition found, using as is", indent);
            expr.clone()
          }
        },
        Expression::Add(terms) => {
          println!("{}Processing addition with {} terms", indent, terms.len());
          let processed = Expression::Add(
            terms
              .iter()
              .map(|term| {
                let result =
                  process_expr::<D, _>(term, definitions, aux_map, new_circuit, depth + 1);
                println!("{}Processed term: {} -> {}", indent, term, result);
                result
              })
              .collect(),
          );
          println!("{}Addition result: {}", indent, processed);
          processed
        },
        Expression::Mul(factors) => {
          println!("{}Processing multiplication with {} factors", indent, factors.len());
          let processed = Expression::Mul(
            factors
              .iter()
              .map(|factor| {
                let result =
                  process_expr::<D, _>(factor, definitions, aux_map, new_circuit, depth + 1);
                println!("{}Processed factor: {} -> {}", indent, factor, result);
                result
              })
              .collect(),
          );
          println!("{}Multiplication result: {}", indent, processed);
          processed
        },
        _ => {
          println!("{}Base case: {}", indent, expr);
          expr.clone()
        },
      }
    }

    println!("\nProcessing output expressions:");
    let mut output_exprs = Vec::new();
    for (expr, var) in self.expressions {
      if let Variable::Output(_) = var {
        println!("\nProcessing output {}", var);
        let new_expr = process_expr::<D, _>(&expr, &definitions, &mut aux_map, &mut new_circuit, 1);
        println!("Output {} result: {}", var, new_expr);
        output_exprs.push((var, new_expr));
      }
    }

    println!("\nMarking outputs in new circuit:");
    for (var, expr) in output_exprs {
      println!("Marking output {} := {}", var, expr);
      new_circuit.mark_output(expr);
    }

    println!("\nFinal new circuit state:");
    for (expr, var) in &new_circuit.expressions {
      println!("{} := {} (degree {})", var, expr, compute_degree(expr));
    }

    // Convert to optimized circuit
    Circuit {
      pub_inputs:   new_circuit.pub_inputs,
      wit_inputs:   new_circuit.wit_inputs,
      aux_count:    new_circuit.aux_count,
      output_count: new_circuit.output_count,
      expressions:  new_circuit.expressions,
      memo:         new_circuit.memo,
      _marker:      PhantomData,
    }
  }
}

impl<const D: usize, F: Field> Circuit<Optimized<D>, F> {
  pub fn into_ccs(self) -> CCS<F> {
    let mut ccs = CCS::new_degree(D);

    // Calculate dimensions
    let num_cols = 1 + self.pub_inputs + self.wit_inputs + self.aux_count + self.output_count;

    // Initialize matrices
    for matrix in &mut ccs.matrices {
      *matrix = SparseMatrix::new_rows_cols(num_cols, num_cols);
    }

    // Process expressions with a more generic approach
    for (expr, var) in &self.expressions {
      let row = self.get_z_position(var);
      self.create_constraint(&mut ccs, D, row, expr, var);
    }

    ccs
  }

  // Modified create_constraint to handle generic degree
  fn create_constraint(
    &self,
    ccs: &mut CCS<F>,
    d: usize,
    row: usize,
    expr: &Expression<F>,
    output: &Variable,
  ) {
    // Write -1 times the output variable to the last matrix
    let output_pos = self.get_z_position(output);
    ccs.matrices.last_mut().unwrap().write(row, output_pos, -F::ONE);

    match expr {
      Expression::Add(terms) =>
        for term in terms {
          self.process_term(ccs, d, row, term);
        },
      _ => self.process_term(ccs, d, row, expr),
    }
  }

  // Generic process_term that handles any degree up to d
  fn process_term(&self, ccs: &mut CCS<F>, d: usize, row: usize, term: &Expression<F>) {
    match term {
      Expression::Mul(factors) => {
        let degree = factors.len();
        assert!(degree <= d, "Term degree exceeds maximum");

        // For each factor, we need to process it recursively
        let mut processed_factors = Vec::new();
        for factor in factors {
          match factor {
            Expression::Variable(_) | Expression::Constant(_) => {
              processed_factors.push(factor.clone());
            },
            Expression::Mul(inner_factors) => {
              // If a factor is itself a multiplication, we need to merge it
              processed_factors.extend(inner_factors.iter().cloned());
            },
            Expression::Add(terms) => {
              // If a factor is an addition, we need to distribute
              // This is a more complex case that we might want to handle differently
              // For now, we'll just collect all terms
              for term in terms {
                self.process_term(ccs, d, row, term);
              }
              return;
            },
          }
        }

        // Calculate starting matrix index based on degree
        let start_idx = match processed_factors.len() {
          n if n == D => 0, // Highest degree terms start at 0
          n => {
            // For degree k, start after all matrices used by higher degrees
            // For degree 3: 0
            // For degree 2: 3
            // For degree 1: 5
            let mut offset = 0;
            for deg in (n + 1)..=D {
              offset += deg;
            }
            offset
          },
        };
        for (i, factor) in processed_factors.iter().enumerate() {
          let pos = self.get_variable_position(factor);
          ccs.matrices[start_idx + i].write(row, pos, F::ONE);
        }
      },
      Expression::Add(terms) => {
        // Process each term independently
        for term in terms {
          self.process_term(ccs, d, row, term);
        }
      },
      _ => {
        // Base case: single variable or constant
        let pos = self.get_variable_position(term);
        ccs.matrices.last_mut().unwrap().write(row, pos, F::ONE);
      },
    }
  }
}

impl<S: CircuitState, F: Field> Circuit<S, F> {
  pub fn expressions(&self) -> &[(Expression<F>, Variable)] { &self.expressions }

  // TODO: Should this really only be some kind of `#[cfg(test)]` fn?
  pub fn expand(&self, expr: &Expression<F>) -> Expression<F> {
    match expr {
      // Base cases: constants and input variables remain unchanged
      Expression::Constant(_)
      | Expression::Variable(Variable::Public(_))
      | Expression::Variable(Variable::Witness(_)) => expr.clone(),

      // For auxiliary and output variables, look up their definition
      Expression::Variable(var @ (Variable::Aux(_) | Variable::Output(_))) => {
        if let Some(definition) = self.get_definition(var) {
          self.expand(definition)
        } else {
          expr.clone()
        }
      },

      // Recursively expand all subexpressions
      Expression::Add(terms) =>
        Expression::Add(terms.iter().map(|term| self.expand(term)).collect()),
      Expression::Mul(factors) =>
        Expression::Mul(factors.iter().map(|factor| self.expand(factor)).collect()),
    }
  }

  fn get_definition(&self, var: &Variable) -> Option<&Expression<F>> {
    match var {
      Variable::Aux(idx) | Variable::Output(idx) =>
        self.expressions.get(*idx).map(|(expr, _)| expr),
      _ => None,
    }
  }

  // Helper function to get position of a variable in z vector
  fn get_z_position(&self, var: &Variable) -> usize {
    match var {
      // Public inputs start at position 1
      Variable::Public(i) => 1 + i,
      // Witness variables follow public inputs
      Variable::Witness(i) => 1 + self.pub_inputs + i,
      // Auxiliary variables follow witness variables
      Variable::Aux(i) => 1 + self.pub_inputs + self.wit_inputs + i,
      // Output variables follow auxiliary variables
      Variable::Output(i) => 1 + self.pub_inputs + self.wit_inputs + self.aux_count + i,
    }
  }

  // Helper to get position of a variable in z vector
  fn get_variable_position(&self, expr: &Expression<F>) -> usize {
    match expr {
      Expression::Variable(var) => self.get_z_position(var),
      Expression::Constant(_) => 0,
      Expression::Mul(factors) => {
        // For a product, we need to handle each factor
        assert!(factors.len() == 1, "Expected simplified multiplication");
        self.get_variable_position(&factors[0])
      },
      Expression::Add(terms) => {
        // For a sum, we need to handle each term
        assert!(terms.len() == 1, "Expected simplified addition");
        self.get_variable_position(&terms[0])
      },
    }
  }
}

fn compute_degree<F: Field>(expr: &Expression<F>) -> usize {
  match expr {
    // Base cases: variables and constants have degree 1
    Expression::Variable(_) | Expression::Constant(_) => 1,

    // For addition, take the maximum degree of any term
    Expression::Add(terms) => terms.iter().map(|term| compute_degree(term)).max().unwrap_or(0),

    // For multiplication, sum the degrees of all factors
    Expression::Mul(factors) => factors.iter().map(|factor| compute_degree(factor)).sum(),
  }
}

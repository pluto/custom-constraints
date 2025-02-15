//! Circuit building and optimization for CCS.
//!
//! Provides a staged compilation pipeline:
//! 1. Building: Initial circuit construction
//! 2. DegreeConstrained: Circuit with enforced degree bounds
//! 3. Optimized: Circuit after optimization passes

use ccs::Generic;

use super::*;

use std::{collections::HashMap, marker::PhantomData};

use crate::{ccs::CCS, matrix::SparseMatrix};

pub mod expression;
#[cfg(test)]
mod tests;

use self::expression::*;

/// State marker for initial circuit construction.
#[derive(Debug)]
pub struct Building;

/// State marker for degree-constrained circuit.
#[derive(Debug)]
pub struct DegreeConstrained<const DEGREE: usize>;

/// State marker for optimized circuit.
#[derive(Debug)]
pub struct Optimized<const DEGREE: usize>;

/// Circuit state trait, implemented by state markers.
pub trait CircuitState {}

impl CircuitState for Building {}
impl<const DEGREE: usize> CircuitState for DegreeConstrained<DEGREE> {}
impl<const DEGREE: usize> CircuitState for Optimized<DEGREE> {}

/// An arithmetic circuit with typed state transitions.
#[derive(Debug, Clone, Default)]
pub struct Circuit<S: CircuitState, F: Field> {
  /// Number of public inputs
  pub pub_inputs: usize,
  /// Number of witness inputs  
  pub wit_inputs: usize,
  /// Number of auxiliary variables
  pub aux_count: usize,
  /// Number of output variables
  pub output_count: usize,
  /// Circuit expressions and their assigned variables
  expressions: Vec<(Expression<F>, Variable)>,
  /// Memoization cache for expressions
  memo: HashMap<String, Variable>,
  /// State type marker
  _marker: PhantomData<S>,
}

impl<F: Field> Circuit<Building, F> {
  /// Creates a new empty circuit.
  pub fn new() -> Self {
    Self {
      pub_inputs: 0,
      wit_inputs: 0,
      aux_count: 0,
      output_count: 0,
      expressions: Vec::new(),
      memo: HashMap::new(),
      _marker: PhantomData,
    }
  }

  /// Creates a public input variable x_i.
  pub fn x(&mut self, i: usize) -> Expression<F> {
    assert!(i <= self.pub_inputs);
    self.pub_inputs = self.pub_inputs.max(i + 1);
    Expression::Variable(Variable::Public(i))
  }

  /// Creates a witness variable w_i.
  pub fn w(&mut self, i: usize) -> Expression<F> {
    assert!(i <= self.wit_inputs);
    self.wit_inputs = self.wit_inputs.max(i + 1);
    Expression::Variable(Variable::Witness(i))
  }

  /// Creates a constant expression.
  pub const fn constant(c: F) -> Expression<F> {
    Expression::Constant(c)
  }

  /// Adds an internal auxiliary variable.
  pub fn add_internal(&mut self, expr: Expression<F>) -> Expression<F> {
    self.get_or_create_aux(&expr)
  }

  /// Marks an expression as a circuit output.
  pub fn mark_output(&mut self, expr: Expression<F>) -> Expression<F> {
    if let Expression::Variable(Variable::Aux(aux_idx)) = expr {
      // Find and convert the specific auxiliary variable we want to change
      for (_, var) in &mut self.expressions {
        if *var == Variable::Aux(aux_idx) {
          *var = Variable::Output(self.output_count);
          break; // Found and converted the variable
        }
      }
      let output_idx = self.output_count;
      self.output_count += 1;
      self.aux_count -= 1; // Decrease aux count since we converted one
      Expression::Variable(Variable::Output(output_idx))
    } else {
      // For other expressions, create a new output variable
      let output_idx = self.output_count;
      let var = Variable::Output(output_idx);
      self.output_count += 1;
      self.expressions.push((expr, var));
      Expression::Variable(var)
    }
  }

  // TODO: Remove clone
  /// Transitions circuit to degree-constrained state.
  pub fn fix_degree<const D: usize>(mut self) -> Circuit<DegreeConstrained<D>, F> {
    // First, collect all expressions we need to process
    let expressions_to_process: Vec<_> = self.expressions.clone();

    // Clear existing expressions since we'll rebuild them
    self.expressions.clear();

    // Process non-output expressions first
    for (expr, var) in &expressions_to_process {
      if let Variable::Output(_) = var {
        continue;
      }
      let reduced = self.reduce_degree(expr.clone(), D);
      self.expressions.push((reduced, *var));
    }

    // Now handle output expressions
    for (expr, var) in &expressions_to_process {
      if let Variable::Output(_) = var {
        let reduced = self.reduce_degree(expr.clone(), D);
        self.expressions.push((reduced, *var));
      }
    }

    Circuit {
      pub_inputs: self.pub_inputs,
      wit_inputs: self.wit_inputs,
      aux_count: self.aux_count,
      output_count: self.output_count,
      expressions: self.expressions,
      memo: self.memo,
      _marker: PhantomData,
    }
  }

  /// Creates a new auxiliary variable and increments the counter.
  const fn new_aux(&mut self) -> Variable {
    let var = Variable::Aux(self.aux_count);
    self.aux_count += 1;
    var
  }

  /// Returns existing auxiliary variable for expression or creates new one.
  ///
  /// Used for memoization/common subexpression elimination.
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

  /// Reduces expression degree through auxiliary variable introduction.
  ///
  /// # Arguments
  /// * `expr` - Expression to reduce
  /// * `d` - Target degree bound
  fn reduce_degree(&mut self, expr: Expression<F>, d: usize) -> Expression<F> {
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

          if current_group_degree + factor_degree > d && !current_group.is_empty() {
            let group_expr = if current_group.len() == 1 {
              current_group.pop().unwrap()
            } else {
              Expression::Mul(std::mem::take(&mut current_group))
            };
            reduced_factors.push(self.get_or_create_aux(&group_expr));
            current_group_degree = 0;
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
}

impl<const D: usize, F: Field> Circuit<DegreeConstrained<D>, F> {
  /// Converts circuit to CCS format.
  pub fn into_ccs(self) -> CCS<Generic<F>, F> {
    let mut ccs = CCS::new_degree(D);

    // Calculate dimensions
    let num_cols = 1 + self.pub_inputs + self.wit_inputs + self.aux_count + self.output_count;

    // TODO: Num rows does not need to equal the num cols
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

  /// Creates constraint for an expression at specified row.
  ///
  /// # Arguments
  /// * `ccs` - Target CCS
  /// * `d` - Maximum degree
  /// * `row` - Row index for constraint
  /// * `expr` - Expression to constrain
  /// * `output` - Output variable
  fn create_constraint(
    &self,
    ccs: &mut CCS<Generic<F>, F>,
    d: usize,
    row: usize,
    expr: &Expression<F>,
    output: &Variable,
  ) {
    // Write -1 times the output variable to the last matrix
    let output_pos = self.get_z_position(output);
    ccs.matrices.last_mut().unwrap().write(row, output_pos, -F::ONE);

    match expr {
      Expression::Add(terms) => {
        for term in terms {
          self.process_term(ccs, d, row, term);
        }
      },
      _ => self.process_term(ccs, d, row, expr),
    }
  }

  /// Processes term in constraint creation.
  fn process_term(&self, ccs: &mut CCS<Generic<F>, F>, d: usize, row: usize, term: &Expression<F>) {
    // First, fully expand the expression
    let expanded = expand_expression(term);

    match expanded {
      Expression::Add(terms) => {
        // Process each term in the addition
        for term in terms {
          self.process_simple_term(ccs, d, row, &term);
        }
      },
      _ => self.process_simple_term(ccs, d, row, &expanded),
    }
  }

  /// Processes a simple (non-compound) term.
  fn process_simple_term(
    &self,
    ccs: &mut CCS<Generic<F>, F>,
    d: usize,
    row: usize,
    term: &Expression<F>,
  ) {
    match term {
      Expression::Mul(factors) => {
        // Collect constants and variables
        let mut coefficient = F::ONE;
        let mut var_factors: Vec<_> = Vec::new();

        for factor in factors {
          match factor {
            Expression::Constant(c) => coefficient *= *c,
            Expression::Variable(_) => var_factors.push(factor),
            _ => panic!("Unexpected non-simple factor after expansion"),
          }
        }

        let degree = var_factors.len();
        assert!(degree <= d, "Term degree exceeds maximum");

        if degree == 0 {
          // Pure constant term goes in last matrix
          ccs.matrices.last_mut().unwrap().write(row, 0, coefficient);
        } else {
          // Calculate starting matrix index based on variable factors only
          let start_idx = if degree == d { 0 } else { (degree + 1..=d).sum() };

          // Write variable factors with coefficient on first one
          for (i, factor) in var_factors.iter().enumerate() {
            let pos = self.get_variable_position(factor);
            if i == 0 {
              ccs.matrices[start_idx + i].write(row, pos, coefficient);
            } else {
              ccs.matrices[start_idx + i].write(row, pos, F::ONE);
            }
          }
        }
      },
      Expression::Variable(_) => {
        // Single variable goes in last matrix
        let pos = self.get_variable_position(term);
        ccs.matrices.last_mut().unwrap().write(row, pos, F::ONE);
      },
      Expression::Constant(c) => {
        // Single constant goes in last matrix
        ccs.matrices.last_mut().unwrap().write(row, 0, *c);
      },
      Expression::Add(_) => panic!("Unexpected complex term after expansion"),
    }
  }

  /// Optimizes the circuit by eliminating auxiliary variables less than degree `D`
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
      pub_inputs: new_circuit.pub_inputs,
      wit_inputs: new_circuit.wit_inputs,
      aux_count: new_circuit.aux_count,
      output_count: new_circuit.output_count,
      expressions: new_circuit.expressions,
      memo: new_circuit.memo,
      _marker: PhantomData,
    }
  }
}

/// Expands expressions by distributing multiplication over addition.
fn expand_expression<F: Field>(expr: &Expression<F>) -> Expression<F> {
  match expr {
    Expression::Mul(factors) => {
      // First expand each factor
      let expanded_factors: Vec<_> = factors.iter().map(|f| expand_expression(f)).collect();

      // If any factor is an addition, we need to distribute
      let mut result = expanded_factors[0].clone();
      for factor in expanded_factors.iter().skip(1) {
        result = multiply_expressions(&result, factor);
      }
      result
    },
    Expression::Add(terms) => {
      // Expand each term and combine
      let expanded_terms: Vec<_> = terms.iter().map(|t| expand_expression(t)).collect();
      Expression::Add(expanded_terms)
    },
    // Variables and constants stay as they are
    _ => expr.clone(),
  }
}

/// Multiplies two expressions with distribution.
fn multiply_expressions<F: Field>(a: &Expression<F>, b: &Expression<F>) -> Expression<F> {
  match (a, b) {
    (Expression::Add(terms_a), _) => {
      // Distribute multiplication over addition
      let distributed: Vec<_> = terms_a.iter().map(|term| multiply_expressions(term, b)).collect();
      Expression::Add(distributed)
    },
    (_, Expression::Add(terms_b)) => {
      // Distribute multiplication over addition
      let distributed: Vec<_> = terms_b.iter().map(|term| multiply_expressions(a, term)).collect();
      Expression::Add(distributed)
    },
    (Expression::Mul(factors_a), Expression::Mul(factors_b)) => {
      // Combine the factors
      let mut new_factors = factors_a.clone();
      new_factors.extend(factors_b.clone());
      Expression::Mul(new_factors)
    },
    (Expression::Mul(factors), b) | (b, Expression::Mul(factors)) => {
      // Add the new factor to the existing ones
      let mut new_factors = factors.clone();
      new_factors.push(b.clone());
      Expression::Mul(new_factors)
    },
    (a, b) => Expression::Mul(vec![a.clone(), b.clone()]),
  }
}

impl<const D: usize, F: Field> Circuit<Optimized<D>, F> {
  /// Converts and `Optimized` circuit into CCS.
  pub fn into_ccs(self) -> CCS<Generic<F>, F> {
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

  /// Creates a constraint in the constraint system
  fn create_constraint(
    &self,
    ccs: &mut CCS<Generic<F>, F>,
    d: usize,
    row: usize,
    expr: &Expression<F>,
    output: &Variable,
  ) {
    // Write -1 times the output variable to the last matrix
    let output_pos = self.get_z_position(output);
    ccs.matrices.last_mut().unwrap().write(row, output_pos, -F::ONE);

    match expr {
      Expression::Add(terms) => {
        for term in terms {
          self.process_term(ccs, d, row, term);
        }
      },
      _ => self.process_term(ccs, d, row, expr),
    }
  }

  /// Processes term in constraint creation.
  fn process_term(&self, ccs: &mut CCS<Generic<F>, F>, d: usize, row: usize, term: &Expression<F>) {
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
  /// Returns circuit expressions.
  pub fn expressions(&self) -> &[(Expression<F>, Variable)] {
    &self.expressions
  }

  // TODO: Should this really only be some kind of `#[cfg(test)]` fn?
  /// Expands an expression by substituting definitions.
  pub fn expand(&self, expr: &Expression<F>) -> Expression<F> {
    match expr {
      // Base cases: constants and input variables remain unchanged
      Expression::Constant(_)
      | Expression::Variable(Variable::Public(_) | Variable::Witness(_)) => expr.clone(),

      // For auxiliary and output variables, look up their definition
      Expression::Variable(var @ (Variable::Aux(_) | Variable::Output(_))) => {
        self.get_definition(var).map_or_else(|| expr.clone(), |definition| self.expand(definition))
      },

      Expression::Add(terms) => {
        Expression::Add(terms.iter().map(|term| self.expand(term)).collect())
      },
      Expression::Mul(factors) => {
        Expression::Mul(factors.iter().map(|factor| self.expand(factor)).collect())
      },
    }
  }

  /// Gets definition for a variable if it exists.
  fn get_definition(&self, var: &Variable) -> Option<&Expression<F>> {
    match var {
      Variable::Aux(idx) | Variable::Output(idx) => {
        self.expressions.get(*idx).map(|(expr, _)| expr)
      },
      _ => None,
    }
  }

  /// Gets position of variable in z vector.
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

  /// Gets position of expression in z vector.
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

/// Computes the degree of an expression.
fn compute_degree<F: Field>(expr: &Expression<F>) -> usize {
  match expr {
    // Constants are degree 0
    Expression::Constant(_) => 0,
    // Base cases: variables degree 1
    Expression::Variable(_) => 1,

    // For addition, take the maximum degree of any term
    Expression::Add(terms) => terms.iter().map(|term| compute_degree(term)).max().unwrap_or(0),

    // For multiplication, sum the degrees of all factors
    Expression::Mul(factors) => factors.iter().map(|factor| compute_degree(factor)).sum(),
  }
}

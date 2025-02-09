use std::{
  collections::HashMap,
  fmt::{self, Display, Formatter},
};

use ark_ff::Field;

use crate::{ccs::CCS, matrix::SparseMatrix};

pub mod expression;
#[cfg(test)] mod tests;

use self::expression::*;

pub struct Building;
pub struct DegreeConstrained<const DEGREE: usize>;
pub struct Optimized<const DEGREE: usize>;

pub trait CircuitState {}

impl CircuitState for Building {}
impl<const DEGREE: usize> CircuitState for DegreeConstrained<DEGREE> {}
impl<const DEGREE: usize> CircuitState for Optimized<DEGREE> {}

#[derive(Clone, Debug)]
pub struct Circuit<F: Field> {
  pub pub_inputs:   usize,
  pub wit_inputs:   usize,
  pub aux_count:    usize,
  pub output_count: usize,
  pub expressions:  Vec<(Expression<F>, Variable)>,
  memo:             HashMap<String, Variable>,
}

impl<F: Field> Circuit<F> {
  pub fn new() -> Self {
    Self {
      pub_inputs:   0,
      wit_inputs:   0,
      aux_count:    0,
      output_count: 0,
      expressions:  Vec::new(),
      memo:         HashMap::new(),
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

  fn get_definition(&self, var: &Variable) -> Option<&Expression<F>> {
    match var {
      Variable::Aux(idx) | Variable::Output(idx) =>
        self.expressions.get(*idx).map(|(expr, _)| expr),
      _ => None,
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

  pub fn into_ccs(self, d: usize) -> CCS<F> {
    assert!(d >= 2, "Degree must be at least 2");

    // First create CCS with degree d
    let mut ccs = CCS::new_degree(d);

    // Calculate dimensions for our matrices:
    // - First column (index 0) is for constants (1)
    // - Then public inputs (x_i)
    // - Then witness inputs (w_i)
    // - Then auxiliary variables (y_i)
    // - Finally output variables (o_i)
    let num_cols = 1 + self.pub_inputs + self.wit_inputs + self.aux_count + self.output_count;

    // Initialize all matrices with proper dimensions
    // We need d matrices for degree d terms, 2 for degree 2 terms, and 1 for degree 1 terms
    for matrix in &mut ccs.matrices {
      *matrix = SparseMatrix::new_rows_cols(num_cols, num_cols);
    }

    // Process each expression to build our constraints
    for (expr, var) in &self.expressions {
      // The constraint for each variable goes in the row matching its position
      let row = self.get_z_position(var);
      self.create_constraint(&mut ccs, d, row, expr, var);
    }

    ccs
  }

  fn create_constraint(
    &self,
    ccs: &mut CCS<F>,
    d: usize,
    row: usize,
    expr: &Expression<F>,
    output: &Variable,
  ) {
    // Always write -1 times the output variable to the last matrix (M5 in degree 3 case)
    let output_pos = self.get_z_position(output);
    ccs.matrices.last_mut().unwrap().write(row, output_pos, -F::ONE);

    match expr {
      Expression::Add(terms) => {
        // For addition, process each term independently in the same row
        for term in terms {
          self.process_term(ccs, d, row, term);
        }
      },
      // Single term (not an addition)
      _ => self.process_term(ccs, d, row, expr),
    }
  }

  fn process_term(&self, ccs: &mut CCS<F>, d: usize, row: usize, term: &Expression<F>) {
    match term {
      Expression::Mul(factors) => {
        match factors.len() {
          n if n == d => {
            // Highest degree term - use first d matrices
            // Each factor goes into its corresponding matrix
            for (i, factor) in factors.iter().enumerate() {
              let pos = self.get_variable_position(factor);
              ccs.matrices[i].write(row, pos, F::ONE);
            }
          },
          2 => {
            // Quadratic term - use the next two matrices after the degree d matrices
            let offset = d; // Start after the degree d matrices
            for (i, factor) in factors.iter().enumerate() {
              let pos = self.get_variable_position(factor);
              ccs.matrices[offset + i].write(row, pos, F::ONE);
            }
          },
          1 => {
            // Linear term - use the last matrix
            let pos = self.get_variable_position(&factors[0]);
            ccs.matrices.last_mut().unwrap().write(row, pos, F::ONE);
          },
          _ => panic!("Term degree must be 1, 2, or {}", d),
        }
      },
      // Variables and constants go in the last matrix
      _ => {
        let pos = self.get_variable_position(term);
        ccs.matrices.last_mut().unwrap().write(row, pos, F::ONE);
      },
    }
  }

  // Helper to get position of a variable in z vector
  fn get_variable_position(&self, expr: &Expression<F>) -> usize {
    match expr {
      Expression::Variable(var) => self.get_z_position(var),
      Expression::Constant(_) => 0, // Constants are at position 0
      _ => panic!("Expected a variable or constant"),
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

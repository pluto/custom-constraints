use std::{
  collections::HashMap,
  fmt::{self, Display, Formatter},
};

use ark_ff::Field;

use crate::{ccs::CCS, matrix::SparseMatrix};

// The CircuitBuilder struct remains unchanged from your implementation
#[derive(Clone, Debug)]
pub struct CircuitBuilder<F: Field> {
  pub pub_inputs:   usize,
  pub wit_inputs:   usize,
  pub aux_count:    usize,
  pub output_count: usize,
  pub expressions:  Vec<(Expression<F>, Variable)>,
  // Add a memo map that stores expression strings to their corresponding variables
  memo:             HashMap<String, Variable>,
}

// Variable and Expression enums remain unchanged
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variable {
  Public(usize),
  Witness(usize),
  Aux(usize),
  Output(usize),
}

#[derive(Clone, Debug)]
pub enum Expression<F: Field> {
  Variable(Variable),
  Constant(F),
  Add(Vec<Expression<F>>),
  Mul(Vec<Expression<F>>),
}

impl<F: Field> CircuitBuilder<F> {
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
    self.pub_inputs = self.pub_inputs.max(i + 1);
    Expression::Variable(Variable::Public(i))
  }

  pub fn w(&mut self, i: usize) -> Expression<F> {
    self.wit_inputs = self.wit_inputs.max(i + 1);
    Expression::Variable(Variable::Witness(i))
  }

  pub fn constant(c: F) -> Expression<F> { Expression::Constant(c) }

  fn new_aux(&mut self) -> Variable {
    let var = Variable::Aux(self.aux_count);
    self.aux_count += 1;
    var
  }

  fn get_or_create_aux(&mut self, expr: &Expression<F>) -> Expression<F> {
    // Create a string representation of the expression for memoization
    let expr_key = format!("{}", expr);

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

  fn get_definition(&self, var: &Variable) -> Option<&Expression<F>> {
    match var {
      Variable::Aux(idx) | Variable::Output(idx) =>
        self.expressions.get(*idx).map(|(expr, _)| expr),
      _ => None,
    }
  }

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

  fn compute_degree(&self, expr: &Expression<F>) -> usize {
    match expr {
      // Base cases: variables and constants have degree 1
      Expression::Variable(_) | Expression::Constant(_) => 1,

      // For addition, take the maximum degree of any term
      Expression::Add(terms) =>
        terms.iter().map(|term| self.compute_degree(term)).max().unwrap_or(0),

      // For multiplication, sum the degrees of all factors
      Expression::Mul(factors) => factors.iter().map(|factor| self.compute_degree(factor)).sum(),
    }
  }

  /// Reduces an expression to have maximum degree d by introducing auxiliary variables
  pub fn reduce_degree(&mut self, expr: Expression<F>, d: usize) -> Expression<F> {
    let current_degree = self.compute_degree(&expr);
    if current_degree <= d {
      return expr;
    }

    match expr {
      Expression::Mul(factors) => {
        let mut current_group = Vec::new();
        let mut current_group_degree = 0;
        let mut reduced_factors = Vec::new();

        for factor in factors {
          let factor_degree = self.compute_degree(&factor);

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
          let reduced_factor_degree = self.compute_degree(&reduced_factor);

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
    // First create the CCS with the right degree
    let mut ccs = CCS::new_degree(d);

    // Calculate total size of z vector
    let z_len = 1 + self.pub_inputs + self.wit_inputs + self.aux_count + self.output_count;

    // Initialize all matrices with the right dimensions
    for matrix in ccs.matrices.iter_mut() {
      *matrix = SparseMatrix::new_rows_cols(z_len, z_len);
    }

    // Process each expression in our circuit
    for (expr, var) in self.expressions.iter() {
      // Get the row index where this constraint should appear
      let row_idx = self.get_z_position(var);

      // Create constraint for this expression
      //   self.create_constraint(&mut ccs, row_idx, expr, *var);
    }
    ccs
  }

  // Helper to get position of a value in expression
  fn get_variable_position(&self, expr: &Expression<F>) -> usize {
    match expr {
      Expression::Variable(var) => self.get_z_position(var),
      Expression::Constant(_) => 0,
      _ => panic!("Expected a variable or constant"),
    }
  }
}

// Existing operator implementations remain unchanged
impl<F: Field> std::ops::Add for Expression<F> {
  type Output = Expression<F>;

  fn add(self, rhs: Self) -> Self::Output {
    match (self, rhs) {
      (Expression::Add(mut v1), Expression::Add(v2)) => {
        v1.extend(v2);
        Expression::Add(v1)
      },
      (Expression::Add(mut v), rhs) => {
        v.push(rhs);
        Expression::Add(v)
      },
      (lhs, Expression::Add(mut v)) => {
        v.insert(0, lhs);
        Expression::Add(v)
      },
      (lhs, rhs) => Expression::Add(vec![lhs, rhs]),
    }
  }
}

impl<F: Field> std::ops::Mul for Expression<F> {
  type Output = Expression<F>;

  fn mul(self, rhs: Self) -> Self::Output {
    match (self, rhs) {
      (Expression::Mul(mut v1), Expression::Mul(v2)) => {
        v1.extend(v2);
        Expression::Mul(v1)
      },
      (Expression::Mul(mut v), rhs) => {
        v.push(rhs);
        Expression::Mul(v)
      },
      (lhs, Expression::Mul(mut v)) => {
        v.insert(0, lhs);
        Expression::Mul(v)
      },
      (lhs, rhs) => Expression::Mul(vec![lhs, rhs]),
    }
  }
}

impl<F: Field> std::ops::Neg for Expression<F> {
  type Output = Expression<F>;

  fn neg(self) -> Self::Output {
    // Negation is multiplication by -1
    Expression::Mul(vec![Expression::Constant(F::from(-1)), self])
  }
}

// Implement subtraction
impl<F: Field> std::ops::Sub for Expression<F> {
  type Output = Expression<F>;

  fn sub(self, rhs: Self) -> Self::Output {
    // a - b is the same as a + (-b)
    self + (-rhs)
  }
}

// Existing Display implementations remain unchanged
impl<F: Field + Display> Display for Expression<F> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    match self {
      Expression::Variable(var) => write!(f, "{}", var),
      Expression::Constant(c) => write!(f, "{}", c),
      Expression::Add(terms) => {
        write!(f, "(")?;
        for (i, term) in terms.iter().enumerate() {
          if i > 0 {
            write!(f, " + ")?;
          }
          write!(f, "{}", term)?;
        }
        write!(f, ")")
      },
      Expression::Mul(factors) => {
        write!(f, "(")?;
        for (i, factor) in factors.iter().enumerate() {
          if i > 0 {
            write!(f, " * ")?;
          }
          write!(f, "{}", factor)?;
        }
        write!(f, ")")
      },
    }
  }
}

impl Display for Variable {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    match self {
      Self::Public(i) => write!(f, "x_{}", i),
      Self::Witness(j) => write!(f, "w_{}", j),
      Self::Aux(k) => write!(f, "y_{}", k),
      Self::Output(l) => write!(f, "o_{}", l),
    }
  }
}

// TODO: all these tests really need to check things more strictly
#[cfg(test)]
mod tests {
  use super::*;
  use crate::mock::F17;

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_expression_arithmetic() {
    let mut builder = CircuitBuilder::new();

    // Create base values
    let x0 = builder.x(0);
    let x1 = builder.x(1);
    let w0 = builder.w(0);
    let three = CircuitBuilder::constant(F17::from(3));

    // Test negation: -x0
    let neg_x0 = -x0;
    let y0 = builder.add_internal(neg_x0);

    // Test subtraction: w0 - x1
    let sub_expr = w0 - x1;
    let y1 = builder.add_internal(sub_expr);

    // Test complex expression: 3 * (w0 - x1) - (-x0)
    let complex_expr = three * y1 - y0;
    let y2 = builder.add_internal(complex_expr);
    builder.mark_output(y2);

    println!("\nOriginal expressions:");
    for (expr, var) in builder.expressions() {
      if let Variable::Aux(idx) = var {
        println!("y_{} := {}", idx, expr);
      }
    }

    println!("\nExpanded forms:");
    for (expr, var) in builder.expressions() {
      match var {
        Variable::Aux(idx) => println!("Auxiliary y_{} := {}", idx, builder.expand(expr)),
        Variable::Output(idx) => println!("Output   o_{} := {}", idx, builder.expand(expr)),
        _ => println!("Other    {} := {}", var, builder.expand(expr)),
      }
    }
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_multiple_outputs() {
    let mut builder = CircuitBuilder::<F17>::new();

    // Let's create a circuit that computes several outputs
    let x = builder.x(0); // Public input x
    let y = builder.w(0); // Witness y
    let z = builder.w(1); // Witness z

    // First output: x * y
    let mul1 = x.clone() * y.clone();
    let o1 = builder.mark_output(mul1); // This should become o_0

    // Second output: y * z
    let mul2 = y * z;
    let o2 = builder.mark_output(mul2); // This should become o_1

    // Third output: x * o1 (using a previous output)
    let mul3 = x * o1;
    let o3 = builder.mark_output(mul3); // This should become o_2

    println!("\nMultiple outputs test:");
    for (expr, var) in builder.expressions() {
      match var {
        Variable::Output(idx) => println!("Output   o_{} := {}", idx, builder.expand(expr)),
        _ => println!("Other    {} := {}", var, builder.expand(expr)),
      }
    }

    // Verify we have the right number of outputs
    assert_eq!(builder.output_count, 3);
    assert_eq!(builder.aux_count, 0); // No auxiliary variables needed
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_aux_to_output_conversion() {
    let mut builder = CircuitBuilder::<F17>::new();

    // Create a more complex computation that needs auxiliary variables
    let x = builder.x(0);
    let y = builder.w(0);

    // Create some intermediate computations
    let square = x.clone() * x.clone();
    let aux1 = builder.add_internal(square); // y_0

    let cube = aux1.clone() * x.clone();
    let aux2 = builder.add_internal(cube); // y_1

    // Now convert both to outputs
    builder.mark_output(aux1); // Should become o_0
    builder.mark_output(aux2); // Should become o_1

    println!("\nAux to output conversion test:");
    for (expr, var) in builder.expressions() {
      match var {
        Variable::Output(idx) => println!("Output   o_{} := {}", idx, builder.expand(expr)),
        Variable::Aux(idx) => println!("Aux      y_{} := {}", idx, builder.expand(expr)),
        _ => println!("Other    {} := {}", var, builder.expand(expr)),
      }
    }

    // Verify our counts
    assert_eq!(builder.output_count, 2);
    assert_eq!(builder.aux_count, 0); // Both aux vars were converted
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_mixed_aux_and_output() {
    let mut builder = CircuitBuilder::<F17>::new();

    let x = builder.x(0);
    let y = builder.w(0);

    // Create an auxiliary computation we'll keep as auxiliary
    let square = x.clone() * x.clone();
    let aux1 = builder.add_internal(square); // y_0

    // Create an output directly
    let direct_output = y.clone() * y.clone();
    let o1 = builder.mark_output(direct_output); // o_0

    // Create another auxiliary and convert it
    let cube = aux1.clone() * x.clone();
    let aux2 = builder.add_internal(cube); // y_1
    builder.mark_output(aux2); // Converts to o_1

    println!("\nMixed aux and output test:");
    for (expr, var) in builder.expressions() {
      match var {
        Variable::Output(idx) => println!("Output   o_{} := {}", idx, builder.expand(expr)),
        Variable::Aux(idx) => println!("Aux      y_{} := {}", idx, builder.expand(expr)),
        _ => println!("Other    {} := {}", var, builder.expand(expr)),
      }
    }

    // Verify final state
    assert_eq!(builder.output_count, 2);
    assert_eq!(builder.aux_count, 1); // aux1 remains as auxiliary
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_reduce_degree() {
    let mut builder = CircuitBuilder::<F17>::new();

    // Create expression: x0 * x1 * x2 * x3
    let x0 = builder.x(0);
    let x1 = builder.x(1);
    let x2 = builder.x(2);
    let x3 = builder.x(3);
    let expr = x0 * x1 * x2 * x3; // degree 4

    // Reduce to degree 2
    let reduced = builder.reduce_degree(expr, 2);

    println!("Reduced expression: {}", reduced);
    println!("\nAuxiliary variables:");
    for (expr, var) in builder.expressions() {
      println!("{} := {}", var, expr);
    }
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_complex_degree_reduction() {
    let mut builder = CircuitBuilder::<F17>::new();

    // Let's evaluate two related polynomials:
    // P1(x,y) = (x^3 + y^2)^2 * (x + y)
    // P2(x,y) = x * y^4 + (x^3 + y^2)^3

    // Create our inputs
    let x = builder.x(0); // Public input x
    let y = builder.w(0); // Witness input y

    // First, let's build some common subexpressions
    // x^3 = x * x * x
    let x_cubed = x.clone() * x.clone() * x.clone();

    // y^2 = y * y
    let y_squared = y.clone() * y.clone();

    // (x^3 + y^2) - this is used in both polynomials
    let common_term = x_cubed.clone() + y_squared.clone();

    // Now build P1(x,y) = (x^3 + y^2)^2 * (x + y)
    let common_term_squared = common_term.clone() * common_term.clone(); // degree 4
    let x_plus_y = x.clone() + y.clone(); // degree 1
    let p1 = common_term_squared.clone() * x_plus_y; // Total degree: 5

    // Now build P2(x,y) = x * y^4 + (x^3 + y^2)^3
    let y_fourth = y_squared.clone() * y_squared.clone(); // degree 4
    let term1 = x.clone() * y_fourth; // degree 5
    let common_term_cubed = common_term * common_term_squared; // degree 6
    let p2 = term1 + common_term_cubed; // Max degree: 6

    // Let's reduce these to degree 3 expressions
    println!("\nReducing expressions to degree 3:");
    let p1_reduced = builder.reduce_degree(p1.clone(), 3);
    let p2_reduced = builder.reduce_degree(p2.clone(), 3);

    // Mark both as outputs
    builder.mark_output(p1_reduced);
    builder.mark_output(p2_reduced);

    // Print the full computation graph
    println!("\nOriginal P1: {}", p1);
    println!("Original P2: {}", p2);
    println!("\nAuxiliary and output variables:");
    for (expr, var) in builder.expressions() {
      match var {
        Variable::Aux(idx) => println!("y_{} := {}", idx, expr),
        Variable::Output(idx) => println!("o_{} := {}", idx, expr),
        _ => println!("{} := {}", var, expr),
      }
    }

    // Verify degrees of all expressions
    println!("\nVerifying degrees of all expressions:");
    for (expr, var) in builder.expressions() {
      let degree = builder.compute_degree(expr);
      println!("{} has degree {}", var, degree);
      assert!(degree <= 3, "Expression {} exceeds degree bound", var);
    }
  }
}

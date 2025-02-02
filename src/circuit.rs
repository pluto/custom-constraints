use std::fmt::{self, Display, Formatter};

use ark_ff::Field;

#[derive(Clone, Debug)]
pub struct CircuitBuilder<F: Field> {
  // Track input counts
  pub_inputs:  usize,
  wit_inputs:  usize,
  // Track intermediate values
  aux_count:   usize,
  // Store all expressions
  expressions: Vec<(Expression<F>, Variable)>,
}

// Variables that can be referenced in our circuit
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Variable {
  // x_i - public inputs
  Public(usize),
  // w_j - private witness values
  Witness(usize),
  // y_k - auxiliary/intermediate values
  Aux(usize),
}

// Core expression type that represents our arithmetic circuit
#[derive(Clone, Debug)]
pub enum Expression<F: Field> {
  // Terminal values
  Variable(Variable),
  Constant(F),
  // Operations over multiple terms
  Add(Vec<Expression<F>>),
  Mul(Vec<Expression<F>>),
}

impl<F: Field> CircuitBuilder<F> {
  pub fn new() -> Self {
    Self { pub_inputs: 0, wit_inputs: 0, aux_count: 0, expressions: Vec::new() }
  }

  // Helper functions to create input variables
  pub fn x(&mut self, i: usize) -> Expression<F> {
    self.pub_inputs = self.pub_inputs.max(i + 1);
    Expression::Variable(Variable::Public(i))
  }

  pub fn w(&mut self, i: usize) -> Expression<F> {
    self.wit_inputs = self.wit_inputs.max(i + 1);
    Expression::Variable(Variable::Witness(i))
  }

  pub fn constant(c: F) -> Expression<F> { Expression::Constant(c) }

  // Generate a new auxiliary variable
  fn new_aux(&mut self) -> Variable {
    let aux = Variable::Aux(self.aux_count);
    self.aux_count += 1;
    aux
  }

  // Add an expression to the circuit and return a variable referencing its output
  pub fn add_expression(&mut self, expr: Expression<F>) -> Expression<F> {
    let var = self.new_aux();
    self.expressions.push((expr, var.clone()));
    Expression::Variable(var)
  }

  // Get all constraints in the circuit
  pub fn expressions(&self) -> &[(Expression<F>, Variable)] { &self.expressions }
}

// Implement arithmetic operations for Expression
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

// Display implementations
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
      Variable::Public(i) => write!(f, "x_{}", i),
      Variable::Witness(j) => write!(f, "w_{}", j),
      Variable::Aux(k) => write!(f, "y_{}", k),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::mock::F17;

  #[test]
  fn test_complex_expressions() {
    let mut builder = CircuitBuilder::new();

    // Create base values
    let x0 = builder.x(0);
    let x1 = builder.x(1);
    let w0 = builder.w(0);
    let w1 = builder.w(1);
    let five = CircuitBuilder::constant(F17::from(5));

    // Create first expression: 5 * x_0 * w_0
    let expr1 = five * x0 * w0;
    let y0 = builder.add_expression(expr1.clone());

    // Create second expression: x_1 * w_1
    let expr2 = x1 * w1;
    let y1 = builder.add_expression(expr2.clone());

    // Create final expression using intermediate results: y_0 + y_1
    let expr3 = y0 + y1;
    builder.add_expression(expr3);

    // Verify the expressions were stored correctly
    let exprs = builder.expressions();
    assert_eq!(exprs.len(), 3);
    assert_eq!(exprs[0].1, Variable::Aux(0));
    assert_eq!(exprs[1].1, Variable::Aux(1));
    assert_eq!(exprs[2].1, Variable::Aux(2));

    println!("{}", exprs[0].0);
    println!("{}", exprs[1].0);
    println!("{}", exprs[2].0);

    // Check that builder tracked the input counts
    assert_eq!(builder.pub_inputs, 2);
    assert_eq!(builder.wit_inputs, 2);
    assert_eq!(builder.aux_count, 3);
  }
}

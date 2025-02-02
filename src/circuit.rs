use std::fmt::{self, Display, Formatter};

use ark_ff::Field;

// Core expression type that represents our arithmetic circuit
#[derive(Clone, Debug)]
pub enum Expression<F: Field> {
  // Terminal values
  Input(Input),
  Constant(F),
  // Operations over multiple terms
  Add(Vec<Expression<F>>),
  Mul(Vec<Expression<F>>),
}

// Input variables in our circuit
#[derive(Clone, Debug)]
pub enum Input {
  // x_i - public inputs
  Public(usize),
  // w_j - private witness values
  Witness(usize),
}

// Implementation for operator overloading
impl<F: Field> std::ops::Add for Expression<F> {
  type Output = Expression<F>;

  fn add(self, rhs: Self) -> Self::Output {
    match (self, rhs) {
      // If either side is already an Add, extend it
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
      // Otherwise create new Add expression
      (lhs, rhs) => Expression::Add(vec![lhs, rhs]),
    }
  }
}

impl<F: Field> std::ops::Mul for Expression<F> {
  type Output = Expression<F>;

  fn mul(self, rhs: Self) -> Self::Output {
    match (self, rhs) {
      // If either side is already a Mul, extend it
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
      // Otherwise create new Mul expression
      (lhs, rhs) => Expression::Mul(vec![lhs, rhs]),
    }
  }
}

// Convenience conversion from Input to Expression
impl<F: Field> From<Input> for Expression<F> {
  fn from(input: Input) -> Self { Expression::Input(input) }
}

// Convenience conversion from Field element to Expression
impl<F: Field> From<F> for Expression<F> {
  fn from(value: F) -> Self { Expression::Constant(value) }
}

// Display implementation for pretty-printing
impl<F: Field + Display> Display for Expression<F> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    match self {
      Expression::Input(input) => write!(f, "{}", input),
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

impl Display for Input {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    match self {
      Input::Public(i) => write!(f, "x_{}", i),
      Input::Witness(j) => write!(f, "w_{}", j),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::mock::F17;

  #[test]
  fn test_multivariate_expressions() {
    // Create public inputs x_0, x_1
    let x0 = Expression::from(Input::Public(0));
    let x1 = Expression::from(Input::Public(1));

    // Create witness values w_0, w_1
    let w0 = Expression::from(Input::Witness(0));
    let w1 = Expression::from(Input::Witness(1));

    // Create a constant
    let c = Expression::from(F17::from(5));

    // Test complex expression: 5 * x_0 * w_0 + x_1 * w_1
    let expr = (c * x0 * w0) + (x1 * w1);
    assert_eq!(expr.to_string(), "((5 * x_0 * w_0) + (x_1 * w_1))");
  }
}

//! Defines arithmetic expressions used in circuit construction.
//!
//! This module provides types for building and manipulating arithmetic expressions
//! over a field, supporting operations like addition, multiplication, and negation.

use super::*;

/// Variables used in arithmetic expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variable {
  /// Public input variable x_i
  Public(usize),
  /// Witness variable w_i
  Witness(usize),
  /// Auxiliary variable y_i
  Aux(usize),
  /// Output variable o_i
  Output(usize),
}

/// An arithmetic expression over a field F.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Expression<F: Field> {
  /// A single variable
  Variable(Variable),
  /// A constant field element
  Constant(F),
  /// Sum of expressions
  Add(Vec<Expression<F>>),
  /// Product of expressions
  Mul(Vec<Expression<F>>),
}

impl<F: Field> std::ops::Add for Expression<F> {
  type Output = Self;

  /// Implements addition between expressions.
  ///
  /// Flattens nested additions to maintain a canonical form:
  /// - `(a + b) + c` becomes `a + b + c`
  /// - `a + (b + c)` becomes `a + b + c`
  fn add(self, rhs: Self) -> Self::Output {
    match (self, rhs) {
      (Self::Add(mut v1), Self::Add(v2)) => {
        v1.extend(v2);
        Self::Add(v1)
      },
      (Self::Add(mut v), rhs) => {
        v.push(rhs);
        Self::Add(v)
      },
      (lhs, Self::Add(mut v)) => {
        v.insert(0, lhs);
        Self::Add(v)
      },
      (lhs, rhs) => Self::Add(vec![lhs, rhs]),
    }
  }
}

impl<F: Field> std::ops::Mul for Expression<F> {
  type Output = Self;

  /// Implements multiplication between expressions.
  ///
  /// Flattens nested multiplications to maintain a canonical form:
  /// - `(a * b) * c` becomes `a * b * c`
  /// - `a * (b * c)` becomes `a * b * c`
  fn mul(self, rhs: Self) -> Self::Output {
    match (self, rhs) {
      (Self::Mul(mut v1), Self::Mul(v2)) => {
        v1.extend(v2);
        Self::Mul(v1)
      },
      (Self::Mul(mut v), rhs) => {
        v.push(rhs);
        Self::Mul(v)
      },
      (lhs, Self::Mul(mut v)) => {
        v.insert(0, lhs);
        Self::Mul(v)
      },
      (lhs, rhs) => Self::Mul(vec![lhs, rhs]),
    }
  }
}

impl<F: Field> std::ops::Neg for Expression<F> {
  type Output = Self;

  /// Implements negation by multiplying by -1.
  fn neg(self) -> Self::Output {
    // Negation is multiplication by -1
    Self::Mul(vec![Self::Constant(F::from(-1)), self])
  }
}

// Implement subtraction
impl<F: Field> std::ops::Sub for Expression<F> {
  type Output = Self;

  /// Implements subtraction as addition with negation: a - b = a + (-b)
  fn sub(self, rhs: Self) -> Self::Output {
    // a - b is the same as a + (-b)
    self + (-rhs)
  }
}

impl<F: Field + Display> Display for Expression<F> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    match self {
      Self::Variable(var) => write!(f, "{var}"),
      Self::Constant(c) => write!(f, "{c}"),
      Self::Add(terms) => {
        write!(f, "(")?;
        for (i, term) in terms.iter().enumerate() {
          if i > 0 {
            write!(f, " + ")?;
          }
          write!(f, "{term}")?;
        }
        write!(f, ")")
      },
      Self::Mul(factors) => {
        write!(f, "(")?;
        for (i, factor) in factors.iter().enumerate() {
          if i > 0 {
            write!(f, " * ")?;
          }
          write!(f, "{factor}")?;
        }
        write!(f, ")")
      },
    }
  }
}

impl Display for Variable {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    match self {
      Self::Public(i) => write!(f, "x_{i}"),
      Self::Witness(j) => write!(f, "w_{j}"),
      Self::Aux(k) => write!(f, "y_{k}"),
      Self::Output(l) => write!(f, "o_{l}"),
    }
  }
}

use std::fmt::{self, Display, Formatter};

use ark_ff::Field;

// The CircuitBuilder struct remains unchanged from your implementation
#[derive(Clone, Debug)]
pub struct CircuitBuilder<F: Field> {
  pub pub_inputs:  usize,
  pub wit_inputs:  usize,
  pub aux_count:   usize,
  pub expressions: Vec<(Expression<F>, Variable)>,
}

// Variable and Expression enums remain unchanged
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variable {
  Public(usize),
  Witness(usize),
  Aux(usize),
}

#[derive(Clone, Debug)]
pub enum Expression<F: Field> {
  Variable(Variable),
  Constant(F),
  Add(Vec<Expression<F>>),
  Mul(Vec<Expression<F>>),
}

impl<F: Field> CircuitBuilder<F> {
  // All existing methods remain the same
  pub fn new() -> Self {
    Self { pub_inputs: 0, wit_inputs: 0, aux_count: 0, expressions: Vec::new() }
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
    let aux = Variable::Aux(self.aux_count);
    self.aux_count += 1;
    aux
  }

  pub fn add_expression(&mut self, expr: Expression<F>) -> Expression<F> {
    let var = self.new_aux();
    self.expressions.push((expr, var.clone()));
    Expression::Variable(var)
  }

  pub fn expressions(&self) -> &[(Expression<F>, Variable)] { &self.expressions }

  // New methods for expression expansion

  // Helper method to look up the original expression for an auxiliary variable
  fn get_aux_definition(&self, aux_idx: usize) -> Option<&Expression<F>> {
    self.expressions.get(aux_idx).map(|(expr, _)| expr)
  }

  // Main method to expand an expression by resolving all auxiliary variables
  pub fn expand(&self, expr: &Expression<F>) -> Expression<F> {
    match expr {
      // Base cases: constants and non-auxiliary variables remain unchanged
      Expression::Constant(_)
      | Expression::Variable(Variable::Public(_))
      | Expression::Variable(Variable::Witness(_)) => expr.clone(),

      // For auxiliary variables, look up their definition and expand recursively
      Expression::Variable(Variable::Aux(idx)) => {
        if let Some(definition) = self.get_aux_definition(*idx) {
          self.expand(definition)
        } else {
          expr.clone()
        }
      },

      // For operations, expand all their subexpressions recursively
      Expression::Add(terms) =>
        Expression::Add(terms.iter().map(|term| self.expand(term)).collect()),
      Expression::Mul(factors) =>
        Expression::Mul(factors.iter().map(|factor| self.expand(factor)).collect()),
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
    let y0 = builder.add_expression(neg_x0);

    // Test subtraction: w0 - x1
    let sub_expr = w0 - x1;
    let y1 = builder.add_expression(sub_expr);

    // Test complex expression: 3 * (w0 - x1) - (-x0)
    let complex_expr = three * y1 - y0;
    let y2 = builder.add_expression(complex_expr);

    // Print original expressions
    println!("\nOriginal expressions:");
    for (expr, var) in builder.expressions() {
      if let Variable::Aux(idx) = var {
        println!("y_{} := {}", idx, expr);
      }
    }

    // Print expanded expressions
    println!("\nExpanded expressions:");
    for (expr, var) in builder.expressions() {
      if let Variable::Aux(idx) = var {
        println!("y_{} := {}", idx, builder.expand(expr));
      }
    }
  }

  #[test]
  #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
  fn test_expression_expansion() {
    let mut builder = CircuitBuilder::new();

    // Create base values
    let x0 = builder.x(0);
    let x1 = builder.x(1);
    let w0 = builder.w(0);
    let w1 = builder.w(1);
    let five = CircuitBuilder::constant(F17::from(5));

    // Create first expression: 5 * x_0 * w_0
    let expr1 = five * x0 * w0;
    let y0 = builder.add_expression(expr1);

    // Create second expression: x_1 * w_1
    let expr2 = x1 * w1;
    let y1 = builder.add_expression(expr2);

    // Create final expression using intermediate results: y_0 + y_1
    let expr3 = y0 + y1;
    let y2 = builder.add_expression(expr3);

    // Print original and expanded forms
    println!("\nOriginal expressions:");
    for (expr, var) in builder.expressions() {
      if let Variable::Aux(idx) = var {
        println!("y_{} := {}", idx, expr);
      }
    }

    println!("\nExpanded expressions:");
    for (expr, var) in builder.expressions() {
      if let Variable::Aux(idx) = var {
        println!("y_{} := {}", idx, builder.expand(expr));
      }
    }
  }
}

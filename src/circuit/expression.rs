use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variable {
  Public(usize),
  Witness(usize),
  Aux(usize),
  Output(usize),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Expression<F: Field> {
  Variable(Variable),
  Constant(F),
  Add(Vec<Expression<F>>),
  Mul(Vec<Expression<F>>),
}

impl<F: Field> std::ops::Add for Expression<F> {
  type Output = Self;

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

  fn neg(self) -> Self::Output {
    // Negation is multiplication by -1
    Self::Mul(vec![Self::Constant(F::from(-1)), self])
  }
}

// Implement subtraction
impl<F: Field> std::ops::Sub for Expression<F> {
  type Output = Self;

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

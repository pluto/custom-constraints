use super::*;

pub struct Plonkish<F>(PhantomData<F>);
impl<F> CCSType<F> for Plonkish<F> {
  type Selectors = Vec<F>;
}

impl<C: CCSType<F>, F: Field> CCS<C, F> {
  pub fn new_width(width: usize) -> Self {
    todo!()
  }
}

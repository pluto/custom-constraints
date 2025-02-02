#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

use ark_ff::Field;
#[cfg(all(target_arch = "wasm32", test))]
use wasm_bindgen_test::wasm_bindgen_test;
#[cfg(test)] use {mock::F17, rstest::rstest};

pub mod ccs;
pub mod circuit;
pub mod matrix;
pub mod optimizer;

#[cfg(test)]
mod mock {
  use ark_ff::{Fp, MontBackend, MontConfig};

  #[derive(MontConfig)]
  #[modulus = "17"]
  #[generator = "3"]
  pub struct F17Config;
  pub type F17 = Fp<MontBackend<F17Config, 1>, 1>;
}

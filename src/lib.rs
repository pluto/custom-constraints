#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

#[cfg(all(target_arch = "wasm32", test))]
use wasm_bindgen_test::wasm_bindgen_test;
#[cfg(test)]
use {mock::F17, rstest::rstest};

use ark_ff::Field;

pub mod ccs;
pub mod matrix;

#[cfg(test)]
mod mock {
  use ark_ff::{Fp, MontBackend, MontConfig};

  #[derive(MontConfig)]
  #[modulus = "17"]
  #[generator = "3"]
  pub struct F17Config;
  pub type F17 = Fp<MontBackend<F17Config, 1>, 1>;
}

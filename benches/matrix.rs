#![feature(test)]
extern crate test;

use ark_ff::{Fp256, MontBackend, MontConfig};
use custom_constraints::matrix::SparseMatrix;
use test::Bencher;

// Define a large prime field for testing
#[allow(unexpected_cfgs)]
#[derive(MontConfig)]
#[modulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
#[generator = "7"]
pub struct FqConfig;
pub type Fq = Fp256<MontBackend<FqConfig, 4>>;

// Helper function to create a random field element
fn random_field_element() -> Fq {
  // Create a random field element using ark_ff's random generation
  use ark_ff::UniformRand;
  let mut rng = ark_std::test_rng();
  Fq::rand(&mut rng)
}

// Helper to create a sparse matrix with given density
fn create_sparse_matrix(rows: usize, cols: usize, density: f64) -> SparseMatrix<Fq> {
  let mut row_offsets = vec![0];
  let mut col_indices = Vec::new();
  let mut values = Vec::new();
  let mut current_offset = 0;

  for _ in 0..rows {
    for j in 0..cols {
      if rand::random::<f64>() < density {
        col_indices.push(j);
        values.push(random_field_element());
        current_offset += 1;
      }
    }
    row_offsets.push(current_offset);
  }

  SparseMatrix::new(row_offsets, col_indices, values, cols)
}

const COLS: usize = 100;
const SMALL: usize = 2_usize.pow(10);
const MEDIUM: usize = 2_usize.pow(15);
const LARGE: usize = 2_usize.pow(20);

// Matrix-vector multiplication benchmarks
#[bench]
fn bench_sparse_matrix_vec_mul_small(b: &mut Bencher) {
  let matrix = create_sparse_matrix(SMALL, COLS, 0.1);
  let vector: Vec<Fq> = (0..COLS).map(|_| random_field_element()).collect();

  b.iter(|| &matrix * &vector);
}

#[bench]
fn bench_sparse_matrix_vec_mul_medium(b: &mut Bencher) {
  let matrix = create_sparse_matrix(MEDIUM, COLS, 0.01);
  let vector: Vec<Fq> = (0..COLS).map(|_| random_field_element()).collect();

  b.iter(|| &matrix * &vector);
}

#[bench]
fn bench_sparse_matrix_vec_mul_large(b: &mut Bencher) {
  let matrix = create_sparse_matrix(LARGE, LARGE, 0.01);
  let vector: Vec<Fq> = (0..LARGE).map(|_| random_field_element()).collect();

  b.iter(|| &matrix * &vector);
}

#[bench]
fn bench_sparse_matrix_hadamard_small(b: &mut Bencher) {
  let matrix1 = create_sparse_matrix(SMALL, COLS, 0.1);
  let matrix2 = create_sparse_matrix(SMALL, COLS, 0.1);

  b.iter(|| &matrix1 * &matrix2);
}

#[bench]
fn bench_sparse_matrix_hadamard_medium(b: &mut Bencher) {
  let matrix1 = create_sparse_matrix(MEDIUM, COLS, 0.1);
  let matrix2 = create_sparse_matrix(MEDIUM, COLS, 0.1);

  b.iter(|| &matrix1 * &matrix2);
}

#[bench]
fn bench_sparse_matrix_hadamard_large(b: &mut Bencher) {
  let matrix1 = create_sparse_matrix(LARGE, COLS, 0.1);
  let matrix2 = create_sparse_matrix(LARGE, COLS, 0.1);

  b.iter(|| &matrix1 * &matrix2);
}

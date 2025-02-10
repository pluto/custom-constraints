// TODO: all these tests really need to check things more strictly
use super::*;
use crate::mock::F17;

#[test]
fn test_compute_degree_base_cases() {
  // Constants and variables should have degree 1
  let constant = Expression::Constant(F17::from(5));
  assert_eq!(compute_degree(&constant), 1, "Constants should have degree 1");

  let public = Expression::<F17>::Variable(Variable::Public(0));
  assert_eq!(compute_degree(&public), 1, "Public variables should have degree 1");

  let witness = Expression::<F17>::Variable(Variable::Witness(0));
  assert_eq!(compute_degree(&witness), 1, "Witness variables should have degree 1");

  let aux = Expression::<F17>::Variable(Variable::Aux(0));
  assert_eq!(compute_degree(&aux), 1, "Auxiliary variables should have degree 1");
}

#[test]
fn test_compute_degree_addition() {
  // Addition should take the maximum degree of its terms
  let x = Expression::<F17>::Variable(Variable::Public(0));
  let y = Expression::Variable(Variable::Witness(0));

  // Simple addition: x + y (degree 1)
  let simple_add = Expression::Add(vec![x.clone(), y.clone()]);
  assert_eq!(compute_degree(&simple_add), 1, "x + y should have degree 1");

  // x + (x * y) (degree 2)
  let mul = Expression::Mul(vec![x.clone(), y.clone()]);
  let mixed_add = Expression::Add(vec![x.clone(), mul.clone()]);
  assert_eq!(compute_degree(&mixed_add), 2, "x + (x * y) should have degree 2");

  // x + (x * y) + (x * y * y) (degree 3)
  let triple_mul = Expression::Mul(vec![x.clone(), y.clone(), y.clone()]);
  let complex_add = Expression::Add(vec![x, mul, triple_mul]);
  assert_eq!(compute_degree(&complex_add), 3, "x + (x * y) + (x * y * y) should have degree 3");
}

#[test]
fn test_compute_degree_multiplication() {
  // Multiplication should sum the degrees of its factors
  let x = Expression::<F17>::Variable(Variable::Public(0));
  let y = Expression::Variable(Variable::Witness(0));

  // Simple multiplication: x * y (degree 2)
  let simple_mul = Expression::Mul(vec![x.clone(), y.clone()]);
  assert_eq!(compute_degree(&simple_mul), 2, "x * y should have degree 2");

  // x * y * y (degree 3)
  let triple_mul = Expression::Mul(vec![x.clone(), y.clone(), y.clone()]);
  assert_eq!(compute_degree(&triple_mul), 3, "x * y * y should have degree 3");

  // (x * y) * (y * y) (degree 4)
  let double_mul = Expression::Mul(vec![y.clone(), y.clone()]);
  let nested_mul = Expression::Mul(vec![simple_mul, double_mul]);
  assert_eq!(compute_degree(&nested_mul), 4, "(x * y) * (y * y) should have degree 4");
}

#[test]
fn test_compute_degree_complex_expressions() {
  let x = Expression::<F17>::Variable(Variable::Public(0));
  let y = Expression::Variable(Variable::Witness(0));

  // Build (x * y * y) + (x + y)
  let triple_mul = Expression::Mul(vec![x.clone(), y.clone(), y.clone()]);
  let simple_add = Expression::Add(vec![x.clone(), y.clone()]);
  let complex = Expression::Add(vec![triple_mul, simple_add]);
  assert_eq!(compute_degree(&complex), 3, "(x * y * y) + (x + y) should have degree 3");

  // Build ((x * y) + y) * (x * x)
  let mul_add = Expression::Add(vec![Expression::Mul(vec![x.clone(), y.clone()]), y.clone()]);
  let square = Expression::Mul(vec![x.clone(), x.clone()]);
  let complex_mul = Expression::Mul(vec![mul_add, square]);
  assert_eq!(compute_degree(&complex_mul), 4, "((x * y) + y) * (x * x) should have degree 4");
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn test_expression_arithmetic() {
  let mut builder = Circuit::new();

  // Create base values
  let x0 = builder.x(0);
  let x1 = builder.x(1);
  let w0 = builder.w(0);
  let three = Circuit::constant(F17::from(3));

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
  let mut builder = Circuit::<_, F17>::new();

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
  let mut builder = Circuit::<_, F17>::new();

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
  let mut builder = Circuit::<_, F17>::new();

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
  let mut builder = Circuit::<_, F17>::new();

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
  let mut builder = Circuit::<_, F17>::new();

  // Create inputs and build expressions (same as before)
  let x = builder.x(0);
  let y = builder.w(0);

  let x_cubed = x.clone() * x.clone() * x.clone();
  let y_squared = y.clone() * y.clone();
  let common_term = x_cubed.clone() + y_squared.clone();

  // Build P1: (x^3 + y^2)^2 * (x + y)
  let common_term_squared = common_term.clone() * common_term.clone();
  let x_plus_y = x.clone() + y.clone();
  let p1 = common_term_squared.clone() * x_plus_y;

  // Build P2: x * y^4 + (x^3 + y^2)^3
  let y_fourth = y_squared.clone() * y_squared.clone();
  let term1 = x.clone() * y_fourth;
  let common_term_cubed = common_term * common_term_squared;
  let p2 = term1 + common_term_cubed;

  // Print original expressions before reduction
  println!("\nOriginal P1: {}", p1);
  println!("Original P2: {}", p2);

  // Mark outputs
  builder.mark_output(p1);
  builder.mark_output(p2);

  println!("\nOriginal circuit state:");
  for (expr, var) in builder.expressions() {
    match var {
      Variable::Aux(idx) => println!("y_{} := {}", idx, expr),
      Variable::Output(idx) => println!("o_{} := {}", idx, expr),
      _ => println!("{} := {}", var, expr),
    }
  }

  // Now fix the degree
  let deg_3_circuit = builder.fix_degree::<3>();

  // Verify degrees after fixing
  println!("\nDegree-constrained expressions:");
  for (expr, var) in deg_3_circuit.expressions() {
    let degree = compute_degree(expr);
    println!("{} := {} (degree {})", var, expr, degree);
    assert!(degree <= 3, "Expression {} exceeds degree bound", var);
  }

  let optimized_circuit = deg_3_circuit.optimize();
  let ccs = optimized_circuit.into_ccs();
  println!("\nFinal CCS:\n{}", ccs);
}

// TODO: This test can show that if we run the optimzer, we may unjustifiably kill off constraints.
// So we need to rethink what optimization means in this case.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn test_raw_low_degree_constraint_not_removed() {
  let mut builder = Circuit::<_, F17>::new();

  // Create inputs and build expressions (same as before)
  let x = builder.x(0);
  let y = builder.w(0);

  // Enforce x is a bool
  let bool = x.clone() * (Expression::Constant(F17::from(1)) - x.clone());
  let toggle = x * y + Expression::Constant(F17::ONE);

  builder.add_internal(bool);
  builder.add_internal(toggle);

  let fixed = builder.fix_degree::<3>();
  // Verify degrees after fixing

  println!("\nDegree-constrained expressions:");
  for (expr, var) in fixed.expressions() {
    let degree = compute_degree(expr);
    println!("{} := {} (degree {})", var, expr, degree);
    assert!(degree <= 3, "Expression {} exceeds degree bound", var);
  }

  let ccs = fixed.into_ccs();

  println!("CCS: {ccs}");
}

// TODO: all these tests really need to check things more strictly
use super::*;
use crate::mock::F17;

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
  let mut builder = Circuit::<F17>::new();

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
  let mut builder = Circuit::<F17>::new();

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
  let mut builder = Circuit::<F17>::new();

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
  let mut builder = Circuit::<F17>::new();

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

// TODO: looking at this more closely, I don't believe the aux `y_3` needs to exist. How we can
// check this is to see if that variable is ever used elsewhere. So we need some optimizer step
// that depends on degrees. In other words, it's like row reduction, but we can remove whole rows
// from every matrix. Also, y1 should not exist. There should be no aux variables that are degree
// < 3 ever as it means we could have packed them better.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn test_complex_degree_reduction() {
  let mut builder = Circuit::<F17>::new();

  // Let's evaluate two related polynomials:
  // P1(x,y) = (x^3 + y^2)^2 * (x + y)
  // P2(x,y) = x * y^4 + (x^3 + y^2)^3

  // Create our inputs
  let x = builder.x(0); // Public input x
  let y = builder.w(0); // Witness input y

  // First, let's build some common subexpressions
  // x^3 = x * x * x
  let x_cubed = x.clone() * x.clone() * x.clone();

  // y^2 = y * y
  let y_squared = y.clone() * y.clone();

  // (x^3 + y^2) - this is used in both polynomials
  let common_term = x_cubed.clone() + y_squared.clone();

  // Now build P1(x,y) = (x^3 + y^2)^2 * (x + y)
  let common_term_squared = common_term.clone() * common_term.clone(); // degree 4
  let x_plus_y = x.clone() + y.clone(); // degree 1
  let p1 = common_term_squared.clone() * x_plus_y; // Total degree: 5

  // Now build P2(x,y) = x * y^4 + (x^3 + y^2)^3
  let y_fourth = y_squared.clone() * y_squared.clone(); // degree 4
  let term1 = x.clone() * y_fourth; // degree 5
  let common_term_cubed = common_term * common_term_squared; // degree 6
  let p2 = term1 + common_term_cubed; // Max degree: 6

  // Let's reduce these to degree 3 expressions
  println!("\nReducing expressions to degree 3:");
  let p1_reduced = builder.reduce_degree(p1.clone(), 3);
  let p2_reduced = builder.reduce_degree(p2.clone(), 3);

  // Mark both as outputs
  builder.mark_output(p1_reduced);
  builder.mark_output(p2_reduced);

  // Print the full computation graph
  println!("\nOriginal P1: {}", p1);
  println!("Original P2: {}", p2);
  println!("\nAuxiliary and output variables:");
  for (expr, var) in builder.expressions() {
    match var {
      Variable::Aux(idx) => println!("y_{} := {}", idx, expr),
      Variable::Output(idx) => println!("o_{} := {}", idx, expr),
      _ => println!("{} := {}", var, expr),
    }
  }

  // Verify degrees of all expressions
  println!("\nVerifying degrees of all expressions:");
  for (expr, var) in builder.expressions() {
    let degree = compute_degree(expr);
    println!("{} has degree {}", var, degree);
    assert!(degree <= 3, "Expression {} exceeds degree bound", var);
  }

  let ccs = builder.into_ccs(3);
  println!("{ccs}");
}

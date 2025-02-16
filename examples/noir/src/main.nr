pub fn main(x0: pub Field, w: [Field; 2]) -> pub Field {
    let val0 = x0 * w[0] + w[1] + 1;
    let val1 = (x0 + 3) * w[1] + w[0];
    val0 + val1
}

// 1*(x0 * w0) + 1*(x0 * w1) + 1*w0 + 4*w1 + 1 - OUT == 0

// -4 (x0 * w0) + -4(x0 * w1) -4w0 - 16w1 - 4 - OUT == 0

// OUT - (x0 * w0) - (x0 * w1) - w0 -4w1 - 1 == 0
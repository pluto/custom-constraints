pub fn main(x0: pub Field, w: [Field; 2]) -> pub Field {
    1 * x0 * x0
        + 2 * x0 * w[0]
        + 3 * x0 * w[1]
        + 4 * w[0] * w[0]
        + 5 * w[0] * w[1]
        + 6 * w[1] * w[1]
        + 7 * x0
        + 8 * w[0]
        + 9 * w[1]
        + 10
}

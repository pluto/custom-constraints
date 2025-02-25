pub fn main(witnesses: [Field; 10]) -> pub Field {
    let mut expr = 0;
    for w in witnesses {
        expr += w;
        expr *= w;
    }
    expr
}

use std::field::bn254::lt;
// use std::hash::sha256;
use std::hash::poseidon::bn254::hash_2;

pub fn main(salt: pub [Field; 32], private: [Field; 16]) -> pub Field {
    assert(salt.all(|x| lt(x, 255)));
    // This uses a memory op!
    // let vec: [Field; 48] = salt.as_slice().append(private).as_array();
    let mut hash = 0;
    for x in salt {
        hash = hash_2([hash, x]);
    }
    for x in private {
        hash = hash_2([hash, x]);
    }
    hash
}

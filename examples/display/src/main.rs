use tensor::prelude::*;
use tensor_cpu::CpuBackend;

type B = CpuBackend;

fn main() {
    let tensor = Tensor::<B, f32>::full(0.0, [2, 2]);
    println!("{}", tensor)
}

use tensor::prelude::*;
use tensor_cpu::CpuBackend;

type B = CpuBackend;

fn print<B: Backend + FlatIter, T: SupportedDType<B>>(tensor: Tensor<B, T>) {
    println!("{}", tensor.display_content())
}

fn main() {
    let device = Device::<B>::default();
    let tensor = Tensor::full(0.0, [2, 2], &device);
    print(tensor)
}

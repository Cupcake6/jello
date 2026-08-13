use tensor::prelude::*;
use tensor_cpu::CpuBackend;

type B = CpuBackend;

fn print<B: Backend, T: SupportedDType<B>>(tensor: Tensor<B, T>)
where
    B: tensor_features::FlatIter,
{
    println!("{}", tensor.display_content())
}

fn main() {
    let device = Device::<B>::default();
    let tensor = Tensor::full(0.0, [2, 2], &device);
    print(tensor)
}

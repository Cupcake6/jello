use tensor_core::{backend::Backend, dtype::SupportedDType};

mod dtype;
mod tensor;

pub struct Cpu;

impl Backend for Cpu {
    type DefaultDType = f32;
    type TensorPrimitive<T: SupportedDType<Self>> = tensor::CpuTensor<T>;
}

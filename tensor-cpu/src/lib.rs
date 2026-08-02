use tensor_core::{backend::Backend, dtype::SupportedDType};

mod dtype;
mod tensor;

pub struct CpuBackend;

impl Backend for CpuBackend {
    const NAME: &'static str = "cpu";
    type DefaultDType = f32;
    type TensorPrimitive<T: SupportedDType<Self>> = tensor::CpuTensor<T>;
}

use tensor_core::{backend::Backend, dtype::SupportedDType};

mod dtype;
mod tensor;
mod tensor_features;

mod device {
    pub struct CpuDevice;
}

pub struct CpuBackend;

impl Backend for CpuBackend {
    const NAME: &'static str = "cpu";
    type DefaultDType = f32;
    type TensorPrimitive<T: SupportedDType<Self>> = tensor::CpuTensor<T>;
    type Device = device::CpuDevice;
}

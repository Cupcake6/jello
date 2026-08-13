use tensor_core::{backend::Backend, dtype::SupportedDType};

mod device;
mod dtype;
mod tensor;
mod tensor_features;

pub struct CpuBackend;

impl Backend for CpuBackend {
    const NAME: &'static str = "cpu";
    type DefaultDType = f32;

    type TensorImpl<T: SupportedDType<Self>> = tensor::CpuTensorImpl<T>;
    type DeviceImpl = device::CpuDeviceImpl;
}

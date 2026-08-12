use crate::{dtype::SupportedDType, tensor::TensorOps};

pub mod tensor_features;

pub trait Backend: Sized {
    const NAME: &'static str;
    type DefaultDType: SupportedDType<Self>;
    type TensorPrimitive<T: SupportedDType<Self>>: TensorOps<T>;
    type DeviceInner;
}

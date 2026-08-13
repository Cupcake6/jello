use crate::{dtype::SupportedDType, tensor::TensorOps};

pub mod tensor_features;

pub trait Backend: Sized {
    const NAME: &'static str;
    type DefaultDType: SupportedDType<Self>;

    type TensorImpl<T: SupportedDType<Self>>: TensorOps<Self, T>;
    type DeviceImpl;
}

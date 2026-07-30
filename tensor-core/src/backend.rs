use crate::{dtype::SupportedDType, tensor_ops::TensorOps};

pub trait Backend: Sized {
    type DefaultDType: SupportedDType<Self>;
    type TensorPrimitive<T: SupportedDType<Self>>: TensorOps<T>;
}

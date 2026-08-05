use crate::{
    dtype::SupportedDType,
    tensor::{TensorDisplay, TensorOps},
};

pub mod tensor_features;

pub trait Backend: Sized {
    const NAME: &'static str;
    type DefaultDType: SupportedDType<Self>;
    type TensorPrimitive<T: SupportedDType<Self>>: TensorOps<T> + TensorDisplay<T>;
}

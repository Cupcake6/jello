use crate::{
    dtype::SupportedDType,
    tensor::{TensorDisplay, TensorOps},
};

pub trait Backend: Sized {
    type DefaultDType: SupportedDType<Self>;
    type TensorPrimitive<T: SupportedDType<Self>>: TensorOps<T> + TensorDisplay<T>;
}

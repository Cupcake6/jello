use crate::{dtype::SupportedDType, tensor_metadata::shape::Shape};

pub trait Backend: TensorOps + Sized {}

pub trait TensorOps {
    type Tensor<T: SupportedDType<Self>>;

    fn full<T: SupportedDType<Self>>(fill_value: T, shape: Shape) -> Self::Tensor<T>
    where
        Self: Backend;
}

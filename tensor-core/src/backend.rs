use crate::{dtype::SupportedDType, tensor_metadata::shape::Shape};

pub trait BackendKind: Sized {}

pub trait TensorOps: BackendKind {
    type Tensor<T: SupportedDType<Self>>;

    fn full<T: SupportedDType<Self>>(fill_value: T, shape: Shape) -> Self::Tensor<T>;
    fn num_dims<T: SupportedDType<Self>>(tensor: &Self::Tensor<T>) -> usize;
}

use crate::{
    dtype::SupportedDType,
    tensor_metadata::{shape::Shape, stride::Stride},
};

pub trait BackendKind: Sized {}

pub trait TensorOps: BackendKind {
    type Tensor<T: SupportedDType<Self>>;

    fn full<T: SupportedDType<Self>>(fill_value: T, shape: Shape) -> Self::Tensor<T>;
    fn num_dims<T: SupportedDType<Self>>(tensor: &Self::Tensor<T>) -> usize;
    fn num_items<T: SupportedDType<Self>>(tensor: &Self::Tensor<T>) -> u64;
    fn shape<T: SupportedDType<Self>>(tensor: &Self::Tensor<T>) -> &Shape;
    fn stride<T: SupportedDType<Self>>(tensor: &Self::Tensor<T>) -> &Stride;
}

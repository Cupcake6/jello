use crate::{
    backend::Backend,
    dtype::{DType, SupportedDType},
    tensor_metadata::{shape::Shape, stride::Stride},
};

pub mod error;

pub trait TensorOps<B: Backend, T: SupportedDType<B>>: Sized {
    fn full(fill_value: T, shape: Shape, device: &B::DeviceImpl) -> Self;
    fn from_flat_slice(
        shape: Shape,
        flat_slice: &[T],
        device: &B::DeviceImpl,
    ) -> Result<Self, error::ItemNumberMismatchError>;
    fn num_dims(&self) -> usize;
    fn num_items(&self) -> u64;
    fn shape(&self) -> &Shape;
    fn stride(&self) -> &Stride;
    fn dtype(&self) -> DType;
}

use crate::{
    dtype::DType,
    tensor_metadata::{shape::Shape, stride::Stride},
};

pub mod error;

pub trait TensorOps<T>: Sized {
    fn full(fill_value: T, shape: Shape) -> Self;
    fn from_flat_slice(
        shape: Shape,
        flat_slice: &[T],
    ) -> Result<Self, error::ItemNumberMismatchError>;
    fn num_dims(&self) -> usize;
    fn num_items(&self) -> u64;
    fn shape(&self) -> &Shape;
    fn stride(&self) -> &Stride;
    fn dtype(&self) -> DType;
}

use crate::{
    backend::Backend,
    dtype::{DType, SupportedDType},
    dtype_traits,
    tensor_metadata::shape::Shape,
};

pub mod error;

pub trait TensorOps<B: Backend, T: SupportedDType<B>>: Sized + Clone {
    fn full(fill_value: T, shape: Shape, device: &B::DeviceImpl) -> Self;
    fn from_flat_slice(
        shape: Shape,
        flat_slice: &[T],
        device: &B::DeviceImpl,
    ) -> Result<Self, error::ItemNumberMismatchError>;
    fn num_dims(&self) -> usize;
    fn num_items(&self) -> u64;
    fn shape(&self) -> &Shape;
    fn dtype(&self) -> DType;

    fn zeros(shape: Shape, device: &B::DeviceImpl) -> Self
    where
        T: dtype_traits::Zero,
    {
        Self::full(<T as dtype_traits::Zero>::ZERO, shape, device)
    }

    fn ones(shape: Shape, device: &B::DeviceImpl) -> Self
    where
        T: dtype_traits::One,
    {
        Self::full(<T as dtype_traits::One>::ONE, shape, device)
    }
}

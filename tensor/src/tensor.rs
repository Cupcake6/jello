use crate::{backend::Backend, device::Device};
use tensor_core::{
    backend::tensor_features,
    dtype::{DType, SupportedDType},
    dtype_traits,
    tensor::TensorOps,
    tensor_metadata::{shape::Shape, stride::Stride},
};

pub mod display;
pub mod error;

type TensorImpl<B, T> = <B as Backend>::TensorImpl<T>;

pub struct Tensor<B, T = <B as Backend>::DefaultDType>(TensorImpl<B, T>)
where
    B: Backend,
    T: SupportedDType<B>;

impl<B: Backend, T: SupportedDType<B>> Tensor<B, T> {
    pub fn full<S: Into<Shape>>(fill_value: T, shape: S, device: &Device<B>) -> Self {
        Self(TensorImpl::<B, T>::full(
            fill_value,
            shape.into(),
            &device.0,
        ))
    }

    pub fn from_flat_slice<S: Into<Shape>>(
        shape: S,
        flat_slice: &[T],
        device: &Device<B>,
    ) -> Result<Self, error::ItemNumberMismatchError> {
        Ok(Self(TensorImpl::<B, T>::from_flat_slice(
            shape.into(),
            flat_slice,
            &device.0,
        )?))
    }

    pub fn num_dims(&self) -> usize {
        self.0.num_dims()
    }

    pub fn num_items(&self) -> u64 {
        self.0.num_items()
    }

    pub fn shape(&self) -> &Shape {
        self.0.shape()
    }

    pub fn stride(&self) -> &Stride {
        self.0.stride()
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype()
    }

    pub fn zeros<S: Into<Shape>>(shape: S, device: &Device<B>) -> Self
    where
        T: dtype_traits::Zero,
    {
        Self(TensorImpl::<B, T>::zeros(shape.into(), &device.0))
    }

    pub fn ones<S: Into<Shape>>(shape: S, device: &Device<B>) -> Self
    where
        T: dtype_traits::One,
    {
        Self(TensorImpl::<B, T>::ones(shape.into(), &device.0))
    }
}

impl<B, T: SupportedDType<B>> Tensor<B, T>
where
    B: Backend + tensor_features::FlatIter,
{
    pub fn flat_iter<'a>(&'a self) -> impl Iterator<Item = &'a T> {
        B::flat_iter(&self.0)
    }

    pub fn flat_iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> {
        B::flat_iter_mut(&mut self.0)
    }
}

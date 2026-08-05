use crate::backend::Backend;
use std::fmt;
use tensor_core::{
    backend::tensor_features,
    dtype::{DType, SupportedDType},
    tensor::{TensorDisplay, TensorOps},
    tensor_metadata::{shape::Shape, stride::Stride},
};

type TensorPrimitive<B, T> = <B as Backend>::TensorPrimitive<T>;

pub struct Tensor<B, T>(TensorPrimitive<B, T>)
where
    B: Backend,
    T: SupportedDType<B>;

impl<B: Backend, T: SupportedDType<B>> Tensor<B, T> {
    pub fn full<S: Into<Shape>>(fill_value: T, shape: S) -> Self {
        Self(TensorPrimitive::<B, T>::full(fill_value, shape.into()))
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

impl<B: Backend, T: SupportedDType<B>> fmt::Display for Tensor<B, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.display(f)
    }
}

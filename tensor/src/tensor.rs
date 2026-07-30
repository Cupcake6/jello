use crate::backend::Backend;
use tensor_core::{
    dtype::SupportedDType,
    tensor_metadata::{shape::Shape, stride::Stride},
    tensor_ops::TensorOps,
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
}

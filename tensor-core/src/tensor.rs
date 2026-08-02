use crate::{
    dtype::DType,
    tensor_metadata::{shape::Shape, stride::Stride},
};
use std::fmt;

pub trait TensorOps<T> {
    fn full(fill_value: T, shape: Shape) -> Self;
    fn num_dims(&self) -> usize;
    fn num_items(&self) -> u64;
    fn shape(&self) -> &Shape;
    fn stride(&self) -> &Stride;
    fn dtype(&self) -> DType;
}

pub trait TensorDisplay<T> {
    fn display(&self, f: &mut fmt::Formatter) -> fmt::Result;
}

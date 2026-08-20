use crate::tensor_metadata::{shape::Shape, stride::Stride};

pub mod dimensions;
pub mod shape;
pub mod stride;

#[derive(Clone)]
pub struct TensorMetadata {
    shape: Shape,
    stride: Stride,
    offset: u64,
}

impl TensorMetadata {
    pub fn new(shape: Shape) -> Self {
        let stride = Stride::contiguous(&shape);

        Self {
            shape,
            stride,
            offset: 0,
        }
    }

    pub fn num_dims(&self) -> usize {
        self.shape.num_dims()
    }

    pub fn num_items(&self) -> u64 {
        self.shape.num_items()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

use crate::tensor_metadata::{dimensions::Dimensions, memory_layout::MemoryLayout, shape::Shape};

pub mod dimensions;
pub mod memory_layout;
pub mod shape;

#[derive(Clone)]
pub struct TensorMetadata {
    shape: Shape,
    memory_layout: MemoryLayout,
}

impl TensorMetadata {
    pub fn new(shape: Shape) -> Self {
        let memory_layout = MemoryLayout::make_contiguous(&shape);

        Self {
            shape,
            memory_layout,
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

    pub fn contiguous(&self) -> bool {
        self.memory_layout.contiguous()
    }

    pub fn stride(&self) -> &Dimensions {
        self.memory_layout.stride()
    }
}

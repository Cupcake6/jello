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

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn memory_layout(&self) -> &MemoryLayout {
        &self.memory_layout
    }
}

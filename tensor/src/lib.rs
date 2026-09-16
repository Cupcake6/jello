use memory_layout::MemoryLayout;
use shape::Shape;
use std::sync::Arc;

mod memory_layout;
mod shape;

pub trait TensorItem: Clone + Copy {}
impl<T: Clone + Copy> TensorItem for T {}

pub struct Tensor<const N: usize, T: TensorItem> {
    data: Arc<[T]>,
    shape: Shape<N>,
    memory_layout: MemoryLayout<N>,
}

impl<const N: usize, T: TensorItem> Tensor<N, T> {
    #[inline]
    pub fn num_items(&self) -> u64 {
        self.shape.num_items()
    }

    #[inline]
    pub fn num_dims(&self) -> usize {
        N
    }

    pub fn full(value: T, shape: impl Into<Shape<N>>) -> Self {
        let shape = shape.into();
        let data = Arc::from(vec![value; shape.num_items() as usize]);

        Self {
            memory_layout: MemoryLayout::make_contiguous(shape),
            shape: shape,
            data,
        }
    }
}

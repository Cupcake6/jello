use crate::tensor_metadata::{dimensions::Dimensions, shape::Shape};

#[derive(Clone)]
pub struct MemoryLayout {
    stride: Dimensions,
    contiguous: bool,
    offset: u64,
}

impl MemoryLayout {
    pub fn stride(&self) -> &Dimensions {
        &self.stride
    }

    pub fn contiguous(&self) -> bool {
        self.contiguous
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn make_contiguous(shape: &Shape) -> Self {
        let mut stride = Dimensions::zeros(shape.num_dims());
        let mut suffix_product = 1;

        for i in (0..shape.num_dims()).rev() {
            stride[i] = suffix_product;
            suffix_product *= shape[i];
        }

        Self {
            stride,
            contiguous: true,
            offset: 0,
        }
    }
}

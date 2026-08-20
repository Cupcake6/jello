use crate::tensor_metadata::{dimensions::Dimensions, shape::Shape};
use std::ops::Deref;

#[derive(Clone)]
pub struct Stride(Dimensions);

impl Stride {
    pub fn dimensions(&self) -> &Dimensions {
        &self.0
    }

    pub fn contiguous(shape: &Shape) -> Self {
        let mut stride = Dimensions::zeros(shape.num_dims());
        let mut suffix_product = 1;

        for i in (0..shape.num_dims()).rev() {
            stride[i] = suffix_product;
            suffix_product *= shape[i];
        }

        Self(stride)
    }
}

impl Deref for Stride {
    type Target = Dimensions;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Into<Dimensions>> From<T> for Stride {
    fn from(value: T) -> Self {
        Stride(value.into())
    }
}

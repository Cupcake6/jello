use crate::shape::Shape;

pub struct MemoryLayout<const N: usize> {
    stride: [u64; N],
    offset: u64,
}

impl<const N: usize> MemoryLayout<N> {
    pub fn make_contiguous(shape: Shape<N>) -> Self {
        let mut stride = [0; N];
        let mut suffix_product = 1;

        for i in (0..shape.num_dims()).rev() {
            stride[i] = suffix_product;
            suffix_product *= shape[i];
        }

        Self { stride, offset: 0 }
    }
}

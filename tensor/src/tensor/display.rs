use crate::tensor::Tensor;
use std::fmt;
use tensor_core::{
    backend::{Backend, tensor_features::FlatIter},
    dtype::SupportedDType,
    tensor_metadata::shape::Shape,
};

impl<B: Backend, T: SupportedDType<B>> fmt::Display for Tensor<B, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(backend={}, dtype={}, shape={})",
            B::NAME,
            self.dtype().name(),
            self.shape().dimensions()
        )
    }
}

pub struct DisplayContent<'a, B, T>(&'a Tensor<B, T>)
where
    B: Backend + FlatIter,
    T: SupportedDType<B>;

impl<B, T> Tensor<B, T>
where
    B: Backend,
    T: SupportedDType<B>,
{
    pub fn display_content<'a>(&'a self) -> DisplayContent<'a, B, T>
    where
        B: FlatIter,
    {
        DisplayContent(self)
    }
}

fn bracket_count(index: u64, shape: &Shape) -> u64 {
    let mut bracket_count = 0;
    let mut suffix_product = 1;

    for i in (0..shape.len()).rev() {
        suffix_product *= shape[i];
        if index % suffix_product == 0 {
            bracket_count += 1;
        } else {
            return bracket_count;
        }
    }

    bracket_count
}

impl<'a, B, T> fmt::Display for DisplayContent<'a, B, T>
where
    B: Backend + FlatIter,
    T: SupportedDType<B>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for _ in 0..self.0.num_dims() {
            write!(f, "[")?;
        }

        let mut flat_iter = self.0.flat_iter();

        for i in 1..self.0.num_items() {
            write!(f, "{:.2}", flat_iter.next().unwrap())?;
            let bracket_count = bracket_count(i, self.0.shape());

            for _ in 0..bracket_count {
                write!(f, "]")?;
            }

            write!(f, ", ")?;

            for _ in 0..bracket_count {
                write!(f, "[")?;
            }
        }

        write!(f, "{}", flat_iter.next().unwrap())?;

        for _ in 0..self.0.num_dims() {
            write!(f, "]")?;
        }

        Ok(())
    }
}

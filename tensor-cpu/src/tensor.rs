use crate::CpuBackend as B;
use smallvec::{SmallVec, smallvec};
use tensor_core::{
    dtype::DType,
    dtype::SupportedDType,
    tensor::TensorOps,
    tensor_metadata::{TensorMetadata, shape::Shape, stride::Stride},
};

mod display;

pub struct CpuTensor<T: SupportedDType<B>> {
    data: SmallVec<[T; 1]>,
    metadata: TensorMetadata,
}

impl<T: SupportedDType<B>> TensorOps<T> for CpuTensor<T> {
    fn full(fill_value: T, shape: Shape) -> Self {
        let metadata = TensorMetadata::new(shape);
        let data = smallvec![fill_value; metadata.num_items() as usize];

        Self { data, metadata }
    }

    fn num_dims(&self) -> usize {
        self.metadata.num_dims()
    }

    fn num_items(&self) -> u64 {
        self.metadata.num_items()
    }

    fn shape(&self) -> &Shape {
        self.metadata.shape()
    }

    fn stride(&self) -> &Stride {
        self.metadata.stride()
    }

    fn dtype(&self) -> DType {
        <T as SupportedDType<B>>::DTYPE
    }
}

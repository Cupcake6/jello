use crate::CpuBackend as B;
use smallvec::{SmallVec, smallvec};
use tensor_core::{
    dtype::{DType, SupportedDType},
    tensor::{TensorOps, error},
    tensor_metadata::{TensorMetadata, shape::Shape, stride::Stride},
};

pub struct CpuTensor<T: SupportedDType<B>> {
    pub(crate) data: SmallVec<[T; 1]>,
    pub(crate) metadata: TensorMetadata,
}

impl<T: SupportedDType<B>> CpuTensor<T> {
    const fn dtype(&self) -> DType {
        <T as SupportedDType<B>>::DTYPE
    }
}

impl<T: SupportedDType<B>> TensorOps<B, T> for CpuTensor<T> {
    fn full(fill_value: T, shape: Shape) -> Self {
        let metadata = TensorMetadata::new(shape);
        let data = smallvec![fill_value; metadata.num_items() as usize];

        Self { data, metadata }
    }

    fn from_flat_slice(
        shape: Shape,
        flat_slice: &[T],
    ) -> Result<Self, error::ItemNumberMismatchError> {
        let metadata = TensorMetadata::new(shape);

        let expected = metadata.num_items();
        let provided = flat_slice.len() as u64;
        if expected != provided {
            return Err(error::ItemNumberMismatchError(expected, provided));
        }

        Ok(Self {
            data: SmallVec::from_slice(flat_slice),
            metadata,
        })
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
        CpuTensor::dtype(&self)
    }
}

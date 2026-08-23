use crate::CpuBackend as B;
use crate::device::CpuDeviceImpl as DeviceImpl;
use smallvec::{SmallVec, smallvec};
use std::sync::Arc;
use tensor_core::{
    dtype::{DType, SupportedDType},
    tensor::{TensorOps, error},
    tensor_metadata::{TensorMetadata, shape::Shape},
};

#[derive(Clone)]
pub struct CpuTensorImpl<T: SupportedDType<B>> {
    pub(crate) data: Arc<SmallVec<[T; 1]>>,
    pub(crate) metadata: TensorMetadata,
}

impl<T: SupportedDType<B>> TensorOps<B, T> for CpuTensorImpl<T> {
    fn full(fill_value: T, shape: Shape, _device: &DeviceImpl) -> Self {
        let metadata = TensorMetadata::new(shape);
        let data = Arc::new(smallvec![fill_value; metadata.shape().num_items() as usize]);

        Self { data, metadata }
    }

    fn from_flat_slice(
        shape: Shape,
        flat_slice: &[T],
        _device: &DeviceImpl,
    ) -> Result<Self, error::ItemNumberMismatchError> {
        let metadata = TensorMetadata::new(shape);

        let expected = metadata.shape().num_items();
        let provided = flat_slice.len() as u64;
        if expected != provided {
            return Err(error::ItemNumberMismatchError(expected, provided));
        }

        Ok(Self {
            data: Arc::new(SmallVec::from_slice(flat_slice)),
            metadata,
        })
    }

    fn num_dims(&self) -> usize {
        self.metadata.shape().num_dims()
    }

    fn num_items(&self) -> u64 {
        self.metadata.shape().num_items()
    }

    fn shape(&self) -> &Shape {
        self.metadata.shape()
    }

    fn dtype(&self) -> DType {
        <T as SupportedDType<B>>::DTYPE
    }
}

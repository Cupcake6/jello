use crate::backend::Backend;

pub struct Device<B: Backend>(pub(crate) B::DeviceImpl);

impl<B: Backend> Default for Device<B>
where
    B::DeviceImpl: Default,
{
    fn default() -> Self {
        Self(B::DeviceImpl::default())
    }
}

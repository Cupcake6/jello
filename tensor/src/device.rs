use crate::backend::Backend;

pub struct Device<B: Backend>(pub(crate) B::Device);

impl<B: Backend> Default for Device<B>
where
    B::Device: Default,
{
    fn default() -> Self {
        Self(B::Device::default())
    }
}

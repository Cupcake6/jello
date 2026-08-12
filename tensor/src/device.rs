use crate::backend::Backend;

pub struct Device<B: Backend>(pub(crate) B::Device);

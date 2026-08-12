use crate::backend::Backend;

pub struct Device<B: Backend>(B::Device);

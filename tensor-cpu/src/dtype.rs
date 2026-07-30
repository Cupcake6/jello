use crate::CpuBackend as B;
use tensor_core::dtype::SupportedDType;

impl SupportedDType<B> for f32 {}
impl SupportedDType<B> for u32 {}
impl SupportedDType<B> for i32 {}
impl SupportedDType<B> for bool {}

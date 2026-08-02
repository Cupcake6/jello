use crate::backend::Backend;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DType {
    F32,
    U32,
    I32,
    Bool,
}

impl DType {
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::U32 => "U32",
            Self::I32 => "I32",
            Self::Bool => "Bool",
        }
    }
}

mod sealed {
    use super::DType;

    pub trait Primitive: Copy + Send + Sync + 'static {
        const DTYPE: DType;
    }

    impl Primitive for f32 {
        const DTYPE: DType = DType::F32;
    }

    impl Primitive for u32 {
        const DTYPE: DType = DType::U32;
    }

    impl Primitive for i32 {
        const DTYPE: DType = DType::I32;
    }

    impl Primitive for bool {
        const DTYPE: DType = DType::Bool;
    }
}

pub trait SupportedDType<B: Backend>: sealed::Primitive {
    const DTYPE: DType = <Self as sealed::Primitive>::DTYPE;
}

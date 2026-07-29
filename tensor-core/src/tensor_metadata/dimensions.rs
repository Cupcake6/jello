use smallvec::{SmallVec, smallvec};
use std::ops::{Deref, DerefMut};

pub struct Dimensions(SmallVec<[u64; Self::NUM_INLINE_DIMS]>);

impl Dimensions {
    const NUM_INLINE_DIMS: usize = 5;

    pub fn num_dims(&self) -> usize {
        self.0.len() as usize
    }

    pub fn as_slice(&self) -> &[u64] {
        self.0.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        self.0.as_mut_slice()
    }

    pub fn zeros(num_dims: usize) -> Self {
        Self(smallvec![0; num_dims as usize])
    }
}

impl<const N: usize> From<[u64; N]> for Dimensions {
    fn from(value: [u64; N]) -> Self {
        Self(SmallVec::from_slice(&value))
    }
}

impl From<&[u64]> for Dimensions {
    fn from(value: &[u64]) -> Self {
        Self(SmallVec::from_slice(value))
    }
}

impl Deref for Dimensions {
    type Target = [u64];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl DerefMut for Dimensions {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}

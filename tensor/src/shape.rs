use std::ops::Deref;

#[derive(Clone, Copy)]
pub struct Shape<const N: usize> {
    dimensions: [u64; N],
    num_items: u64,
}

impl<const N: usize> Shape<N> {
    pub fn dimensions(&self) -> &[u64; N] {
        &self.dimensions
    }

    pub fn num_items(&self) -> u64 {
        self.num_items
    }

    pub fn num_dims(&self) -> usize {
        N
    }
}

impl<const N: usize, T> From<T> for Shape<N>
where
    T: Into<[u64; N]>,
{
    fn from(value: T) -> Self {
        let dimensions = value.into();

        Self {
            num_items: dimensions.iter().product(),
            dimensions,
        }
    }
}

impl<const N: usize> Deref for Shape<N> {
    type Target = [u64; N];

    fn deref(&self) -> &Self::Target {
        &self.dimensions
    }
}

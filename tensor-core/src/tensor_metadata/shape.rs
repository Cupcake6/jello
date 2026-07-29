use crate::tensor_metadata::dimensions::Dimensions;
use std::ops::Deref;

pub struct Shape {
    dimensions: Dimensions,
    num_items: u64,
}

impl Shape {
    pub fn num_items(&self) -> u64 {
        self.num_items
    }

    pub fn dimensions(&self) -> &Dimensions {
        &self.dimensions
    }
}

impl Deref for Shape {
    type Target = Dimensions;

    fn deref(&self) -> &Self::Target {
        &self.dimensions
    }
}

impl<T: Into<Dimensions>> From<T> for Shape {
    fn from(value: T) -> Self {
        let dimensions = value.into();

        Self {
            num_items: dimensions.iter().product(),
            dimensions,
        }
    }
}

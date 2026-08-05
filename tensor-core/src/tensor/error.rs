#[derive(Debug, thiserror::Error)]
#[error("mismatched number of items: expected {0}, but {1} were provided")]
pub struct ItemNumberMismatchError(pub u64, pub u64);

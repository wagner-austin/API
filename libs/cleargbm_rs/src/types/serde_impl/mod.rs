//! Manual serde implementations for types.
//!
//! These implementations avoid the `?` operator per project rules.
//!
//! Visitor types are `pub(crate)` to allow testing of `expecting()` error paths.

pub(crate) mod histogram_buffer;
mod split_config;
mod tree_node;
mod tree_node_config;

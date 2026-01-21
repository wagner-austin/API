//! Failing deserializer for testing error paths.

use core::fmt::{self, Display};
use serde::de::{self, Visitor};

/// Error type for failing deserializer.
#[derive(Debug)]
pub struct DeError {
    /// Error message.
    pub message: String,
}

impl Display for DeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for DeError {}

impl de::Error for DeError {
    fn custom<T: Display>(msg: T) -> Self {
        DeError {
            message: msg.to_string(),
        }
    }
}

/// What wrong type to provide.
pub enum WrongTypeMode {
    /// Provide an i64 when something else is expected.
    Integer,
    /// Provide a string when something else is expected.
    StringValue,
    /// Provide a map with an integer key to trigger field expecting().
    MapWithIntegerKey,
    /// Provide a map with valid key but wrong value type.
    MapWithWrongValue(&'static str),
    /// Provide a map with duplicate field to trigger duplicate_field error.
    MapWithDuplicateField(&'static str),
    /// Provide a map with duplicate field for struct types (returns valid struct values).
    StructDuplicateField(&'static str),
    /// Provide a map that errors on first next_key call.
    MapErrorOnKey,
    /// Provide a map that succeeds on key but errors on next_value.
    MapErrorOnValue(&'static str),
}

/// Deserializer that provides wrong types to trigger expecting().
pub struct FailingDeserializer {
    mode: WrongTypeMode,
}

impl FailingDeserializer {
    /// Create a deserializer that provides an integer.
    #[must_use]
    pub fn integer() -> Self {
        FailingDeserializer {
            mode: WrongTypeMode::Integer,
        }
    }

    /// Create a deserializer that provides a string.
    #[must_use]
    pub fn string_value() -> Self {
        FailingDeserializer {
            mode: WrongTypeMode::StringValue,
        }
    }

    /// Create a deserializer that provides a map with integer key.
    #[must_use]
    pub fn map_with_integer_key() -> Self {
        FailingDeserializer {
            mode: WrongTypeMode::MapWithIntegerKey,
        }
    }

    /// Create a deserializer that provides valid key but wrong value.
    #[must_use]
    pub fn map_with_wrong_value(field_name: &'static str) -> Self {
        FailingDeserializer {
            mode: WrongTypeMode::MapWithWrongValue(field_name),
        }
    }

    /// Create a deserializer that provides duplicate field.
    #[must_use]
    pub fn map_with_duplicate_field(field_name: &'static str) -> Self {
        FailingDeserializer {
            mode: WrongTypeMode::MapWithDuplicateField(field_name),
        }
    }

    /// Create a deserializer that provides duplicate field for struct types.
    /// This returns valid struct/sequence values so the first field succeeds,
    /// triggering the duplicate check on the second occurrence.
    #[must_use]
    pub fn struct_duplicate_field(field_name: &'static str) -> Self {
        FailingDeserializer {
            mode: WrongTypeMode::StructDuplicateField(field_name),
        }
    }

    /// Create a deserializer that errors on first next_key call.
    /// This tests the `Err(e) => return Err(e)` path for map.next_key().
    #[must_use]
    pub fn map_error_on_key() -> Self {
        FailingDeserializer {
            mode: WrongTypeMode::MapErrorOnKey,
        }
    }

    /// Create a deserializer that succeeds on key but errors on next_value.
    /// This tests the `Err(e) => return Err(e)` path for map.next_value().
    #[must_use]
    pub fn map_error_on_value(field_name: &'static str) -> Self {
        FailingDeserializer {
            mode: WrongTypeMode::MapErrorOnValue(field_name),
        }
    }
}

/// Map access that returns an integer for the key.
pub struct IntegerKeyMapAccess {
    /// Whether we've returned the key.
    pub done: bool,
}

impl<'de> de::MapAccess<'de> for IntegerKeyMapAccess {
    type Error = DeError;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: de::DeserializeSeed<'de>,
    {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        match seed.deserialize(FailingDeserializer::integer()) {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(e),
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        seed.deserialize(FailingDeserializer::integer())
    }
}

/// Map access that returns valid key but wrong value type.
pub struct WrongValueMapAccess {
    field_name: &'static str,
    /// Whether key has been returned (exposed for tests).
    pub returned_key: bool,
}

impl WrongValueMapAccess {
    /// Create new map access with given field name.
    #[must_use]
    pub fn new(field_name: &'static str) -> Self {
        WrongValueMapAccess {
            field_name,
            returned_key: false,
        }
    }
}

impl<'de> de::MapAccess<'de> for WrongValueMapAccess {
    type Error = DeError;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: de::DeserializeSeed<'de>,
    {
        if self.returned_key {
            return Ok(None);
        }
        self.returned_key = true;
        match seed.deserialize(FieldNameDeserializer {
            field: self.field_name,
        }) {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(e),
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        // Use string value to trigger type mismatch errors for numeric fields
        seed.deserialize(FailingDeserializer::string_value())
    }
}

/// Map access that returns duplicate field.
pub struct DuplicateFieldMapAccess {
    field_name: &'static str,
    /// How many times key has been returned (0, 1, or 2 = done).
    pub key_count: usize,
}

impl DuplicateFieldMapAccess {
    /// Create new map access with given field name.
    #[must_use]
    pub fn new(field_name: &'static str) -> Self {
        DuplicateFieldMapAccess {
            field_name,
            key_count: 0_usize,
        }
    }
}

impl<'de> de::MapAccess<'de> for DuplicateFieldMapAccess {
    type Error = DeError;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: de::DeserializeSeed<'de>,
    {
        if self.key_count >= 2_usize {
            return Ok(None);
        }
        self.key_count += 1_usize;
        match seed.deserialize(FieldNameDeserializer {
            field: self.field_name,
        }) {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(e),
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        seed.deserialize(FailingDeserializer::integer())
    }
}

/// Map access that returns a field twice, with a valid struct value on first access.
/// This is specifically designed to test duplicate_field error paths for complex types.
pub struct StructDuplicateFieldMapAccess {
    field_name: &'static str,
    key_count: usize,
}

impl StructDuplicateFieldMapAccess {
    /// Create a new access that returns the given field name twice with struct values.
    pub const fn new(field_name: &'static str) -> Self {
        Self {
            field_name,
            key_count: 0_usize,
        }
    }
}

impl<'de> de::MapAccess<'de> for StructDuplicateFieldMapAccess {
    type Error = DeError;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: de::DeserializeSeed<'de>,
    {
        if self.key_count >= 2_usize {
            return Ok(None);
        }
        self.key_count += 1_usize;
        match seed.deserialize(FieldNameDeserializer {
            field: self.field_name,
        }) {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(e),
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        // Return a minimal valid struct/sequence for the first value
        seed.deserialize(MinimalStructDeserializer)
    }
}

/// Map access that errors on first next_key call.
/// Used to test error propagation paths in deserialize implementations.
pub struct ErrorOnKeyMapAccess;

impl<'de> de::MapAccess<'de> for ErrorOnKeyMapAccess {
    type Error = DeError;

    fn next_key_seed<K>(&mut self, _seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: de::DeserializeSeed<'de>,
    {
        Err(de::Error::custom("injected next_key error"))
    }

    fn next_value_seed<V>(&mut self, _seed: V) -> Result<V::Value, Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        Err(de::Error::custom("no value after key error"))
    }
}

/// Map access that succeeds on key but errors on next_value.
/// Used to test error propagation paths for value deserialization.
pub struct ErrorOnValueMapAccess {
    field_name: &'static str,
    returned_key: bool,
}

impl ErrorOnValueMapAccess {
    /// Create new map access with given field name.
    #[must_use]
    pub const fn new(field_name: &'static str) -> Self {
        Self {
            field_name,
            returned_key: false,
        }
    }
}

impl<'de> de::MapAccess<'de> for ErrorOnValueMapAccess {
    type Error = DeError;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: de::DeserializeSeed<'de>,
    {
        if self.returned_key {
            return Ok(None);
        }
        self.returned_key = true;
        match seed.deserialize(FieldNameDeserializer {
            field: self.field_name,
        }) {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(e),
        }
    }

    fn next_value_seed<V>(&mut self, _seed: V) -> Result<V::Value, Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        Err(de::Error::custom("injected next_value error"))
    }
}

/// Deserializer that returns minimal valid data for struct or sequence types.
/// For structs, it provides a map with integer values for all expected fields.
pub struct MinimalStructDeserializer;

impl<'de> de::Deserializer<'de> for MinimalStructDeserializer {
    type Error = DeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        // Try to visit as an empty map (works for most structs)
        visitor.visit_map(EmptyMapAccess)
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        // Provide a map that returns integer values for all expected fields
        visitor.visit_map(AllFieldsMapAccess::new(fields))
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(EmptySeqAccess)
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        // Return "Left" for NanDirection and similar enum-like string types
        visitor.visit_str("Left")
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char string bytes
        byte_buf option unit unit_struct newtype_struct tuple tuple_struct
        map enum identifier ignored_any
    }
}

/// Map access that returns all expected fields with integer values.
pub(super) struct AllFieldsMapAccess {
    /// The list of fields to return.
    pub(super) fields: &'static [&'static str],
    /// Current index.
    pub(super) index: usize,
}

impl AllFieldsMapAccess {
    /// Create new map access with given field list.
    pub(super) const fn new(fields: &'static [&'static str]) -> Self {
        Self { fields, index: 0 }
    }
}

impl<'de> de::MapAccess<'de> for AllFieldsMapAccess {
    type Error = DeError;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: de::DeserializeSeed<'de>,
    {
        if self.index >= self.fields.len() {
            return Ok(None);
        }
        let field = self.fields[self.index];
        self.index += 1_usize;
        match seed.deserialize(FieldNameDeserializer { field }) {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(e),
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        // Return a minimal valid value (integer or nested struct)
        seed.deserialize(MinimalValueDeserializer)
    }
}

/// Deserializer that returns minimal valid values for any type.
pub(crate) struct MinimalValueDeserializer;

impl<'de> de::Deserializer<'de> for MinimalValueDeserializer {
    type Error = DeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        // Default to integer for unknown types
        visitor.visit_i64(1_i64)
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(AllFieldsMapAccess::new(fields))
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(EmptySeqAccess)
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        // Return "Right" for NanDirection to cover the Right branch
        visitor.visit_str("Right")
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char string bytes
        byte_buf option unit unit_struct newtype_struct tuple tuple_struct
        map enum identifier ignored_any
    }
}

/// Empty map access that returns no keys.
pub(super) struct EmptyMapAccess;

impl<'de> de::MapAccess<'de> for EmptyMapAccess {
    type Error = DeError;

    fn next_key_seed<K>(&mut self, _seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: de::DeserializeSeed<'de>,
    {
        Ok(None)
    }

    fn next_value_seed<V>(&mut self, _seed: V) -> Result<V::Value, Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        Err(de::Error::custom("no value"))
    }
}

/// Empty sequence access that returns no elements.
pub(super) struct EmptySeqAccess;

impl<'de> de::SeqAccess<'de> for EmptySeqAccess {
    type Error = DeError;

    fn next_element_seed<T>(&mut self, _seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: de::DeserializeSeed<'de>,
    {
        Ok(None)
    }
}

/// Deserializer that returns a field name string.
pub struct FieldNameDeserializer {
    /// The field name to return.
    pub field: &'static str,
}

impl<'de> de::Deserializer<'de> for FieldNameDeserializer {
    type Error = DeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_str(self.field)
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_str(self.field)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map struct enum ignored_any
    }
}

impl<'de> de::Deserializer<'de> for FailingDeserializer {
    type Error = DeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        match self.mode {
            WrongTypeMode::Integer => visitor.visit_i64(42_i64),
            WrongTypeMode::StringValue => visitor.visit_str("wrong_type_string"),
            WrongTypeMode::MapWithIntegerKey => {
                visitor.visit_map(IntegerKeyMapAccess { done: false })
            }
            WrongTypeMode::MapWithWrongValue(field) => {
                visitor.visit_map(WrongValueMapAccess::new(field))
            }
            WrongTypeMode::MapWithDuplicateField(field) => {
                visitor.visit_map(DuplicateFieldMapAccess::new(field))
            }
            WrongTypeMode::StructDuplicateField(field) => {
                visitor.visit_map(StructDuplicateFieldMapAccess::new(field))
            }
            WrongTypeMode::MapErrorOnKey => visitor.visit_map(ErrorOnKeyMapAccess),
            WrongTypeMode::MapErrorOnValue(field) => {
                visitor.visit_map(ErrorOnValueMapAccess::new(field))
            }
        }
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        match self.mode {
            WrongTypeMode::Integer => visitor.visit_i64(42_i64),
            WrongTypeMode::StringValue => visitor.visit_str("wrong_type_string"),
            WrongTypeMode::MapWithIntegerKey => {
                visitor.visit_map(IntegerKeyMapAccess { done: false })
            }
            WrongTypeMode::MapWithWrongValue(field) => {
                visitor.visit_map(WrongValueMapAccess::new(field))
            }
            WrongTypeMode::MapWithDuplicateField(field) => {
                visitor.visit_map(DuplicateFieldMapAccess::new(field))
            }
            WrongTypeMode::StructDuplicateField(field) => {
                visitor.visit_map(StructDuplicateFieldMapAccess::new(field))
            }
            WrongTypeMode::MapErrorOnKey => visitor.visit_map(ErrorOnKeyMapAccess),
            WrongTypeMode::MapErrorOnValue(field) => {
                visitor.visit_map(ErrorOnValueMapAccess::new(field))
            }
        }
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(42_i64)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum ignored_any
    }
}

// =============================================================================
// Separate Deserializer Types
// =============================================================================
// These types avoid phantom generic instantiations by having each deserializer
// use only one specific MapAccess type in its implementation.

/// Deserializer that returns an integer key (for testing field expecting error).
pub struct IntegerKeyDeserializer;

impl<'de> de::Deserializer<'de> for IntegerKeyDeserializer {
    type Error = DeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(IntegerKeyMapAccess { done: false })
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(IntegerKeyMapAccess { done: false })
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(42_i64)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum ignored_any
    }
}

/// Deserializer that returns a map with duplicate field.
pub struct DuplicateFieldDeserializer {
    field_name: &'static str,
}

impl DuplicateFieldDeserializer {
    /// Create a new duplicate field deserializer.
    #[must_use]
    pub const fn new(field_name: &'static str) -> Self {
        Self { field_name }
    }
}

impl<'de> de::Deserializer<'de> for DuplicateFieldDeserializer {
    type Error = DeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(DuplicateFieldMapAccess::new(self.field_name))
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(DuplicateFieldMapAccess::new(self.field_name))
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(42_i64)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum ignored_any
    }
}

/// Deserializer that returns duplicate field with struct values.
pub struct StructDuplicateFieldDeserializer {
    field_name: &'static str,
}

impl StructDuplicateFieldDeserializer {
    /// Create a new struct duplicate field deserializer.
    #[must_use]
    pub const fn new(field_name: &'static str) -> Self {
        Self { field_name }
    }
}

impl<'de> de::Deserializer<'de> for StructDuplicateFieldDeserializer {
    type Error = DeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(StructDuplicateFieldMapAccess::new(self.field_name))
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(StructDuplicateFieldMapAccess::new(self.field_name))
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(42_i64)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum ignored_any
    }
}

/// Deserializer that returns valid key but wrong value type.
pub struct WrongValueDeserializer {
    field_name: &'static str,
}

impl WrongValueDeserializer {
    /// Create a new wrong value deserializer.
    #[must_use]
    pub const fn new(field_name: &'static str) -> Self {
        Self { field_name }
    }
}

impl<'de> de::Deserializer<'de> for WrongValueDeserializer {
    type Error = DeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(WrongValueMapAccess::new(self.field_name))
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(WrongValueMapAccess::new(self.field_name))
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(42_i64)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum ignored_any
    }
}

/// Deserializer that errors on first next_key call.
pub struct ErrorOnKeyDeserializer;

impl<'de> de::Deserializer<'de> for ErrorOnKeyDeserializer {
    type Error = DeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(ErrorOnKeyMapAccess)
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(ErrorOnKeyMapAccess)
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(42_i64)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum ignored_any
    }
}

/// Deserializer that succeeds on key but errors on value.
pub struct ErrorOnValueDeserializer {
    field_name: &'static str,
}

impl ErrorOnValueDeserializer {
    /// Create a new error on value deserializer.
    #[must_use]
    pub const fn new(field_name: &'static str) -> Self {
        Self { field_name }
    }
}

impl<'de> de::Deserializer<'de> for ErrorOnValueDeserializer {
    type Error = DeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(ErrorOnValueMapAccess::new(self.field_name))
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(ErrorOnValueMapAccess::new(self.field_name))
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(42_i64)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum ignored_any
    }
}

/// Deserializer that returns an integer value.
pub struct IntegerDeserializer;

impl<'de> de::Deserializer<'de> for IntegerDeserializer {
    type Error = DeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(42_i64)
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(42_i64)
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(42_i64)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum ignored_any
    }
}

/// Deserializer that returns a string value.
pub struct StringDeserializer;

impl<'de> de::Deserializer<'de> for StringDeserializer {
    type Error = DeError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_str("wrong_type_string")
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_str("wrong_type_string")
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(42_i64)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum ignored_any
    }
}

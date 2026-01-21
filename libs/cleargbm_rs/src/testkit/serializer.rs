//! Failing serializer for testing error paths.

use core::fmt::{self, Display};
use serde::ser::{self, Serialize};

/// Error type for failing serializer.
#[derive(Debug)]
pub struct SerError {
    /// Error message.
    pub message: String,
}

impl Display for SerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SerError {}

impl ser::Error for SerError {
    fn custom<T: Display>(msg: T) -> Self {
        SerError {
            message: msg.to_string(),
        }
    }
}

/// Serializer that fails after N fields.
pub struct FailingSerializer {
    /// Current field count.
    count: usize,
    /// Fail after this many fields.
    fail_after: usize,
    /// Whether to fail on serialize_struct call.
    fail_on_struct: bool,
    /// Whether to fail on end() call.
    fail_on_end: bool,
}

impl FailingSerializer {
    /// Create serializer that fails after n fields.
    #[must_use]
    pub fn fail_after(n: usize) -> Self {
        FailingSerializer {
            count: 0,
            fail_after: n,
            fail_on_struct: false,
            fail_on_end: false,
        }
    }

    /// Create serializer that fails immediately on serialize_struct.
    #[must_use]
    pub fn fail_on_struct() -> Self {
        FailingSerializer {
            count: 0,
            fail_after: usize::MAX,
            fail_on_struct: true,
            fail_on_end: false,
        }
    }

    /// Create serializer that succeeds on all fields but fails on end().
    #[must_use]
    pub fn fail_on_end() -> Self {
        FailingSerializer {
            count: 0,
            fail_after: usize::MAX,
            fail_on_struct: false,
            fail_on_end: true,
        }
    }
}

/// Struct serializer state.
pub struct FailingSerializerStruct<'a> {
    /// Reference to the parent serializer.
    ser: &'a mut FailingSerializer,
}

impl<'a> ser::SerializeStruct for FailingSerializerStruct<'a> {
    type Ok = ();
    type Error = SerError;

    fn serialize_field<T>(&mut self, _key: &'static str, _value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        self.ser.count += 1;
        if self.ser.count > self.ser.fail_after {
            Err(SerError {
                message: "intentional failure".to_string(),
            })
        } else {
            Ok(())
        }
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        if self.ser.fail_on_end {
            Err(SerError {
                message: "intentional failure on end".to_string(),
            })
        } else {
            Ok(())
        }
    }
}

impl<'a> ser::Serializer for &'a mut FailingSerializer {
    type Ok = ();
    type Error = SerError;
    type SerializeSeq = ser::Impossible<(), SerError>;
    type SerializeTuple = ser::Impossible<(), SerError>;
    type SerializeTupleStruct = ser::Impossible<(), SerError>;
    type SerializeTupleVariant = ser::Impossible<(), SerError>;
    type SerializeMap = ser::Impossible<(), SerError>;
    type SerializeStruct = FailingSerializerStruct<'a>;
    type SerializeStructVariant = ser::Impossible<(), SerError>;

    fn serialize_bool(self, _v: bool) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_i8(self, _v: i8) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_i16(self, _v: i16) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_i32(self, _v: i32) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_i64(self, _v: i64) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_u8(self, _v: u8) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_u16(self, _v: u16) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_u32(self, _v: u32) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_u64(self, _v: u64) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_f32(self, _v: f32) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_f64(self, _v: f64) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_char(self, _v: char) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_str(self, _v: &str) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_bytes(self, _v: &[u8]) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_none(self) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_some<T: ?Sized + Serialize>(self, _value: &T) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_unit(self) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
    ) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _value: &T,
    ) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<(), SerError> {
        Ok(())
    }
    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, SerError> {
        Err(SerError {
            message: "seq not supported".to_string(),
        })
    }
    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, SerError> {
        Err(SerError {
            message: "tuple not supported".to_string(),
        })
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, SerError> {
        Err(SerError {
            message: "tuple_struct not supported".to_string(),
        })
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, SerError> {
        Err(SerError {
            message: "tuple_variant not supported".to_string(),
        })
    }
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, SerError> {
        Err(SerError {
            message: "map not supported".to_string(),
        })
    }
    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, SerError> {
        if self.fail_on_struct {
            Err(SerError {
                message: "intentional failure on serialize_struct".to_string(),
            })
        } else {
            Ok(FailingSerializerStruct { ser: self })
        }
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, SerError> {
        Err(SerError {
            message: "struct_variant not supported".to_string(),
        })
    }
}

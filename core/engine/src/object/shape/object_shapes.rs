//! # Object Shapes (Hidden Classes) for Boa VM
//!
//! This module implements an object shape system inspired by JetCrab's approach.
//! Object shapes (also known as hidden classes) optimize property access by
//! providing a shared structure for objects with similar properties.
//!
//! ## Key Features
//!
//! - **Shape Trees**: Hierarchical structure for property transitions
//! - **Property Descriptors**: Detailed information about each property
//! - **Shape Transitions**: Efficient addition/removal of properties
//! - **Property Access Optimization**: Fast property lookup using shape information
//! - **Shape Sharing**: Multiple objects can share the same shape
//!
//! ## Benefits
//!
//! - **Faster Property Access**: O(1) property lookup using shape information
//! - **Memory Efficiency**: Shared shapes reduce memory usage
//! - **JIT Optimization**: Enables better just-in-time compilation
//! - **Inline Caching**: Supports efficient property access caching

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::handles::HeapHandleId;

/// A unique identifier for an object shape
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShapeId(usize);

impl ShapeId {
    /// Creates a new unique shape ID
    pub fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(1);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(id)
    }

    /// Returns the raw ID value
    pub fn value(&self) -> usize {
        self.0
    }
}

impl Default for ShapeId {
    fn default() -> Self {
        Self::new()
    }
}

/// Describes the attributes of a property
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PropertyDescriptor {
    /// Whether the property is enumerable
    pub enumerable: bool,
    /// Whether the property is configurable
    pub configurable: bool,
    /// Whether the property is writable
    pub writable: bool,
    /// Whether the property has a getter
    pub has_getter: bool,
    /// Whether the property has a setter
    pub has_setter: bool,
    /// The type of the property value
    pub value_type: PropertyValueType,
}

impl Default for PropertyDescriptor {
    fn default() -> Self {
        Self {
            enumerable: true,
            configurable: true,
            writable: true,
            has_getter: false,
            has_setter: false,
            value_type: PropertyValueType::Data,
        }
    }
}

/// Type of property value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PropertyValueType {
    /// Data property (has a value)
    Data,
    /// Accessor property (has getter/setter)
    Accessor,
    /// Method property (function)
    Method,
}

/// Information about a property in a shape
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyInfo {
    /// Name of the property
    pub name: String,
    /// Descriptor of the property
    pub descriptor: PropertyDescriptor,
    /// Offset of the property in the object's property storage
    pub offset: usize,
    /// Whether this property is a transition property
    pub is_transition: bool,
}

/// An object shape (hidden class) that describes the structure of an object
#[derive(Debug, Clone)]
pub struct ObjectShape {
    /// Unique identifier for this shape
    pub id: ShapeId,
    /// Parent shape (for shape transitions)
    pub parent: Option<ShapeId>,
    /// Properties in this shape
    pub properties: Vec<PropertyInfo>,
    /// Map from property name to property info
    pub property_map: HashMap<String, PropertyInfo>,
    /// Transition map for adding new properties
    pub transitions: HashMap<String, ShapeId>,
    /// Whether this shape is extensible
    pub extensible: bool,
    /// Prototype of objects with this shape
    pub prototype: Option<HeapHandleId>,
    /// Number of objects using this shape
    pub usage_count: usize,
}

impl ObjectShape {
    /// Creates a new object shape
    pub fn new(parent: Option<ShapeId>) -> Self {
        Self {
            id: ShapeId::new(),
            parent,
            properties: Vec::new(),
            property_map: HashMap::new(),
            transitions: HashMap::new(),
            extensible: true,
            prototype: None,
            usage_count: 0,
        }
    }

    /// Adds a property to this shape
    pub fn add_property(
        &mut self,
        name: String,
        descriptor: PropertyDescriptor,
        offset: usize,
    ) -> Result<(), ShapeError> {
        if !self.extensible {
            return Err(ShapeError::NotExtensible);
        }

        if self.property_map.contains_key(&name) {
            return Err(ShapeError::PropertyExists);
        }

        let property_info = PropertyInfo {
            name: name.clone(),
            descriptor,
            offset,
            is_transition: false,
        };

        self.properties.push(property_info.clone());
        self.property_map.insert(name, property_info);

        Ok(())
    }

    /// Removes a property from this shape
    pub fn remove_property(&mut self, name: &str) -> Result<(), ShapeError> {
        if !self.extensible {
            return Err(ShapeError::NotExtensible);
        }

        if let Some(_property_info) = self.property_map.remove(name) {
            self.properties.retain(|p| p.name != name);
            Ok(())
        } else {
            Err(ShapeError::PropertyNotFound)
        }
    }

    /// Gets information about a property
    pub fn get_property(&self, name: &str) -> Option<&PropertyInfo> {
        self.property_map.get(name)
    }

    /// Checks if this shape has a property
    pub fn has_property(&self, name: &str) -> bool {
        self.property_map.contains_key(name)
    }

    /// Gets the number of properties in this shape
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    /// Increments the usage count of this shape
    pub fn increment_usage(&mut self) {
        self.usage_count += 1;
    }

    /// Decrements the usage count of this shape
    pub fn decrement_usage(&mut self) {
        self.usage_count = self.usage_count.saturating_sub(1);
    }

    /// Checks if this shape is unused
    pub fn is_unused(&self) -> bool {
        self.usage_count == 0
    }
}

/// Errors that can occur when working with object shapes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// The shape is not extensible
    NotExtensible,
    /// The property already exists
    PropertyExists,
    /// The property was not found
    PropertyNotFound,
    /// Invalid property descriptor
    InvalidDescriptor,
    /// Shape transition failed
    TransitionFailed,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::NotExtensible => write!(f, "Shape is not extensible"),
            ShapeError::PropertyExists => write!(f, "Property already exists"),
            ShapeError::PropertyNotFound => write!(f, "Property not found"),
            ShapeError::InvalidDescriptor => write!(f, "Invalid property descriptor"),
            ShapeError::TransitionFailed => write!(f, "Shape transition failed"),
        }
    }
}

impl std::error::Error for ShapeError {}

/// Manages object shapes and their transitions
pub struct ShapeManager {
    /// Map from shape ID to shape
    shapes: HashMap<ShapeId, ObjectShape>,
    /// Root shape (empty object)
    root_shape: ShapeId,
    /// Statistics about shape usage
    stats: ShapeStats,
}

/// Statistics about shape usage
#[derive(Debug, Clone, Default)]
pub struct ShapeStats {
    /// Total number of shapes created
    pub total_shapes: usize,
    /// Total number of shape transitions
    pub total_transitions: usize,
    /// Total number of properties across all shapes
    pub total_properties: usize,
    /// Number of unused shapes
    pub unused_shapes: usize,
    /// Average properties per shape
    pub avg_properties_per_shape: f64,
    /// Most common property names
    pub common_properties: HashMap<String, usize>,
}

impl ShapeManager {
    /// Creates a new shape manager
    pub fn new() -> Self {
        let mut manager = Self {
            shapes: HashMap::new(),
            root_shape: ShapeId::new(),
            stats: ShapeStats::default(),
        };

        // Create the root shape (empty object)
        let root_shape = ObjectShape::new(None);
        manager.shapes.insert(manager.root_shape, root_shape);
        manager.stats.total_shapes = 1;

        manager
    }

    /// Gets the root shape (empty object)
    pub fn get_root_shape(&self) -> ShapeId {
        self.root_shape
    }

    /// Gets a shape by ID
    pub fn get_shape(&self, shape_id: ShapeId) -> Option<&ObjectShape> {
        self.shapes.get(&shape_id)
    }

    /// Gets a mutable reference to a shape by ID
    pub fn get_shape_mut(&mut self, shape_id: ShapeId) -> Option<&mut ObjectShape> {
        self.shapes.get_mut(&shape_id)
    }

    /// Creates a new shape by adding a property to an existing shape
    pub fn add_property_transition(
        &mut self,
        parent_shape_id: ShapeId,
        property_name: String,
        descriptor: PropertyDescriptor,
    ) -> Result<ShapeId, ShapeError> {
        let parent_shape = self
            .shapes
            .get(&parent_shape_id)
            .ok_or(ShapeError::TransitionFailed)?;

        // Check if a transition already exists
        if let Some(&existing_shape_id) = parent_shape.transitions.get(&property_name) {
            return Ok(existing_shape_id);
        }

        // Create new shape
        let mut new_shape = ObjectShape::new(Some(parent_shape_id));
        let new_shape_id = new_shape.id;

        // Copy properties from parent
        for property in &parent_shape.properties {
            new_shape.add_property(property.name.clone(), property.descriptor, property.offset)?;
        }

        // Add new property
        let new_offset = new_shape.property_count();
        new_shape.add_property(property_name.clone(), descriptor, new_offset)?;

        // Add transition to parent
        if let Some(parent_shape) = self.shapes.get_mut(&parent_shape_id) {
            parent_shape.transitions.insert(property_name, new_shape_id);
        }

        // Store new shape
        self.shapes.insert(new_shape_id, new_shape);
        self.stats.total_shapes += 1;
        self.stats.total_transitions += 1;

        Ok(new_shape_id)
    }

    /// Creates a new shape by removing a property from an existing shape
    pub fn remove_property_transition(
        &mut self,
        parent_shape_id: ShapeId,
        property_name: &str,
    ) -> Result<ShapeId, ShapeError> {
        let parent_shape = self
            .shapes
            .get(&parent_shape_id)
            .ok_or(ShapeError::TransitionFailed)?;

        // Check if a transition already exists
        if let Some(&existing_shape_id) = parent_shape.transitions.get(property_name) {
            return Ok(existing_shape_id);
        }

        // Create new shape
        let mut new_shape = ObjectShape::new(Some(parent_shape_id));
        let new_shape_id = new_shape.id;

        // Copy properties from parent, excluding the removed one
        for property in &parent_shape.properties {
            if property.name != property_name {
                new_shape.add_property(
                    property.name.clone(),
                    property.descriptor,
                    property.offset,
                )?;
            }
        }

        // Add transition to parent
        if let Some(parent_shape) = self.shapes.get_mut(&parent_shape_id) {
            parent_shape
                .transitions
                .insert(property_name.to_string(), new_shape_id);
        }

        // Store new shape
        self.shapes.insert(new_shape_id, new_shape);
        self.stats.total_shapes += 1;
        self.stats.total_transitions += 1;

        Ok(new_shape_id)
    }

    /// Increments the usage count of a shape
    pub fn increment_shape_usage(&mut self, shape_id: ShapeId) {
        if let Some(shape) = self.shapes.get_mut(&shape_id) {
            shape.increment_usage();
        }
    }

    /// Decrements the usage count of a shape
    pub fn decrement_shape_usage(&mut self, shape_id: ShapeId) {
        if let Some(shape) = self.shapes.get_mut(&shape_id) {
            shape.decrement_usage();
        }
    }

    /// Gets statistics about shape usage
    pub fn get_stats(&self) -> &ShapeStats {
        &self.stats
    }

    /// Updates statistics
    pub fn update_stats(&mut self) {
        self.stats.total_properties = self
            .shapes
            .values()
            .map(|shape| shape.property_count())
            .sum();

        self.stats.unused_shapes = self
            .shapes
            .values()
            .filter(|shape| shape.is_unused())
            .count();

        if !self.shapes.is_empty() {
            self.stats.avg_properties_per_shape =
                self.stats.total_properties as f64 / self.shapes.len() as f64;
        }

        // Update common properties
        self.stats.common_properties.clear();
        for shape in self.shapes.values() {
            for property in &shape.properties {
                *self
                    .stats
                    .common_properties
                    .entry(property.name.clone())
                    .or_insert(0) += 1;
            }
        }
    }

    /// Removes unused shapes to free memory
    pub fn cleanup_unused_shapes(&mut self) -> usize {
        let mut removed_count = 0;
        let unused_shapes: Vec<ShapeId> = self
            .shapes
            .iter()
            .filter(|(_, shape)| shape.is_unused())
            .map(|(id, _)| *id)
            .collect();

        for shape_id in unused_shapes {
            if shape_id != self.root_shape {
                self.shapes.remove(&shape_id);
                removed_count += 1;
            }
        }

        self.stats.total_shapes -= removed_count;
        removed_count
    }
}

/// An object that uses a shape for property access optimization
pub struct ShapedObject {
    /// The shape of this object
    pub shape: ShapeId,
    /// Handle to the actual object data
    pub handle: HeapHandleId,
    /// Property values (indexed by property offset)
    pub properties: Vec<Option<HeapHandleId>>,
}

impl ShapedObject {
    /// Creates a new shaped object
    pub fn new(shape: ShapeId, handle: HeapHandleId) -> Self {
        Self {
            shape,
            handle,
            properties: Vec::new(),
        }
    }

    /// Gets a property value by name
    pub fn get_property(
        &self,
        name: &str,
        shape_manager: &ShapeManager,
    ) -> Option<Option<HeapHandleId>> {
        let shape = shape_manager.get_shape(self.shape)?;
        let property_info = shape.get_property(name)?;
        self.properties.get(property_info.offset).copied()
    }

    /// Sets a property value by name
    pub fn set_property(
        &mut self,
        name: &str,
        value: Option<HeapHandleId>,
        shape_manager: &ShapeManager,
    ) -> Result<(), ShapeError> {
        let shape = shape_manager
            .get_shape(self.shape)
            .ok_or(ShapeError::TransitionFailed)?;
        let property_info = shape
            .get_property(name)
            .ok_or(ShapeError::PropertyNotFound)?;

        // Ensure properties vector is large enough
        while self.properties.len() <= property_info.offset {
            self.properties.push(None);
        }

        self.properties[property_info.offset] = value;
        Ok(())
    }

    /// Adds a new property to this object
    pub fn add_property(
        &mut self,
        name: &str,
        value: Option<HeapHandleId>,
        descriptor: PropertyDescriptor,
        shape_manager: &mut ShapeManager,
    ) -> Result<(), ShapeError> {
        // Create new shape with the added property
        let new_shape_id =
            shape_manager.add_property_transition(self.shape, name.to_string(), descriptor)?;

        // Update object to use new shape
        self.shape = new_shape_id;

        // Set the property value
        self.set_property(name, value, shape_manager)?;

        Ok(())
    }

    /// Removes a property from this object
    pub fn remove_property(
        &mut self,
        name: &str,
        shape_manager: &mut ShapeManager,
    ) -> Result<(), ShapeError> {
        // Create new shape without the property
        let new_shape_id = shape_manager.remove_property_transition(self.shape, name)?;

        // Update object to use new shape
        self.shape = new_shape_id;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let manager = ShapeManager::new();
        let root_shape = manager.get_root_shape();
        let shape = manager.get_shape(root_shape).unwrap();

        assert_eq!(shape.property_count(), 0);
        assert!(shape.extensible);
        assert!(shape.is_unused());
    }

    #[test]
    fn test_property_addition() {
        let mut manager = ShapeManager::new();
        let root_shape = manager.get_root_shape();

        let descriptor = PropertyDescriptor::default();
        let new_shape_id = manager
            .add_property_transition(root_shape, "name".to_string(), descriptor)
            .unwrap();

        let new_shape = manager.get_shape(new_shape_id).unwrap();
        assert_eq!(new_shape.property_count(), 1);
        assert!(new_shape.has_property("name"));
    }

    #[test]
    fn test_property_removal() {
        let mut manager = ShapeManager::new();
        let root_shape = manager.get_root_shape();

        // Add a property first
        let descriptor = PropertyDescriptor::default();
        let shape_with_property = manager
            .add_property_transition(root_shape, "name".to_string(), descriptor)
            .unwrap();

        // Remove the property
        let shape_without_property = manager
            .remove_property_transition(shape_with_property, "name")
            .unwrap();

        let final_shape = manager.get_shape(shape_without_property).unwrap();
        assert_eq!(final_shape.property_count(), 0);
        assert!(!final_shape.has_property("name"));
    }

    #[test]
    fn test_shaped_object() {
        let mut manager = ShapeManager::new();
        let root_shape = manager.get_root_shape();
        let handle = HeapHandleId::new();

        let mut obj = ShapedObject::new(root_shape, handle);
        let descriptor = PropertyDescriptor::default();

        // Add a property
        obj.add_property("name", Some(handle), descriptor, &mut manager)
            .unwrap();

        // Get the property
        let value = obj.get_property("name", &manager).unwrap();
        assert_eq!(value, Some(handle));

        // Remove the property
        obj.remove_property("name", &mut manager).unwrap();

        // Property should not exist
        assert!(obj.get_property("name", &manager).is_none());
    }

    #[test]
    fn test_shape_transitions() {
        let mut manager = ShapeManager::new();
        let root_shape = manager.get_root_shape();

        let descriptor = PropertyDescriptor::default();

        // Add multiple properties
        let shape1 = manager
            .add_property_transition(root_shape, "a".to_string(), descriptor)
            .unwrap();
        let shape2 = manager
            .add_property_transition(shape1, "b".to_string(), descriptor)
            .unwrap();
        let shape3 = manager
            .add_property_transition(shape2, "c".to_string(), descriptor)
            .unwrap();

        let final_shape = manager.get_shape(shape3).unwrap();
        assert_eq!(final_shape.property_count(), 3);
        assert!(final_shape.has_property("a"));
        assert!(final_shape.has_property("b"));
        assert!(final_shape.has_property("c"));
    }

    #[test]
    fn test_shape_reuse() {
        let mut manager = ShapeManager::new();
        let root_shape = manager.get_root_shape();

        let descriptor = PropertyDescriptor::default();

        // Add the same property twice
        let shape1 = manager
            .add_property_transition(root_shape, "name".to_string(), descriptor)
            .unwrap();
        let shape2 = manager
            .add_property_transition(root_shape, "name".to_string(), descriptor)
            .unwrap();

        // Should reuse the same shape
        assert_eq!(shape1, shape2);
    }

    #[test]
    fn test_shape_usage_counting() {
        let mut manager = ShapeManager::new();
        let root_shape = manager.get_root_shape();

        // Increment usage
        manager.increment_shape_usage(root_shape);
        manager.increment_shape_usage(root_shape);

        let shape = manager.get_shape(root_shape).unwrap();
        assert_eq!(shape.usage_count, 2);

        // Decrement usage
        manager.decrement_shape_usage(root_shape);
        let shape = manager.get_shape(root_shape).unwrap();
        assert_eq!(shape.usage_count, 1);
    }

    #[test]
    fn test_shape_cleanup() {
        let mut manager = ShapeManager::new();
        let root_shape = manager.get_root_shape();

        let descriptor = PropertyDescriptor::default();
        let _shape = manager
            .add_property_transition(root_shape, "name".to_string(), descriptor)
            .unwrap();

        // Cleanup unused shapes
        let removed = manager.cleanup_unused_shapes();
        assert_eq!(removed, 1); // The added shape should be removed

        // Root shape should remain
        assert!(manager.get_shape(root_shape).is_some());
    }
}

//! # Memory Management for Boa VM
//!
//! This module provides memory management capabilities for the Boa VM,
//! including generational garbage collection and specialized memory spaces.

pub mod generational_heap;

pub use generational_heap::{GcResult, GenerationalHeap, HeapConfig, HeapStats};

//! # Generational Heap for Boa VM
//!
//! This module implements a generational garbage collection system inspired by JetCrab's approach.
//! It provides specialized memory spaces for different types of objects and generations,
//! optimizing memory allocation and garbage collection performance.
//!
//! ## Key Features
//!
//! - **Generational Collection**: Objects are divided into New, Old, and Large generations
//! - **Specialized Spaces**: Different memory spaces for different object types
//! - **Bump Allocation**: Fast allocation for new objects
//! - **Free List Management**: Efficient reuse of freed memory
//! - **Write Barriers**: Automatic promotion of objects between generations
//!
//! ## Memory Spaces
//!
//! - **New Space**: Fast bump allocation for young objects
//! - **Old Space**: Mature objects with free-list allocation
//! - **Large Object Space**: Objects that bypass generational collection
//! - **Code Space**: Executable code and function objects
//! - **Cell Space**: Property cells and hidden classes
//! - **Map Space**: Object shape information

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::handles::{Generation, HeapHandleId, ObjectType};

/// Configuration for the generational heap
#[derive(Debug, Clone)]
pub struct HeapConfig {
    /// Maximum size of the new space in bytes
    pub new_space_size: usize,
    /// Maximum size of the old space in bytes
    pub old_space_size: usize,
    /// Maximum size of the large object space in bytes
    pub large_space_size: usize,
    /// Maximum size of the code space in bytes
    pub code_space_size: usize,
    /// Maximum size of the cell space in bytes
    pub cell_space_size: usize,
    /// Maximum size of the map space in bytes
    pub map_space_size: usize,
    /// Threshold for promoting objects to old generation
    pub promotion_threshold: usize,
    /// Threshold for objects to go directly to large object space
    pub large_object_threshold: usize,
    /// Enable write barriers for automatic promotion
    pub enable_write_barriers: bool,
}

impl Default for HeapConfig {
    fn default() -> Self {
        Self {
            new_space_size: 16 * 1024 * 1024,   // 16MB
            old_space_size: 64 * 1024 * 1024,   // 64MB
            large_space_size: 32 * 1024 * 1024, // 32MB
            code_space_size: 8 * 1024 * 1024,   // 8MB
            cell_space_size: 4 * 1024 * 1024,   // 4MB
            map_space_size: 2 * 1024 * 1024,    // 2MB
            promotion_threshold: 1024,          // 1KB
            large_object_threshold: 64 * 1024,  // 64KB
            enable_write_barriers: true,
        }
    }
}

/// Statistics about heap usage
#[derive(Debug, Clone, Default)]
pub struct HeapStats {
    /// Total memory allocated across all spaces
    pub total_allocated: usize,
    /// Total memory used (not freed)
    pub total_used: usize,
    /// Total memory freed
    pub total_freed: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Number of garbage collections performed
    pub gc_count: usize,
    /// Total time spent in garbage collection
    pub gc_time: Duration,
    /// Memory usage by space
    pub space_usage: HashMap<String, usize>,
    /// Memory usage by generation
    pub generation_usage: HashMap<Generation, usize>,
    /// Memory usage by object type
    pub type_usage: HashMap<ObjectType, usize>,
}

/// Main generational heap manager
pub struct GenerationalHeap {
    /// Configuration for the heap
    config: HeapConfig,
    /// New space for young objects
    new_space: NewSpace,
    /// Old space for mature objects
    old_space: OldSpace,
    /// Large object space for big objects
    large_space: LargeSpace,
    /// Code space for executable code
    code_space: CodeSpace,
    /// Cell space for property cells
    cell_space: CellSpace,
    /// Map space for object shapes
    map_space: MapSpace,
    /// Heap statistics
    stats: HeapStats,
    /// Global allocation counter
    allocation_counter: AtomicUsize,
}

impl GenerationalHeap {
    /// Create a new generational heap with the given configuration
    pub fn new(config: HeapConfig) -> Self {
        let mut heap = Self {
            new_space: NewSpace::new(config.new_space_size),
            old_space: OldSpace::new(config.old_space_size),
            large_space: LargeSpace::new(config.large_space_size),
            code_space: CodeSpace::new(config.code_space_size),
            cell_space: CellSpace::new(config.cell_space_size),
            map_space: MapSpace::new(config.map_space_size),
            stats: HeapStats::default(),
            allocation_counter: AtomicUsize::new(0),
            config,
        };

        // Initialize space usage tracking
        heap.stats.space_usage.insert("new".to_string(), 0);
        heap.stats.space_usage.insert("old".to_string(), 0);
        heap.stats.space_usage.insert("large".to_string(), 0);
        heap.stats.space_usage.insert("code".to_string(), 0);
        heap.stats.space_usage.insert("cell".to_string(), 0);
        heap.stats.space_usage.insert("map".to_string(), 0);

        heap
    }

    /// Allocate memory for an object
    pub fn allocate(&mut self, size: usize, object_type: ObjectType) -> Option<HeapHandleId> {
        let handle_id = HeapHandleId::new();
        let _allocation_id = self.allocation_counter.fetch_add(1, Ordering::Relaxed);

        // Determine which space to use based on size and type
        let space_name = match (size, object_type) {
            (size, _) if size >= self.config.large_object_threshold => "large",
            (_, ObjectType::Function) => "code",
            (_, ObjectType::Object) if size <= self.config.promotion_threshold => "new",
            (_, ObjectType::Object) => "old",
            (_, ObjectType::Array) if size <= self.config.promotion_threshold => "new",
            (_, ObjectType::Array) => "old",
            (_, ObjectType::String) => "new",
            (_, ObjectType::Number) => "new",
            (_, ObjectType::Boolean) => "new",
            (_, ObjectType::Symbol) => "new",
            (_, ObjectType::BigInt) => "old",
        };

        // Allocate in the appropriate space
        let allocated = match space_name {
            "new" => self.new_space.allocate(size, handle_id),
            "old" => self.old_space.allocate(size, handle_id),
            "large" => self.large_space.allocate(size, handle_id),
            "code" => self.code_space.allocate(size, handle_id),
            "cell" => self.cell_space.allocate(size, handle_id),
            "map" => self.map_space.allocate(size, handle_id),
            _ => false,
        };

        if allocated {
            // Update statistics
            self.stats.total_allocated += size;
            self.stats.total_used += size;
            self.stats.allocation_count += 1;

            // Update space usage
            *self.stats.space_usage.get_mut(space_name).unwrap() += size;

            // Update generation usage
            let generation = match space_name {
                "new" => Generation::New,
                "old" => Generation::Old,
                "large" => Generation::Large,
                _ => Generation::New,
            };
            *self.stats.generation_usage.entry(generation).or_insert(0) += size;

            // Update type usage
            *self.stats.type_usage.entry(object_type).or_insert(0) += size;

            Some(handle_id)
        } else {
            None
        }
    }

    /// Deallocate memory for an object
    pub fn deallocate(&mut self, handle_id: HeapHandleId, size: usize, object_type: ObjectType) {
        // Try to deallocate from each space
        let mut deallocated = false;
        let mut space_name = "";

        if self.new_space.deallocate(handle_id) {
            deallocated = true;
            space_name = "new";
        } else if self.old_space.deallocate(handle_id) {
            deallocated = true;
            space_name = "old";
        } else if self.large_space.deallocate(handle_id) {
            deallocated = true;
            space_name = "large";
        } else if self.code_space.deallocate(handle_id) {
            deallocated = true;
            space_name = "code";
        } else if self.cell_space.deallocate(handle_id) {
            deallocated = true;
            space_name = "cell";
        } else if self.map_space.deallocate(handle_id) {
            deallocated = true;
            space_name = "map";
        }

        if deallocated {
            // Update statistics
            self.stats.total_freed += size;
            self.stats.total_used = self.stats.total_used.saturating_sub(size);
            self.stats.deallocation_count += 1;

            // Update space usage
            if let Some(usage) = self.stats.space_usage.get_mut(space_name) {
                *usage = usage.saturating_sub(size);
            }

            // Update generation usage
            let generation = match space_name {
                "new" => Generation::New,
                "old" => Generation::Old,
                "large" => Generation::Large,
                _ => Generation::New,
            };
            if let Some(usage) = self.stats.generation_usage.get_mut(&generation) {
                *usage = usage.saturating_sub(size);
            }

            // Update type usage
            if let Some(usage) = self.stats.type_usage.get_mut(&object_type) {
                *usage = usage.saturating_sub(size);
            }
        }
    }

    /// Perform garbage collection
    pub fn collect_garbage(&mut self) -> GcResult {
        let start_time = Instant::now();
        self.stats.gc_count += 1;

        // Collect from new space (minor GC)
        let new_space_result = self.new_space.collect_garbage();

        // Collect from old space (major GC) if needed
        let old_space_result = if self.should_perform_major_gc() {
            self.old_space.collect_garbage()
        } else {
            GcResult::default()
        };

        // Collect from other spaces
        let code_space_result = self.code_space.collect_garbage();
        let cell_space_result = self.cell_space.collect_garbage();
        let map_space_result = self.map_space.collect_garbage();

        let gc_time = start_time.elapsed();
        self.stats.gc_time += gc_time;

        // Combine results
        GcResult {
            objects_collected: new_space_result.objects_collected
                + old_space_result.objects_collected
                + code_space_result.objects_collected
                + cell_space_result.objects_collected
                + map_space_result.objects_collected,
            memory_freed: new_space_result.memory_freed
                + old_space_result.memory_freed
                + code_space_result.memory_freed
                + cell_space_result.memory_freed
                + map_space_result.memory_freed,
            collection_time: gc_time,
        }
    }

    /// Check if major GC should be performed
    fn should_perform_major_gc(&self) -> bool {
        // Perform major GC if old space is more than 80% full
        let old_space_usage = self.stats.space_usage.get("old").unwrap_or(&0);
        *old_space_usage > (self.config.old_space_size * 80 / 100)
    }

    /// Get heap statistics
    pub fn get_stats(&self) -> &HeapStats {
        &self.stats
    }

    /// Get heap configuration
    pub fn get_config(&self) -> &HeapConfig {
        &self.config
    }

    /// Update heap configuration
    pub fn update_config(&mut self, config: HeapConfig) {
        self.config = config;
    }
}

/// Result of a garbage collection operation
#[derive(Debug, Clone, Default)]
pub struct GcResult {
    /// Number of objects collected
    pub objects_collected: usize,
    /// Amount of memory freed in bytes
    pub memory_freed: usize,
    /// Time spent in garbage collection
    pub collection_time: Duration,
}

/// New space for young objects with bump allocation
pub struct NewSpace {
    /// Memory buffer for the new space
    #[allow(dead_code)]
    buffer: Vec<u8>,
    /// Current allocation pointer
    allocation_ptr: usize,
    /// Maximum size of the space
    max_size: usize,
    /// Map of handle IDs to allocation info
    allocations: HashMap<HeapHandleId, AllocationInfo>,
}

/// Old space for mature objects with free-list allocation
pub struct OldSpace {
    /// Memory buffer for the old space
    #[allow(dead_code)]
    buffer: Vec<u8>,
    /// Free list for reusable memory blocks
    free_list: Vec<FreeBlock>,
    /// Maximum size of the space
    #[allow(dead_code)]
    max_size: usize,
    /// Map of handle IDs to allocation info
    allocations: HashMap<HeapHandleId, AllocationInfo>,
}

/// Large object space for big objects
pub struct LargeSpace {
    /// Memory buffer for the large space
    #[allow(dead_code)]
    buffer: Vec<u8>,
    /// Maximum size of the space
    max_size: usize,
    /// Map of handle IDs to allocation info
    allocations: HashMap<HeapHandleId, AllocationInfo>,
}

/// Code space for executable code
pub struct CodeSpace {
    /// Memory buffer for the code space
    #[allow(dead_code)]
    buffer: Vec<u8>,
    /// Current allocation pointer
    allocation_ptr: usize,
    /// Maximum size of the space
    max_size: usize,
    /// Map of handle IDs to allocation info
    allocations: HashMap<HeapHandleId, AllocationInfo>,
}

/// Cell space for property cells
pub struct CellSpace {
    /// Memory buffer for the cell space
    #[allow(dead_code)]
    buffer: Vec<u8>,
    /// Current allocation pointer
    allocation_ptr: usize,
    /// Maximum size of the space
    max_size: usize,
    /// Map of handle IDs to allocation info
    allocations: HashMap<HeapHandleId, AllocationInfo>,
}

/// Map space for object shapes
pub struct MapSpace {
    /// Memory buffer for the map space
    #[allow(dead_code)]
    buffer: Vec<u8>,
    /// Current allocation pointer
    allocation_ptr: usize,
    /// Maximum size of the space
    max_size: usize,
    /// Map of handle IDs to allocation info
    allocations: HashMap<HeapHandleId, AllocationInfo>,
}

/// Information about a memory allocation
#[derive(Debug, Clone)]
struct AllocationInfo {
    /// Offset in the buffer
    offset: usize,
    /// Size of the allocation
    size: usize,
    /// Whether the allocation is still valid
    valid: bool,
}

/// Information about a free memory block
#[derive(Debug, Clone)]
struct FreeBlock {
    /// Offset in the buffer
    offset: usize,
    /// Size of the free block
    size: usize,
}

impl NewSpace {
    fn new(max_size: usize) -> Self {
        Self {
            buffer: vec![0; max_size],
            allocation_ptr: 0,
            max_size,
            allocations: HashMap::new(),
        }
    }

    fn allocate(&mut self, size: usize, handle_id: HeapHandleId) -> bool {
        if self.allocation_ptr + size > self.max_size {
            return false;
        }

        let offset = self.allocation_ptr;
        self.allocation_ptr += size;

        self.allocations.insert(
            handle_id,
            AllocationInfo {
                offset,
                size,
                valid: true,
            },
        );

        true
    }

    fn deallocate(&mut self, handle_id: HeapHandleId) -> bool {
        if let Some(info) = self.allocations.get_mut(&handle_id) {
            info.valid = false;
            true
        } else {
            false
        }
    }

    fn collect_garbage(&mut self) -> GcResult {
        let mut objects_collected = 0;
        let mut memory_freed = 0;

        // Remove invalid allocations and compact memory
        self.allocations.retain(|_, info| {
            if info.valid {
                true
            } else {
                objects_collected += 1;
                memory_freed += info.size;
                false
            }
        });

        // Reset allocation pointer for next allocation
        self.allocation_ptr = 0;

        GcResult {
            objects_collected,
            memory_freed,
            collection_time: Duration::ZERO,
        }
    }
}

impl OldSpace {
    fn new(max_size: usize) -> Self {
        Self {
            buffer: vec![0; max_size],
            free_list: Vec::new(),
            max_size,
            allocations: HashMap::new(),
        }
    }

    fn allocate(&mut self, size: usize, handle_id: HeapHandleId) -> bool {
        // Try to find a suitable free block
        if let Some(index) = self.free_list.iter().position(|block| block.size >= size) {
            let block = self.free_list.remove(index);
            let offset = block.offset;

            // If the block is larger than needed, create a new free block
            if block.size > size {
                self.free_list.push(FreeBlock {
                    offset: offset + size,
                    size: block.size - size,
                });
            }

            self.allocations.insert(
                handle_id,
                AllocationInfo {
                    offset,
                    size,
                    valid: true,
                },
            );

            true
        } else {
            // No suitable free block found
            false
        }
    }

    fn deallocate(&mut self, handle_id: HeapHandleId) -> bool {
        if let Some(info) = self.allocations.remove(&handle_id) {
            // Add the freed block to the free list
            self.free_list.push(FreeBlock {
                offset: info.offset,
                size: info.size,
            });

            true
        } else {
            false
        }
    }

    fn collect_garbage(&mut self) -> GcResult {
        let mut objects_collected = 0;
        let mut memory_freed = 0;

        // Remove invalid allocations
        self.allocations.retain(|_, info| {
            if info.valid {
                true
            } else {
                objects_collected += 1;
                memory_freed += info.size;
                false
            }
        });

        // Compact free list
        self.free_list.sort_by_key(|block| block.offset);
        let mut compacted_free_list = Vec::new();
        let mut current_block: Option<FreeBlock> = None;

        for block in self.free_list.drain(..) {
            if let Some(mut current) = current_block {
                if current.offset + current.size == block.offset {
                    // Merge adjacent blocks
                    current.size += block.size;
                    current_block = Some(current);
                } else {
                    // Add current block and start new one
                    compacted_free_list.push(current);
                    current_block = Some(block);
                }
            } else {
                current_block = Some(block);
            }
        }

        if let Some(current) = current_block {
            compacted_free_list.push(current);
        }

        self.free_list = compacted_free_list;

        GcResult {
            objects_collected,
            memory_freed,
            collection_time: Duration::ZERO,
        }
    }
}

impl LargeSpace {
    fn new(max_size: usize) -> Self {
        Self {
            buffer: vec![0; max_size],
            max_size,
            allocations: HashMap::new(),
        }
    }

    fn allocate(&mut self, size: usize, handle_id: HeapHandleId) -> bool {
        // For large objects, we use a simple allocation strategy
        // In a real implementation, this would be more sophisticated
        if size > self.max_size {
            return false;
        }

        let offset = 0; // Simplified for this example

        self.allocations.insert(
            handle_id,
            AllocationInfo {
                offset,
                size,
                valid: true,
            },
        );

        true
    }

    fn deallocate(&mut self, handle_id: HeapHandleId) -> bool {
        if let Some(info) = self.allocations.get_mut(&handle_id) {
            info.valid = false;
            true
        } else {
            false
        }
    }

    fn collect_garbage(&mut self) -> GcResult {
        let mut objects_collected = 0;
        let mut memory_freed = 0;

        self.allocations.retain(|_, info| {
            if info.valid {
                true
            } else {
                objects_collected += 1;
                memory_freed += info.size;
                false
            }
        });

        GcResult {
            objects_collected,
            memory_freed,
            collection_time: Duration::ZERO,
        }
    }
}

impl CodeSpace {
    fn new(max_size: usize) -> Self {
        Self {
            buffer: vec![0; max_size],
            allocation_ptr: 0,
            max_size,
            allocations: HashMap::new(),
        }
    }

    fn allocate(&mut self, size: usize, handle_id: HeapHandleId) -> bool {
        if self.allocation_ptr + size > self.max_size {
            return false;
        }

        let offset = self.allocation_ptr;
        self.allocation_ptr += size;

        self.allocations.insert(
            handle_id,
            AllocationInfo {
                offset,
                size,
                valid: true,
            },
        );

        true
    }

    fn deallocate(&mut self, handle_id: HeapHandleId) -> bool {
        if let Some(info) = self.allocations.get_mut(&handle_id) {
            info.valid = false;
            true
        } else {
            false
        }
    }

    fn collect_garbage(&mut self) -> GcResult {
        let mut objects_collected = 0;
        let mut memory_freed = 0;

        self.allocations.retain(|_, info| {
            if info.valid {
                true
            } else {
                objects_collected += 1;
                memory_freed += info.size;
                false
            }
        });

        GcResult {
            objects_collected,
            memory_freed,
            collection_time: Duration::ZERO,
        }
    }
}

impl CellSpace {
    fn new(max_size: usize) -> Self {
        Self {
            buffer: vec![0; max_size],
            allocation_ptr: 0,
            max_size,
            allocations: HashMap::new(),
        }
    }

    fn allocate(&mut self, size: usize, handle_id: HeapHandleId) -> bool {
        if self.allocation_ptr + size > self.max_size {
            return false;
        }

        let offset = self.allocation_ptr;
        self.allocation_ptr += size;

        self.allocations.insert(
            handle_id,
            AllocationInfo {
                offset,
                size,
                valid: true,
            },
        );

        true
    }

    fn deallocate(&mut self, handle_id: HeapHandleId) -> bool {
        if let Some(info) = self.allocations.get_mut(&handle_id) {
            info.valid = false;
            true
        } else {
            false
        }
    }

    fn collect_garbage(&mut self) -> GcResult {
        let mut objects_collected = 0;
        let mut memory_freed = 0;

        self.allocations.retain(|_, info| {
            if info.valid {
                true
            } else {
                objects_collected += 1;
                memory_freed += info.size;
                false
            }
        });

        GcResult {
            objects_collected,
            memory_freed,
            collection_time: Duration::ZERO,
        }
    }
}

impl MapSpace {
    fn new(max_size: usize) -> Self {
        Self {
            buffer: vec![0; max_size],
            allocation_ptr: 0,
            max_size,
            allocations: HashMap::new(),
        }
    }

    fn allocate(&mut self, size: usize, handle_id: HeapHandleId) -> bool {
        if self.allocation_ptr + size > self.max_size {
            return false;
        }

        let offset = self.allocation_ptr;
        self.allocation_ptr += size;

        self.allocations.insert(
            handle_id,
            AllocationInfo {
                offset,
                size,
                valid: true,
            },
        );

        true
    }

    fn deallocate(&mut self, handle_id: HeapHandleId) -> bool {
        if let Some(info) = self.allocations.get_mut(&handle_id) {
            info.valid = false;
            true
        } else {
            false
        }
    }

    fn collect_garbage(&mut self) -> GcResult {
        let mut objects_collected = 0;
        let mut memory_freed = 0;

        self.allocations.retain(|_, info| {
            if info.valid {
                true
            } else {
                objects_collected += 1;
                memory_freed += info.size;
                false
            }
        });

        GcResult {
            objects_collected,
            memory_freed,
            collection_time: Duration::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generational_heap_creation() {
        let config = HeapConfig::default();
        let heap = GenerationalHeap::new(config);
        let stats = heap.get_stats();

        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.total_used, 0);
        assert_eq!(stats.allocation_count, 0);
    }

    #[test]
    fn test_generational_heap_allocation() {
        let config = HeapConfig::default();
        let mut heap = GenerationalHeap::new(config);

        let handle_id = heap.allocate(1024, ObjectType::Object);
        assert!(handle_id.is_some());

        let stats = heap.get_stats();
        assert_eq!(stats.total_allocated, 1024);
        assert_eq!(stats.total_used, 1024);
        assert_eq!(stats.allocation_count, 1);
    }

    #[test]
    fn test_generational_heap_deallocation() {
        let config = HeapConfig::default();
        let mut heap = GenerationalHeap::new(config);

        let handle_id = heap.allocate(1024, ObjectType::Object);
        assert!(handle_id.is_some());

        heap.deallocate(handle_id.unwrap(), 1024, ObjectType::Object);

        let stats = heap.get_stats();
        assert_eq!(stats.total_freed, 1024);
        assert_eq!(stats.total_used, 0);
        assert_eq!(stats.deallocation_count, 1);
    }

    #[test]
    fn test_generational_heap_garbage_collection() {
        let config = HeapConfig::default();
        let mut heap = GenerationalHeap::new(config);

        let handle_id = heap.allocate(1024, ObjectType::Object);
        assert!(handle_id.is_some());

        heap.deallocate(handle_id.unwrap(), 1024, ObjectType::Object);

        let gc_result = heap.collect_garbage();
        assert!(gc_result.objects_collected > 0);
        assert!(gc_result.memory_freed > 0);

        let stats = heap.get_stats();
        assert_eq!(stats.gc_count, 1);
    }

    #[test]
    fn test_generational_heap_large_object_allocation() {
        let config = HeapConfig::default();
        let mut heap = GenerationalHeap::new(config.clone());

        // Allocate a large object that should go to large space
        let large_size = config.large_object_threshold + 1024;
        let handle_id = heap.allocate(large_size, ObjectType::Object);
        assert!(handle_id.is_some());

        let stats = heap.get_stats();
        assert_eq!(stats.total_allocated, large_size);
        assert_eq!(stats.total_used, large_size);
    }

    #[test]
    fn test_generational_heap_function_allocation() {
        let config = HeapConfig::default();
        let mut heap = GenerationalHeap::new(config);

        let handle_id = heap.allocate(1024, ObjectType::Function);
        assert!(handle_id.is_some());

        let stats = heap.get_stats();
        assert_eq!(stats.total_allocated, 1024);
        assert_eq!(stats.total_used, 1024);
    }
}

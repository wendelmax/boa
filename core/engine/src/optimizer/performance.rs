//! # Performance Monitoring System for Boa Engine
//!
//! This module provides comprehensive performance monitoring and metrics collection
//! for VM execution, inspired by JetCrab's performance monitoring approach.
//!
//! ## Key Features
//!
//! - **Real-time Metrics Collection**: Tracks execution statistics during runtime
//! - **Instruction-level Profiling**: Monitors individual instruction performance
//! - **Memory Usage Tracking**: Tracks heap allocations and garbage collection
//! - **Performance Reports**: Generates detailed performance analysis
//!
//! ## Usage
//!
//! ### Basic Performance Monitoring
//!
//! ```rust
//! use boa_engine::optimizer::performance::{PerformanceMonitor, PerformanceMetrics};
//! use std::time::Duration;
//!
//! let mut monitor = PerformanceMonitor::new();
//! monitor.start_execution();
//!
//! // Execute some code...
//! monitor.record_instruction("Add");
//! monitor.record_instruction("Mul");
//! monitor.record_memory_allocation(1024, "Object");
//! monitor.record_memory_allocation(1024, "Array");
//!
//! monitor.end_execution();
//! let metrics = monitor.get_metrics();
//! println!("Executed {} instructions", metrics.total_instructions);
//! println!("Performance: {:.2} instructions/sec", metrics.instructions_per_second());
//! ```
//!
//! ### Performance Analysis and Reporting
//!
//! ```rust
//! use boa_engine::optimizer::performance::{PerformanceMonitor, PerformanceReport};
//!
//! let mut monitor = PerformanceMonitor::new();
//! monitor.start_execution();
//!
//! // Simulate execution
//! for i in 0..1000 {
//!     monitor.record_instruction("Add");
//!     if i % 100 == 0 {
//!         monitor.record_memory_allocation(1024, "Buffer");
//!     }
//! }
//!
//! monitor.end_execution();
//!
//! let metrics = monitor.get_metrics();
//! let report = PerformanceReport::generate(&metrics);
//!
//! println!("Performance Score: {:.2}", report.analysis.performance_score);
//! for recommendation in &report.recommendations {
//!     println!("Recommendation: {}", recommendation);
//! }
//! ```
//!
//! ### Configuration
//!
//! ```rust
//! use boa_engine::optimizer::performance::{PerformanceMonitor, PerformanceConfig};
//!
//! let config = PerformanceConfig {
//!     enable_instruction_profiling: true,
//!     enable_memory_profiling: true,
//!     enable_gc_profiling: true,
//!     sample_rate: 0.5, // Sample 50% of instructions
//!     max_instruction_tracking: 5000,
//! };
//!
//! let mut monitor = PerformanceMonitor::new();
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance metrics collected during execution
///
/// This struct contains all the performance data collected during VM execution,
/// including instruction counts, timing information, memory usage, and garbage
/// collection statistics.
///
/// # Examples
///
/// ```rust
/// use boa_engine::optimizer::performance::PerformanceMetrics;
/// use std::time::Duration;
///
/// let mut metrics = PerformanceMetrics::new();
/// metrics.total_instructions = 1000;
/// metrics.execution_time = Duration::from_secs(1);
/// metrics.total_memory_allocated = 1024 * 1024; // 1MB
///
/// println!("Performance: {:.2} instructions/sec", metrics.instructions_per_second());
/// println!("Memory rate: {:.2} bytes/sec", metrics.memory_allocation_rate());
/// ```
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total number of instructions executed
    pub total_instructions: usize,
    /// Total execution time
    pub execution_time: Duration,
    /// Number of memory allocations
    pub memory_allocations: usize,
    /// Total memory allocated (in bytes)
    pub total_memory_allocated: usize,
    /// Number of garbage collections
    pub garbage_collections: usize,
    /// Time spent in garbage collection
    pub gc_time: Duration,
    /// Number of stack operations
    pub stack_operations: usize,
    /// Number of heap operations
    pub heap_operations: usize,
    /// Instruction execution counts by type
    pub instruction_counts: HashMap<String, usize>,
    /// Instruction execution times by type
    pub instruction_times: HashMap<String, Duration>,
    /// Memory allocation patterns
    pub allocation_patterns: HashMap<String, usize>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self {
            total_instructions: 0,
            execution_time: Duration::ZERO,
            memory_allocations: 0,
            total_memory_allocated: 0,
            garbage_collections: 0,
            gc_time: Duration::ZERO,
            stack_operations: 0,
            heap_operations: 0,
            instruction_counts: HashMap::new(),
            instruction_times: HashMap::new(),
            allocation_patterns: HashMap::new(),
        }
    }

    /// Get instructions per second
    pub fn instructions_per_second(&self) -> f64 {
        if self.execution_time.as_secs_f64() > 0.0 {
            self.total_instructions as f64 / self.execution_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get memory allocation rate (bytes per second)
    pub fn memory_allocation_rate(&self) -> f64 {
        if self.execution_time.as_secs_f64() > 0.0 {
            self.total_memory_allocated as f64 / self.execution_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get average instruction execution time
    pub fn average_instruction_time(&self) -> Duration {
        if self.total_instructions > 0 {
            self.execution_time / self.total_instructions as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get the most frequently executed instruction
    pub fn most_frequent_instruction(&self) -> Option<(&String, usize)> {
        self.instruction_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(name, count)| (name, *count))
    }

    /// Get the slowest instruction
    pub fn slowest_instruction(&self) -> Option<(&String, Duration)> {
        self.instruction_times
            .iter()
            .max_by_key(|(_, duration)| *duration)
            .map(|(name, duration)| (name, *duration))
    }
}

/// Performance monitor for tracking execution metrics
///
/// The `PerformanceMonitor` is the primary interface for collecting performance
/// metrics during JavaScript execution. It tracks instructions, memory allocations,
/// garbage collection events, and provides detailed performance analysis.
///
/// # Examples
///
/// ```rust
/// use boa_engine::optimizer::performance::PerformanceMonitor;
/// use std::time::Duration;
///
/// let mut monitor = PerformanceMonitor::new();
///
/// // Start monitoring
/// monitor.start_execution();
///
/// // Record execution events
/// monitor.record_instruction("Add");
/// monitor.record_instruction("Mul");
/// monitor.record_memory_allocation(1024, "Array");
/// monitor.record_memory_allocation(512, "String");
///
/// // End monitoring and get results
/// monitor.end_execution();
/// let metrics = monitor.get_metrics();
///
/// println!("Total instructions: {}", metrics.total_instructions);
/// println!("Execution time: {:?}", metrics.execution_time);
/// ```
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Current metrics being collected
    metrics: PerformanceMetrics,
    /// Start time of current execution
    start_time: Option<Instant>,
    /// Current instruction start time
    instruction_start: Option<Instant>,
    /// Whether monitoring is currently active
    active: bool,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            metrics: PerformanceMetrics::new(),
            start_time: None,
            instruction_start: None,
            active: false,
        }
    }

    /// Start monitoring execution
    pub fn start_execution(&mut self) {
        self.start_time = Some(Instant::now());
        self.active = true;
    }

    /// End monitoring execution
    pub fn end_execution(&mut self) {
        if let Some(start) = self.start_time {
            self.metrics.execution_time = start.elapsed();
        }
        self.active = false;
        self.start_time = None;
    }

    /// Record the execution of an instruction
    pub fn record_instruction(&mut self, instruction_name: &str) {
        if !self.active {
            return;
        }

        self.metrics.total_instructions += 1;

        // Update instruction count
        *self
            .metrics
            .instruction_counts
            .entry(instruction_name.to_string())
            .or_insert(0) += 1;

        // Record instruction timing if we have a start time
        if let Some(start) = self.instruction_start {
            let duration = start.elapsed();
            *self
                .metrics
                .instruction_times
                .entry(instruction_name.to_string())
                .or_insert(Duration::ZERO) += duration;
        }

        // Start timing the next instruction
        self.instruction_start = Some(Instant::now());
    }

    /// Record a memory allocation
    pub fn record_memory_allocation(&mut self, size: usize, allocation_type: &str) {
        if !self.active {
            return;
        }

        self.metrics.memory_allocations += 1;
        self.metrics.total_memory_allocated += size;

        *self
            .metrics
            .allocation_patterns
            .entry(allocation_type.to_string())
            .or_insert(0) += 1;
    }

    /// Record a garbage collection
    pub fn record_garbage_collection(&mut self, duration: Duration) {
        if !self.active {
            return;
        }

        self.metrics.garbage_collections += 1;
        self.metrics.gc_time += duration;
    }

    /// Record a stack operation
    pub fn record_stack_operation(&mut self) {
        if !self.active {
            return;
        }

        self.metrics.stack_operations += 1;
    }

    /// Record a heap operation
    pub fn record_heap_operation(&mut self) {
        if !self.active {
            return;
        }

        self.metrics.heap_operations += 1;
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    /// Get a copy of current performance metrics
    pub fn get_metrics_copy(&self) -> PerformanceMetrics {
        self.metrics.clone()
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.metrics = PerformanceMetrics::new();
        self.start_time = None;
        self.instruction_start = None;
        self.active = false;
    }

    /// Check if monitoring is active
    pub fn is_active(&self) -> bool {
        self.active
    }
}

/// Performance report with analysis and recommendations
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Summary metrics
    pub summary: PerformanceMetrics,
    /// Performance analysis
    pub analysis: PerformanceAnalysis,
    /// Recommendations for optimization
    pub recommendations: Vec<String>,
    /// Generated timestamp
    pub generated_at: Instant,
}

/// Performance analysis results
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// Performance score (0-100)
    pub performance_score: f64,
    /// Bottlenecks identified
    pub bottlenecks: Vec<String>,
    /// Memory efficiency score (0-100)
    pub memory_efficiency: f64,
    /// Instruction efficiency score (0-100)
    pub instruction_efficiency: f64,
}

impl PerformanceAnalysis {
    /// Create a new performance analysis with default values
    pub fn new() -> Self {
        Self {
            performance_score: 0.0,
            memory_efficiency: 0.0,
            instruction_efficiency: 0.0,
            bottlenecks: Vec::new(),
        }
    }
}

impl Default for PerformanceAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceReport {
    /// Generate a performance report from metrics
    pub fn generate(metrics: &PerformanceMetrics) -> Self {
        let analysis = Self::analyze_performance(metrics);
        let recommendations = Self::generate_recommendations(&analysis, metrics);

        Self {
            summary: metrics.clone(),
            analysis,
            recommendations,
            generated_at: Instant::now(),
        }
    }

    /// Analyze performance metrics
    fn analyze_performance(metrics: &PerformanceMetrics) -> PerformanceAnalysis {
        let mut bottlenecks = Vec::new();
        let mut performance_score: f64 = 100.0;
        let mut memory_efficiency: f64 = 100.0;
        let mut instruction_efficiency: f64 = 100.0;

        // Analyze instruction efficiency
        let avg_instruction_time = metrics.average_instruction_time();
        if avg_instruction_time > Duration::from_micros(10) {
            bottlenecks.push("Slow instruction execution".to_string());
            instruction_efficiency -= 20.0;
        }

        // Analyze memory efficiency
        let allocation_rate = metrics.memory_allocation_rate();
        if allocation_rate > 1_000_000.0 {
            // 1MB/s
            bottlenecks.push("High memory allocation rate".to_string());
            memory_efficiency -= 30.0;
        }

        // Analyze GC efficiency
        let gc_ratio = if metrics.execution_time.as_secs_f64() > 0.0 {
            metrics.gc_time.as_secs_f64() / metrics.execution_time.as_secs_f64()
        } else {
            0.0
        };

        if gc_ratio > 0.1 {
            // More than 10% time in GC
            bottlenecks.push("Excessive garbage collection time".to_string());
            performance_score -= 25.0;
        }

        // Analyze instruction distribution
        if let Some((_, count)) = metrics.most_frequent_instruction() {
            let frequency = count as f64 / metrics.total_instructions as f64;
            if frequency > 0.5 {
                // More than 50% of instructions are the same type
                bottlenecks.push("Instruction distribution imbalance".to_string());
                instruction_efficiency -= 15.0;
            }
        }

        PerformanceAnalysis {
            performance_score: performance_score.max(0.0),
            bottlenecks,
            memory_efficiency: memory_efficiency.max(0.0),
            instruction_efficiency: instruction_efficiency.max(0.0),
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        analysis: &PerformanceAnalysis,
        metrics: &PerformanceMetrics,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if analysis.performance_score < 70.0 {
            recommendations.push("Consider optimizing overall execution performance".to_string());
        }

        if analysis.memory_efficiency < 70.0 {
            recommendations.push("Optimize memory allocation patterns".to_string());
            recommendations.push("Consider implementing object pooling".to_string());
        }

        if analysis.instruction_efficiency < 70.0 {
            recommendations.push("Optimize instruction execution".to_string());
            recommendations.push("Consider implementing instruction caching".to_string());
        }

        if !analysis.bottlenecks.is_empty() {
            recommendations.push("Address identified performance bottlenecks".to_string());
        }

        if metrics.garbage_collections > metrics.total_instructions / 100 {
            recommendations.push("Optimize garbage collection frequency".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Performance is within acceptable ranges".to_string());
        }

        recommendations
    }
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Copy)]
pub struct PerformanceConfig {
    /// Enable instruction-level profiling
    pub enable_instruction_profiling: bool,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable GC profiling
    pub enable_gc_profiling: bool,
    /// Sample rate for profiling (0.0 to 1.0)
    pub sample_rate: f64,
    /// Maximum number of instructions to track
    pub max_instruction_tracking: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_instruction_profiling: true,
            enable_memory_profiling: true,
            enable_gc_profiling: true,
            sample_rate: 1.0,
            max_instruction_tracking: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor_basic() {
        let mut monitor = PerformanceMonitor::new();

        monitor.start_execution();
        monitor.record_instruction("Add");
        monitor.record_instruction("Sub");
        monitor.record_memory_allocation(1024, "Object");
        monitor.end_execution();

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_instructions, 2);
        assert_eq!(metrics.memory_allocations, 1);
        assert_eq!(metrics.total_memory_allocated, 1024);
        assert!(metrics.execution_time > Duration::ZERO);
    }

    #[test]
    fn test_performance_metrics_calculations() {
        let mut metrics = PerformanceMetrics::new();
        metrics.total_instructions = 1000;
        metrics.execution_time = Duration::from_secs(1);
        metrics.total_memory_allocated = 1024 * 1024; // 1MB

        assert_eq!(metrics.instructions_per_second(), 1000.0);
        assert_eq!(metrics.memory_allocation_rate(), 1024.0 * 1024.0);
        assert_eq!(metrics.average_instruction_time(), Duration::from_millis(1));
    }

    #[test]
    fn test_performance_report_generation() {
        let mut metrics = PerformanceMetrics::new();
        metrics.total_instructions = 1000;
        metrics.execution_time = Duration::from_secs(1);
        metrics.gc_time = Duration::from_millis(100);
        metrics.instruction_counts.insert("Add".to_string(), 500);
        metrics.instruction_counts.insert("Sub".to_string(), 500);

        let report = PerformanceReport::generate(&metrics);
        assert!(report.analysis.performance_score > 0.0);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_performance_monitor_reset() {
        let mut monitor = PerformanceMonitor::new();

        monitor.start_execution();
        monitor.record_instruction("Add");
        monitor.end_execution();

        let metrics_before = monitor.get_metrics_copy();
        assert_eq!(metrics_before.total_instructions, 1);

        monitor.reset();
        let metrics_after = monitor.get_metrics();
        assert_eq!(metrics_after.total_instructions, 0);
        assert!(!monitor.is_active());
    }

    #[test]
    fn test_performance_monitor_edge_cases() {
        let mut monitor = PerformanceMonitor::new();

        // Test recording without starting execution
        monitor.record_instruction("Add");
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_instructions, 0);

        // Test multiple start/end cycles
        monitor.start_execution();
        monitor.record_instruction("Add");
        monitor.end_execution();

        monitor.start_execution();
        monitor.record_instruction("Sub");
        monitor.end_execution();

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_instructions, 2);
    }

    #[test]
    fn test_performance_metrics_edge_cases() {
        let mut metrics = PerformanceMetrics::new();

        // Test with zero values
        assert_eq!(metrics.instructions_per_second(), 0.0);
        assert_eq!(metrics.memory_allocation_rate(), 0.0);
        assert_eq!(metrics.average_instruction_time(), Duration::ZERO);

        // Test with very small values
        metrics.total_instructions = 1;
        metrics.execution_time = Duration::from_nanos(1);
        assert!(metrics.instructions_per_second() > 0.0);
    }

    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();
        assert!(config.enable_instruction_profiling);
        assert!(config.enable_memory_profiling);
        assert!(config.enable_gc_profiling);
        assert_eq!(config.sample_rate, 1.0);
        assert_eq!(config.max_instruction_tracking, 1000);
    }

    #[test]
    fn test_performance_analysis_calculations() {
        let mut analysis = PerformanceAnalysis::new();
        analysis.performance_score = 85.0;
        analysis.memory_efficiency = 90.0;
        analysis.instruction_efficiency = 80.0;
        analysis.bottlenecks.push("Slow execution".to_string());

        assert_eq!(analysis.performance_score, 85.0);
        assert_eq!(analysis.memory_efficiency, 90.0);
        assert_eq!(analysis.instruction_efficiency, 80.0);
        assert!(!analysis.bottlenecks.is_empty());
    }

    #[test]
    fn test_performance_monitor_concurrent_access() {
        use std::thread;

        let monitor = std::sync::Arc::new(std::sync::Mutex::new(PerformanceMonitor::new()));
        let mut handles = vec![];

        for i in 0..5 {
            let monitor_clone = monitor.clone();
            let handle = thread::spawn(move || {
                let mut monitor = monitor_clone.lock().unwrap();
                monitor.start_execution();
                monitor.record_instruction(&format!("Add_{}", i));
                monitor.end_execution();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let monitor = monitor.lock().unwrap();
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_instructions, 5);
    }
}

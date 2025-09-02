//! # Error Handling System for Boa VM
//!
//! This module implements a comprehensive error handling system inspired by JetCrab's approach.
//! It provides unified error management, automatic recovery, and context preservation.
//!
//! ## Key Features
//!
//! - **Unified Error System**: Consistent error handling across all VM components
//! - **Automatic Recovery**: Graceful handling of recoverable errors
//! - **Context Preservation**: Maintain error context for better debugging
//! - **Error Classification**: Categorize errors by severity and type
//! - **Recovery Strategies**: Multiple strategies for different error types
//! - **Error Reporting**: Detailed error information and stack traces
//!
//! ## Error Categories
//!
//! - **Fatal Errors**: Cannot be recovered, require VM termination
//! - **Recoverable Errors**: Can be handled with recovery strategies
//! - **Warning Errors**: Non-fatal issues that should be reported
//! - **Info Errors**: Informational messages for debugging

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Severity level of an error
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Fatal error that cannot be recovered
    Fatal,
    /// Recoverable error that can be handled
    Recoverable,
    /// Warning that should be reported but doesn't stop execution
    Warning,
    /// Informational message for debugging
    Info,
}

/// Category of an error
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ErrorCategory {
    /// Memory-related errors
    Memory,
    /// Type-related errors
    Type,
    /// Runtime errors
    Runtime,
    /// Syntax errors
    Syntax,
    /// Logic errors
    Logic,
    /// System errors
    System,
}

/// A unique identifier for an error
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ErrorId(usize);

impl ErrorId {
    /// Creates a new unique error ID
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

impl Default for ErrorId {
    fn default() -> Self {
        Self::new()
    }
}

/// Error context information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Source code location
    pub source_location: Option<SourceLocation>,
    /// Call stack at the time of error
    pub call_stack: Vec<CallFrame>,
    /// Variable values at the time of error
    pub variable_values: HashMap<String, String>,
    /// Memory state information
    pub memory_info: Option<MemoryInfo>,
    /// Execution state
    pub execution_state: Option<ExecutionState>,
}

/// Source code location information
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// File name
    pub file: String,
    /// Line number
    pub line: usize,
    /// Column number
    pub column: usize,
    /// Source code snippet
    pub snippet: String,
}

/// Call frame information
#[derive(Debug, Clone)]
pub struct CallFrame {
    /// Function name
    pub function_name: String,
    /// Source location
    pub location: SourceLocation,
    /// Local variables
    pub local_variables: HashMap<String, String>,
}

/// Memory state information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total memory allocated
    pub total_allocated: usize,
    /// Memory in use
    pub memory_in_use: usize,
    /// Number of active handles
    pub active_handles: usize,
    /// Garbage collection statistics
    pub gc_stats: String,
}

/// Execution state information
#[derive(Debug, Clone)]
pub struct ExecutionState {
    /// Current instruction pointer
    pub instruction_pointer: usize,
    /// Stack depth
    pub stack_depth: usize,
    /// Number of instructions executed
    pub instructions_executed: usize,
    /// Execution time
    pub execution_time: Duration,
}

/// A comprehensive error with full context
#[derive(Debug, Clone)]
pub struct Error {
    /// Unique error identifier
    pub id: ErrorId,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Error category
    pub category: ErrorCategory,
    /// Error message
    pub message: String,
    /// Detailed description
    pub description: String,
    /// Error context
    pub context: ErrorContext,
    /// Timestamp when the error occurred
    pub timestamp: Instant,
    /// Whether this error has been handled
    pub handled: bool,
    /// Recovery strategy used
    pub recovery_strategy: Option<RecoveryStrategy>,
}

/// Recovery strategy for handling errors
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry the operation
    Retry {
        max_attempts: usize,
        delay: Duration,
    },
    /// Use a fallback value
    Fallback { value: String },
    /// Skip the operation
    Skip,
    /// Restart the component
    Restart,
    /// Graceful degradation
    Degrade,
}

/// Result of error recovery
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Whether recovery was successful
    pub success: bool,
    /// Recovery strategy used
    pub strategy: RecoveryStrategy,
    /// Time taken for recovery
    pub recovery_time: Duration,
    /// Additional information
    pub info: String,
}

/// Error handler trait
pub trait ErrorHandler {
    /// Handle an error
    fn handle_error(&mut self, error: &mut Error) -> RecoveryResult;

    /// Check if this handler can handle the given error
    fn can_handle(&self, error: &Error) -> bool;

    /// Get the priority of this handler (higher = more important)
    fn priority(&self) -> usize;
}

/// Default error handler
pub struct DefaultErrorHandler;

impl ErrorHandler for DefaultErrorHandler {
    fn handle_error(&mut self, error: &mut Error) -> RecoveryResult {
        match error.severity {
            ErrorSeverity::Fatal => RecoveryResult {
                success: false,
                strategy: RecoveryStrategy::Restart,
                recovery_time: Duration::ZERO,
                info: "Fatal error - cannot recover".to_string(),
            },
            ErrorSeverity::Recoverable => RecoveryResult {
                success: true,
                strategy: RecoveryStrategy::Retry {
                    max_attempts: 3,
                    delay: Duration::from_millis(100),
                },
                recovery_time: Duration::from_millis(50),
                info: "Retrying operation".to_string(),
            },
            ErrorSeverity::Warning => RecoveryResult {
                success: true,
                strategy: RecoveryStrategy::Skip,
                recovery_time: Duration::ZERO,
                info: "Warning ignored".to_string(),
            },
            ErrorSeverity::Info => RecoveryResult {
                success: true,
                strategy: RecoveryStrategy::Skip,
                recovery_time: Duration::ZERO,
                info: "Info logged".to_string(),
            },
        }
    }

    fn can_handle(&self, _error: &Error) -> bool {
        true // Default handler can handle all errors
    }

    fn priority(&self) -> usize {
        0 // Lowest priority
    }
}

/// Memory error handler
pub struct MemoryErrorHandler;

impl ErrorHandler for MemoryErrorHandler {
    fn handle_error(&mut self, error: &mut Error) -> RecoveryResult {
        if error.category != ErrorCategory::Memory {
            return RecoveryResult {
                success: false,
                strategy: RecoveryStrategy::Skip,
                recovery_time: Duration::ZERO,
                info: "Not a memory error".to_string(),
            };
        }

        match error.severity {
            ErrorSeverity::Fatal => RecoveryResult {
                success: false,
                strategy: RecoveryStrategy::Restart,
                recovery_time: Duration::ZERO,
                info: "Fatal memory error - restarting".to_string(),
            },
            ErrorSeverity::Recoverable => RecoveryResult {
                success: true,
                strategy: RecoveryStrategy::Retry {
                    max_attempts: 5,
                    delay: Duration::from_millis(200),
                },
                recovery_time: Duration::from_millis(100),
                info: "Retrying memory operation".to_string(),
            },
            _ => RecoveryResult {
                success: true,
                strategy: RecoveryStrategy::Skip,
                recovery_time: Duration::ZERO,
                info: "Memory warning/info logged".to_string(),
            },
        }
    }

    fn can_handle(&self, error: &Error) -> bool {
        error.category == ErrorCategory::Memory
    }

    fn priority(&self) -> usize {
        10 // Higher priority for memory errors
    }
}

/// Type error handler
pub struct TypeErrorHandler;

impl ErrorHandler for TypeErrorHandler {
    fn handle_error(&mut self, error: &mut Error) -> RecoveryResult {
        if error.category != ErrorCategory::Type {
            return RecoveryResult {
                success: false,
                strategy: RecoveryStrategy::Skip,
                recovery_time: Duration::ZERO,
                info: "Not a type error".to_string(),
            };
        }

        match error.severity {
            ErrorSeverity::Fatal => RecoveryResult {
                success: false,
                strategy: RecoveryStrategy::Restart,
                recovery_time: Duration::ZERO,
                info: "Fatal type error - restarting".to_string(),
            },
            ErrorSeverity::Recoverable => RecoveryResult {
                success: true,
                strategy: RecoveryStrategy::Fallback {
                    value: "undefined".to_string(),
                },
                recovery_time: Duration::from_millis(10),
                info: "Using fallback value".to_string(),
            },
            _ => RecoveryResult {
                success: true,
                strategy: RecoveryStrategy::Skip,
                recovery_time: Duration::ZERO,
                info: "Type warning/info logged".to_string(),
            },
        }
    }

    fn can_handle(&self, error: &Error) -> bool {
        error.category == ErrorCategory::Type
    }

    fn priority(&self) -> usize {
        8 // High priority for type errors
    }
}

/// Main error manager
pub struct ErrorManager {
    /// Registered error handlers
    handlers: Vec<Box<dyn ErrorHandler>>,
    /// Error history
    error_history: VecDeque<Error>,
    /// Maximum number of errors to keep in history
    max_history_size: usize,
    /// Statistics about error handling
    stats: ErrorStats,
}

/// Statistics about error handling
#[derive(Debug, Clone, Default)]
pub struct ErrorStats {
    /// Total number of errors handled
    pub total_errors: usize,
    /// Number of fatal errors
    pub fatal_errors: usize,
    /// Number of recoverable errors
    pub recoverable_errors: usize,
    /// Number of warnings
    pub warnings: usize,
    /// Number of info messages
    pub info_messages: usize,
    /// Number of successful recoveries
    pub successful_recoveries: usize,
    /// Number of failed recoveries
    pub failed_recoveries: usize,
    /// Average recovery time
    pub average_recovery_time: Duration,
}

impl ErrorManager {
    /// Create a new error manager
    pub fn new() -> Self {
        let mut manager = Self {
            handlers: Vec::new(),
            error_history: VecDeque::new(),
            max_history_size: 1000,
            stats: ErrorStats::default(),
        };

        // Add default handlers
        manager.add_handler(Box::new(MemoryErrorHandler));
        manager.add_handler(Box::new(TypeErrorHandler));
        manager.add_handler(Box::new(DefaultErrorHandler));

        manager
    }

    /// Add an error handler
    pub fn add_handler(&mut self, handler: Box<dyn ErrorHandler>) {
        self.handlers.push(handler);
        // Sort handlers by priority (highest first)
        self.handlers
            .sort_by(|a, b| b.priority().cmp(&a.priority()));
    }

    /// Handle an error
    pub fn handle_error(&mut self, mut error: Error) -> RecoveryResult {
        // Update statistics
        self.stats.total_errors += 1;
        match error.severity {
            ErrorSeverity::Fatal => self.stats.fatal_errors += 1,
            ErrorSeverity::Recoverable => self.stats.recoverable_errors += 1,
            ErrorSeverity::Warning => self.stats.warnings += 1,
            ErrorSeverity::Info => self.stats.info_messages += 1,
        }

        // Find the best handler for this error
        let mut best_handler = None;
        for (i, handler) in self.handlers.iter().enumerate() {
            if handler.can_handle(&error) {
                best_handler = Some(i);
                break;
            }
        }

        let recovery_result = if let Some(handler_index) = best_handler {
            // Handle the error
            let handler = &mut self.handlers[handler_index];
            let result = handler.handle_error(&mut error);

            // Update recovery statistics
            if result.success {
                self.stats.successful_recoveries += 1;
            } else {
                self.stats.failed_recoveries += 1;
            }

            // Update average recovery time
            let total_recoveries = self.stats.successful_recoveries + self.stats.failed_recoveries;
            if total_recoveries > 0 {
                self.stats.average_recovery_time = Duration::from_millis(
                    (self.stats.average_recovery_time.as_millis() as u64
                        * (total_recoveries - 1) as u64
                        + result.recovery_time.as_millis() as u64)
                        / total_recoveries as u64,
                );
            }

            result
        } else {
            // No handler found - use default recovery
            RecoveryResult {
                success: false,
                strategy: RecoveryStrategy::Restart,
                recovery_time: Duration::ZERO,
                info: "No handler found for error".to_string(),
            }
        };

        // Store the error in history
        error.handled = true;
        error.recovery_strategy = Some(recovery_result.strategy.clone());
        self.add_to_history(error);

        recovery_result
    }

    /// Add an error to the history
    fn add_to_history(&mut self, error: Error) {
        self.error_history.push_back(error);

        // Remove old errors if we exceed the maximum history size
        while self.error_history.len() > self.max_history_size {
            self.error_history.pop_front();
        }
    }

    /// Get error statistics
    pub fn get_stats(&self) -> &ErrorStats {
        &self.stats
    }

    /// Get error history
    pub fn get_error_history(&self) -> &VecDeque<Error> {
        &self.error_history
    }

    /// Clear error history
    pub fn clear_history(&mut self) {
        self.error_history.clear();
    }

    /// Set maximum history size
    pub fn set_max_history_size(&mut self, size: usize) {
        self.max_history_size = size;
    }

    /// Create a new error
    pub fn create_error(
        &self,
        severity: ErrorSeverity,
        category: ErrorCategory,
        message: String,
        description: String,
        context: ErrorContext,
    ) -> Error {
        Error {
            id: ErrorId::new(),
            severity,
            category,
            message,
            description,
            context,
            timestamp: Instant::now(),
            handled: false,
            recovery_strategy: None,
        }
    }
}

impl Default for ErrorManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}: {} - {}",
            self.severity, self.category, self.message, self.description
        )
    }
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Fatal => write!(f, "FATAL"),
            ErrorSeverity::Recoverable => write!(f, "RECOVERABLE"),
            ErrorSeverity::Warning => write!(f, "WARNING"),
            ErrorSeverity::Info => write!(f, "INFO"),
        }
    }
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::Memory => write!(f, "MEMORY"),
            ErrorCategory::Type => write!(f, "TYPE"),
            ErrorCategory::Runtime => write!(f, "RUNTIME"),
            ErrorCategory::Syntax => write!(f, "SYNTAX"),
            ErrorCategory::Logic => write!(f, "LOGIC"),
            ErrorCategory::System => write!(f, "SYSTEM"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_error_creation() {
        let manager = ErrorManager::new();
        let context = ErrorContext {
            source_location: None,
            call_stack: Vec::new(),
            variable_values: HashMap::new(),
            memory_info: None,
            execution_state: None,
        };

        let error = manager.create_error(
            ErrorSeverity::Recoverable,
            ErrorCategory::Memory,
            "Test error".to_string(),
            "This is a test error".to_string(),
            context,
        );

        assert_eq!(error.severity, ErrorSeverity::Recoverable);
        assert_eq!(error.category, ErrorCategory::Memory);
        assert_eq!(error.message, "Test error");
        assert!(!error.handled);
    }

    #[test]
    fn test_error_handling() {
        let mut manager = ErrorManager::new();
        let context = ErrorContext {
            source_location: None,
            call_stack: Vec::new(),
            variable_values: HashMap::new(),
            memory_info: None,
            execution_state: None,
        };

        let error = manager.create_error(
            ErrorSeverity::Recoverable,
            ErrorCategory::Memory,
            "Memory allocation failed".to_string(),
            "Could not allocate memory".to_string(),
            context,
        );

        let result = manager.handle_error(error);
        assert!(result.success);
        assert_eq!(manager.stats.total_errors, 1);
        assert_eq!(manager.stats.recoverable_errors, 1);
    }

    #[test]
    fn test_error_history() {
        let mut manager = ErrorManager::new();
        let context = ErrorContext {
            source_location: None,
            call_stack: Vec::new(),
            variable_values: HashMap::new(),
            memory_info: None,
            execution_state: None,
        };

        let error = manager.create_error(
            ErrorSeverity::Warning,
            ErrorCategory::Type,
            "Type warning".to_string(),
            "Type conversion warning".to_string(),
            context,
        );

        manager.handle_error(error);
        assert_eq!(manager.error_history.len(), 1);
        assert_eq!(manager.stats.warnings, 1);
    }

    #[test]
    fn test_error_handlers() {
        let mut manager = ErrorManager::new();

        // Test memory error handler
        let context = ErrorContext {
            source_location: None,
            call_stack: Vec::new(),
            variable_values: HashMap::new(),
            memory_info: None,
            execution_state: None,
        };

        let memory_error = manager.create_error(
            ErrorSeverity::Recoverable,
            ErrorCategory::Memory,
            "Memory error".to_string(),
            "Memory allocation failed".to_string(),
            context,
        );

        let result = manager.handle_error(memory_error);
        assert!(result.success);

        // Test type error handler
        let context = ErrorContext {
            source_location: None,
            call_stack: Vec::new(),
            variable_values: HashMap::new(),
            memory_info: None,
            execution_state: None,
        };

        let type_error = manager.create_error(
            ErrorSeverity::Recoverable,
            ErrorCategory::Type,
            "Type error".to_string(),
            "Type conversion failed".to_string(),
            context,
        );

        let result = manager.handle_error(type_error);
        assert!(result.success);
    }

    #[test]
    fn test_error_statistics() {
        let mut manager = ErrorManager::new();
        let context = ErrorContext {
            source_location: None,
            call_stack: Vec::new(),
            variable_values: HashMap::new(),
            memory_info: None,
            execution_state: None,
        };

        // Add different types of errors
        let errors = vec![
            (ErrorSeverity::Fatal, ErrorCategory::Memory),
            (ErrorSeverity::Recoverable, ErrorCategory::Type),
            (ErrorSeverity::Warning, ErrorCategory::Runtime),
            (ErrorSeverity::Info, ErrorCategory::System),
        ];

        for (severity, category) in errors {
            let error = manager.create_error(
                severity,
                category,
                "Test error".to_string(),
                "Test description".to_string(),
                context.clone(),
            );
            manager.handle_error(error);
        }

        let stats = manager.get_stats();
        assert_eq!(stats.total_errors, 4);
        assert_eq!(stats.fatal_errors, 1);
        assert_eq!(stats.recoverable_errors, 1);
        assert_eq!(stats.warnings, 1);
        assert_eq!(stats.info_messages, 1);
    }
}

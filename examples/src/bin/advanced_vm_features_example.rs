//! # Advanced VM Features Example
//!
//! This example demonstrates the advanced VM features integrated into the Boa engine,
//! including performance monitoring, bytecode optimization, and error handling.

use boa_engine::{
    Context, JsResult, Source,
    error::error_handling::{ErrorCategory, ErrorContext, ErrorManager, ErrorSeverity},
    optimizer::{
        bytecode_optimizer::{BytecodeOptimizer, Instruction},
        performance::PerformanceMonitor,
    },
};
use std::collections::HashMap;

fn main() -> JsResult<()> {
    println!("🚀 Advanced VM Features Example");
    println!("===============================\n");

    // Create a new context
    let mut context = Context::default();

    // Example 1: Performance Monitoring
    demonstrate_performance_monitoring(&mut context)?;

    // Example 2: Bytecode Optimization
    demonstrate_bytecode_optimization(&mut context)?;

    // Example 3: Error Handling
    demonstrate_error_handling(&mut context)?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Demonstrate performance monitoring capabilities
fn demonstrate_performance_monitoring(context: &mut Context) -> JsResult<()> {
    println!("📊 Performance Monitoring Example");
    println!("---------------------------------");

    // Create performance monitor
    let monitor = PerformanceMonitor::new();

    // Execute some JavaScript code
    let js_code = r#"
        function fibonacci(n) {
            if (n <= 1) return n;
            return fibonacci(n - 1) + fibonacci(n - 2);
        }
        
        let result = fibonacci(10);
        result;
    "#;

    let source = Source::from_bytes(js_code);
    let result = context.eval(source)?;

    // Get metrics
    let metrics = monitor.get_metrics();

    println!("Execution completed!");
    println!("Total instructions: {}", metrics.total_instructions);
    println!("Memory allocations: {}", metrics.memory_allocations);
    println!("Execution time: {:?}", metrics.execution_time);
    println!("Result: {:?}\n", result);

    Ok(())
}

/// Demonstrate bytecode optimization
fn demonstrate_bytecode_optimization(_context: &mut Context) -> JsResult<()> {
    println!("⚡ Bytecode Optimization Example");
    println!("--------------------------------");

    // Create bytecode optimizer
    let mut optimizer = BytecodeOptimizer::new();

    // Create some sample bytecode instructions
    let mut instructions = vec![
        Instruction::LoadConst { index: 0 },
        Instruction::LoadConst { index: 1 },
        Instruction::Add,
        Instruction::LoadConst { index: 0 }, // Duplicate - should be optimized
        Instruction::LoadConst { index: 1 }, // Duplicate - should be optimized
        Instruction::Add,                    // Duplicate - should be optimized
    ];

    println!("Original instructions: {}", instructions.len());

    // Optimize the bytecode
    let stats = optimizer.optimize(&mut instructions);

    println!("Optimized instructions: {}", instructions.len());
    println!("Total passes: {}", stats.total_passes);
    println!("Instructions removed: {}", stats.total_instructions_removed);
    println!("Instructions added: {}", stats.total_instructions_added);
    println!("Size reduction: {} bytes", stats.total_size_reduction);
    println!();

    Ok(())
}

/// Demonstrate error handling system
fn demonstrate_error_handling(_context: &mut Context) -> JsResult<()> {
    println!("🛡️ Error Handling Example");
    println!("-------------------------");

    // Create error manager
    let mut error_manager = ErrorManager::new();

    // Create a sample error context
    let error_context = ErrorContext {
        source_location: None,
        call_stack: Vec::new(),
        variable_values: HashMap::new(),
        memory_info: None,
        execution_state: None,
    };

    // Create and handle different types of errors
    let errors = vec![
        (
            ErrorSeverity::Warning,
            ErrorCategory::Type,
            "Type conversion warning",
        ),
        (
            ErrorSeverity::Recoverable,
            ErrorCategory::Memory,
            "Memory allocation failed",
        ),
        (
            ErrorSeverity::Info,
            ErrorCategory::Runtime,
            "Debug information",
        ),
    ];

    for (severity, category, message) in errors {
        let error = error_manager.create_error(
            severity,
            category,
            message.to_string(),
            format!("Detailed description for {}", message),
            error_context.clone(),
        );

        let result = error_manager.handle_error(error);
        println!("Error: {} - Recovery: {}", message, result.success);
    }

    // Get error statistics
    let stats = error_manager.get_stats();
    println!("\nError Statistics:");
    println!("Total errors: {}", stats.total_errors);
    println!("Warnings: {}", stats.warnings);
    println!("Recoverable: {}", stats.recoverable_errors);
    println!("Info messages: {}", stats.info_messages);
    println!("Successful recoveries: {}", stats.successful_recoveries);
    println!();

    Ok(())
}

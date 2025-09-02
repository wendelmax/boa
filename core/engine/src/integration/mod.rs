//! # Integration Module for JetCrab Improvements
//!
//! This module provides integration between the JetCrab-inspired improvements
//! and the existing Boa VM infrastructure. It serves as a bridge between
//! the new systems and the legacy Boa components.

pub mod benchmarking;
pub mod jit_compiler;
pub mod performance_integration;
pub mod vm_integration;

pub use benchmarking::BenchmarkingSuite;
pub use jit_compiler::JitCompiler;
pub use performance_integration::PerformanceIntegration;
pub use vm_integration::VmIntegration;


//! # Benchmarking Suite for Boa VM
//!
//! This module provides comprehensive benchmarking capabilities to validate
//! the performance improvements from JetCrab-inspired optimizations.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::{
    integration::vm_integration::{IntegrationConfig, IntegrationStats, VmIntegration},
    optimizer::performance::PerformanceMetrics,
};

/// Benchmark test case
#[derive(Debug, Clone)]
pub struct BenchmarkTestCase {
    /// Name of the benchmark
    pub name: String,
    /// JavaScript code to execute
    pub code: String,
    /// Expected execution time (for validation)
    pub expected_execution_time: Option<Duration>,
    /// Expected memory usage (for validation)
    pub expected_memory_usage: Option<usize>,
    /// Number of iterations to run
    pub iterations: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Category of the benchmark
    pub category: BenchmarkCategory,
}

/// Benchmark category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkCategory {
    /// Arithmetic operations
    Arithmetic,
    /// String operations
    String,
    /// Object property access
    ObjectAccess,
    /// Function calls
    FunctionCalls,
    /// Memory allocation
    MemoryAllocation,
    /// Garbage collection
    GarbageCollection,
    /// Control flow
    ControlFlow,
    /// Array operations
    ArrayOperations,
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Test case name
    pub test_name: String,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Minimum execution time
    pub min_execution_time: Duration,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Standard deviation of execution times
    pub execution_time_std_dev: Duration,
    /// Average memory usage
    pub average_memory_usage: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Number of iterations run
    pub iterations_run: usize,
    /// Performance score (0-100)
    pub performance_score: f64,
    /// Whether the benchmark passed validation
    pub passed_validation: bool,
    /// Integration statistics
    pub integration_stats: IntegrationStats,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Benchmark suite configuration
#[derive(Debug, Clone)]
pub struct BenchmarkSuiteConfig {
    /// Enable performance monitoring during benchmarks
    pub enable_performance_monitoring: bool,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable error tracking
    pub enable_error_tracking: bool,
    /// Timeout for individual benchmark runs
    pub benchmark_timeout: Duration,
    /// Maximum memory usage before stopping benchmark
    pub max_memory_usage: usize,
    /// Enable detailed reporting
    pub enable_detailed_reporting: bool,
    /// Output format for results
    pub output_format: OutputFormat,
}

/// Output format for benchmark results
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Human-readable format
    Human,
    /// Markdown format
    Markdown,
}

impl Default for BenchmarkSuiteConfig {
    fn default() -> Self {
        Self {
            enable_performance_monitoring: true,
            enable_memory_profiling: true,
            enable_error_tracking: true,
            benchmark_timeout: Duration::from_secs(30),
            max_memory_usage: 1024 * 1024 * 1024, // 1GB
            enable_detailed_reporting: true,
            output_format: OutputFormat::Human,
        }
    }
}

/// Benchmark suite statistics
#[derive(Debug, Clone, Default)]
pub struct BenchmarkSuiteStats {
    /// Total number of benchmarks run
    pub total_benchmarks: usize,
    /// Number of benchmarks that passed
    pub passed_benchmarks: usize,
    /// Number of benchmarks that failed
    pub failed_benchmarks: usize,
    /// Total execution time for all benchmarks
    pub total_execution_time: Duration,
    /// Average performance score across all benchmarks
    pub average_performance_score: f64,
    /// Best performing benchmark
    pub best_benchmark: Option<String>,
    /// Worst performing benchmark
    pub worst_benchmark: Option<String>,
    /// Total memory allocated during benchmarks
    pub total_memory_allocated: usize,
    /// Total errors encountered
    pub total_errors: usize,
}

/// Main benchmarking suite
#[derive(Clone)]
pub struct BenchmarkingSuite {
    /// Test cases
    test_cases: Vec<BenchmarkTestCase>,
    /// Configuration
    config: BenchmarkSuiteConfig,
    /// Integration configuration
    integration_config: IntegrationConfig,
    /// Benchmark results
    results: Vec<BenchmarkResult>,
    /// Statistics
    stats: BenchmarkSuiteStats,
    /// Baseline results for comparison
    baseline_results: HashMap<String, BenchmarkResult>,
}

impl BenchmarkingSuite {
    /// Create a new benchmarking suite
    pub fn new(config: BenchmarkSuiteConfig, integration_config: IntegrationConfig) -> Self {
        Self {
            test_cases: Vec::new(),
            config,
            integration_config,
            results: Vec::new(),
            stats: BenchmarkSuiteStats::default(),
            baseline_results: HashMap::new(),
        }
    }

    /// Add a benchmark test case
    pub fn add_test_case(&mut self, test_case: BenchmarkTestCase) {
        self.test_cases.push(test_case);
    }

    /// Add multiple test cases
    pub fn add_test_cases(&mut self, test_cases: Vec<BenchmarkTestCase>) {
        self.test_cases.extend(test_cases);
    }

    /// Run all benchmark test cases
    pub fn run_all_benchmarks(&mut self) -> Vec<BenchmarkResult> {
        self.results.clear();
        self.stats = BenchmarkSuiteStats::default();

        for test_case in &self.test_cases.clone() {
            match self.run_single_benchmark(test_case) {
                Ok(result) => {
                    self.results.push(result.clone());
                    self.update_stats(&result);
                }
                Err(e) => {
                    eprintln!("Benchmark '{}' failed: {}", test_case.name, e);
                    self.stats.failed_benchmarks += 1;
                }
            }
        }

        self.results.clone()
    }

    /// Run a single benchmark test case
    fn run_single_benchmark(
        &self,
        test_case: &BenchmarkTestCase,
    ) -> Result<BenchmarkResult, String> {
        let mut integration = VmIntegration::new(self.integration_config.clone());
        let mut context = crate::Context::default();

        // Warmup runs
        for _ in 0..test_case.warmup_iterations {
            drop(integration.execute_code(&test_case.code, &mut context));
        }

        // Actual benchmark runs
        let mut execution_times = Vec::new();
        let mut memory_usages = Vec::new();
        let mut total_errors = 0;

        for _iteration in 0..test_case.iterations {
            let start_time = Instant::now();
            let start_memory = self.get_memory_usage();

            // Execute the code
            let result = integration.execute_code(&test_case.code, &mut context);
            let execution_time = start_time.elapsed();
            let end_memory = self.get_memory_usage();

            // Check for errors
            if result.is_err() {
                total_errors += 1;
            }

            // Check timeout
            if execution_time > self.config.benchmark_timeout {
                return Err(format!("Benchmark timed out after {:?}", execution_time));
            }

            // Check memory usage
            let memory_usage = end_memory.saturating_sub(start_memory);
            if memory_usage > self.config.max_memory_usage {
                return Err(format!(
                    "Memory usage exceeded limit: {} bytes",
                    memory_usage
                ));
            }

            execution_times.push(execution_time);
            memory_usages.push(memory_usage);
        }

        // Calculate statistics
        let average_execution_time = self.calculate_average_duration(&execution_times);
        let min_execution_time = execution_times
            .iter()
            .min()
            .copied()
            .unwrap_or(Duration::ZERO);
        let max_execution_time = execution_times
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO);
        let execution_time_std_dev =
            self.calculate_std_dev_duration(&execution_times, average_execution_time);

        let average_memory_usage = memory_usages.iter().sum::<usize>() / memory_usages.len();
        let peak_memory_usage = memory_usages.iter().max().copied().unwrap_or(0);

        // Calculate performance score
        let performance_score = self.calculate_performance_score(
            average_execution_time,
            average_memory_usage,
            total_errors,
            test_case.iterations,
        );

        // Validate against expected values
        let passed_validation =
            self.validate_benchmark_result(test_case, average_execution_time, average_memory_usage);

        Ok(BenchmarkResult {
            test_name: test_case.name.clone(),
            average_execution_time,
            min_execution_time,
            max_execution_time,
            execution_time_std_dev,
            average_memory_usage,
            peak_memory_usage,
            iterations_run: test_case.iterations,
            performance_score,
            passed_validation,
            integration_stats: integration.get_stats().clone(),
            performance_metrics: integration.get_performance_metrics().clone(),
        })
    }

    /// Calculate average duration
    fn calculate_average_duration(&self, durations: &[Duration]) -> Duration {
        if durations.is_empty() {
            return Duration::ZERO;
        }

        let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
        Duration::from_nanos((total_nanos / durations.len() as u128) as u64)
    }

    /// Calculate standard deviation for durations
    fn calculate_std_dev_duration(&self, durations: &[Duration], average: Duration) -> Duration {
        if durations.len() <= 1 {
            return Duration::ZERO;
        }

        let variance: u128 = durations
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as i128 - average.as_nanos() as i128;
                (diff * diff) as u128
            })
            .sum();

        let std_dev_nanos = (variance / (durations.len() - 1) as u128) as f64;
        Duration::from_nanos(std_dev_nanos.sqrt() as u64)
    }

    /// Calculate performance score
    fn calculate_performance_score(
        &self,
        execution_time: Duration,
        memory_usage: usize,
        errors: usize,
        iterations: usize,
    ) -> f64 {
        let mut score = 100.0;

        // Penalize slow execution (normalize to 100ms baseline)
        let time_penalty = (execution_time.as_millis() as f64 / 100.0).min(50.0);
        score -= time_penalty;

        // Penalize high memory usage (normalize to 1MB baseline)
        let memory_penalty = (memory_usage as f64 / (1024.0 * 1024.0)).min(30.0);
        score -= memory_penalty;

        // Penalize errors
        let error_penalty = (errors as f64 / iterations as f64) * 20.0;
        score -= error_penalty;

        score.max(0.0)
    }

    /// Validate benchmark result against expected values
    fn validate_benchmark_result(
        &self,
        test_case: &BenchmarkTestCase,
        execution_time: Duration,
        memory_usage: usize,
    ) -> bool {
        let mut passed = true;

        if let Some(expected_time) = test_case.expected_execution_time {
            // Allow 20% tolerance
            let tolerance = expected_time.as_nanos() as f64 * 0.2;
            let actual_time = execution_time.as_nanos() as f64;
            let expected_time = expected_time.as_nanos() as f64;

            if (actual_time - expected_time).abs() > tolerance {
                passed = false;
            }
        }

        if let Some(expected_memory) = test_case.expected_memory_usage {
            // Allow 30% tolerance for memory
            let tolerance = expected_memory as f64 * 0.3;
            let actual_memory = memory_usage as f64;
            let expected_memory = expected_memory as f64;

            if (actual_memory - expected_memory).abs() > tolerance {
                passed = false;
            }
        }

        passed
    }

    /// Get current memory usage (simplified)
    fn get_memory_usage(&self) -> usize {
        // In a real implementation, this would use system memory APIs
        0
    }

    /// Update suite statistics
    fn update_stats(&mut self, result: &BenchmarkResult) {
        self.stats.total_benchmarks += 1;
        self.stats.total_execution_time += result.average_execution_time;
        self.stats.total_memory_allocated += result.average_memory_usage;
        self.stats.total_errors += result.integration_stats.total_errors_handled;

        if result.passed_validation {
            self.stats.passed_benchmarks += 1;
        } else {
            self.stats.failed_benchmarks += 1;
        }

        // Update best/worst benchmarks
        if self.stats.best_benchmark.is_none()
            || result.performance_score > self.get_benchmark_score(&self.stats.best_benchmark)
        {
            self.stats.best_benchmark = Some(result.test_name.clone());
        }

        if self.stats.worst_benchmark.is_none()
            || result.performance_score < self.get_benchmark_score(&self.stats.worst_benchmark)
        {
            self.stats.worst_benchmark = Some(result.test_name.clone());
        }

        // Update average performance score
        let total_score: f64 = self.results.iter().map(|r| r.performance_score).sum();
        self.stats.average_performance_score = total_score / self.results.len() as f64;
    }

    /// Get performance score for a benchmark by name
    fn get_benchmark_score(&self, benchmark_name: &Option<String>) -> f64 {
        if let Some(name) = benchmark_name {
            self.results
                .iter()
                .find(|r| r.test_name == *name)
                .map(|r| r.performance_score)
                .unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Set baseline results for comparison
    pub fn set_baseline_results(&mut self, baseline_results: HashMap<String, BenchmarkResult>) {
        self.baseline_results = baseline_results;
    }

    /// Compare results with baseline
    pub fn compare_with_baseline(&self) -> Vec<BenchmarkComparison> {
        let mut comparisons = Vec::new();

        for result in &self.results {
            if let Some(baseline) = self.baseline_results.get(&result.test_name) {
                let comparison = BenchmarkComparison {
                    test_name: result.test_name.clone(),
                    execution_time_improvement: self.calculate_improvement(
                        baseline.average_execution_time.as_millis() as f64,
                        result.average_execution_time.as_millis() as f64,
                    ),
                    memory_usage_improvement: self.calculate_improvement(
                        baseline.average_memory_usage as f64,
                        result.average_memory_usage as f64,
                    ),
                    performance_score_improvement: result.performance_score
                        - baseline.performance_score,
                    overall_improvement: 0.0, // Will be calculated
                };

                comparisons.push(comparison);
            }
        }

        // Calculate overall improvement
        for comparison in &mut comparisons {
            comparison.overall_improvement = (comparison.execution_time_improvement
                + comparison.memory_usage_improvement
                + comparison.performance_score_improvement)
                / 3.0;
        }

        comparisons
    }

    /// Calculate improvement percentage
    fn calculate_improvement(&self, baseline: impl Into<f64>, current: impl Into<f64>) -> f64 {
        let baseline: f64 = baseline.into();
        let current: f64 = current.into();

        if baseline == 0.0 {
            return 0.0;
        }

        ((baseline - current) / baseline) * 100.0
    }

    /// Generate benchmark report
    pub fn generate_report(&self) -> String {
        match self.config.output_format {
            OutputFormat::Json => self.generate_json_report(),
            OutputFormat::Csv => self.generate_csv_report(),
            OutputFormat::Markdown => self.generate_markdown_report(),
            OutputFormat::Human => self.generate_human_report(),
        }
    }

    /// Generate human-readable report
    fn generate_human_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== BENCHMARK SUITE REPORT ===\n\n");
        report.push_str(&format!(
            "Total Benchmarks: {}\n",
            self.stats.total_benchmarks
        ));
        report.push_str(&format!("Passed: {}\n", self.stats.passed_benchmarks));
        report.push_str(&format!("Failed: {}\n", self.stats.failed_benchmarks));
        report.push_str(&format!(
            "Average Performance Score: {:.2}\n",
            self.stats.average_performance_score
        ));
        report.push_str(&format!(
            "Total Execution Time: {:?}\n",
            self.stats.total_execution_time
        ));
        report.push_str(&format!(
            "Total Memory Allocated: {} bytes\n",
            self.stats.total_memory_allocated
        ));
        report.push_str(&format!("Total Errors: {}\n\n", self.stats.total_errors));

        if let Some(best) = &self.stats.best_benchmark {
            report.push_str(&format!("Best Benchmark: {}\n", best));
        }
        if let Some(worst) = &self.stats.worst_benchmark {
            report.push_str(&format!("Worst Benchmark: {}\n", worst));
        }

        report.push_str("\n=== DETAILED RESULTS ===\n\n");

        for result in &self.results {
            report.push_str(&format!("Test: {}\n", result.test_name));
            report.push_str(&format!(
                "  Average Time: {:?}\n",
                result.average_execution_time
            ));
            report.push_str(&format!(
                "  Memory Usage: {} bytes\n",
                result.average_memory_usage
            ));
            report.push_str(&format!(
                "  Performance Score: {:.2}\n",
                result.performance_score
            ));
            report.push_str(&format!("  Passed: {}\n", result.passed_validation));
            report.push_str("\n");
        }

        report
    }

    /// Generate JSON report
    fn generate_json_report(&self) -> String {
        // Simplified JSON generation
        format!(
            r#"{{"total_benchmarks": {}, "passed": {}, "failed": {}, "average_score": {:.2}}}"#,
            self.stats.total_benchmarks,
            self.stats.passed_benchmarks,
            self.stats.failed_benchmarks,
            self.stats.average_performance_score
        )
    }

    /// Generate CSV report
    fn generate_csv_report(&self) -> String {
        let mut csv = String::from(
            "Test Name,Average Time (ms),Memory Usage (bytes),Performance Score,Passed\n",
        );

        for result in &self.results {
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                result.test_name,
                result.average_execution_time.as_millis(),
                result.average_memory_usage,
                result.performance_score,
                result.passed_validation
            ));
        }

        csv
    }

    /// Generate Markdown report
    fn generate_markdown_report(&self) -> String {
        let mut md = String::new();

        md.push_str("# Benchmark Suite Report\n\n");
        md.push_str(&format!(
            "- **Total Benchmarks**: {}\n",
            self.stats.total_benchmarks
        ));
        md.push_str(&format!("- **Passed**: {}\n", self.stats.passed_benchmarks));
        md.push_str(&format!("- **Failed**: {}\n", self.stats.failed_benchmarks));
        md.push_str(&format!(
            "- **Average Performance Score**: {:.2}\n",
            self.stats.average_performance_score
        ));

        md.push_str("\n## Detailed Results\n\n");
        md.push_str("| Test Name | Average Time | Memory Usage | Performance Score | Passed |\n");
        md.push_str("|-----------|--------------|--------------|-------------------|--------|\n");

        for result in &self.results {
            md.push_str(&format!(
                "| {} | {:?} | {} bytes | {:.2} | {} |\n",
                result.test_name,
                result.average_execution_time,
                result.average_memory_usage,
                result.performance_score,
                result.passed_validation
            ));
        }

        md
    }

    /// Get benchmark results
    pub fn get_results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Get suite statistics
    pub fn get_stats(&self) -> &BenchmarkSuiteStats {
        &self.stats
    }

    /// Create default test cases
    pub fn create_default_test_cases() -> Vec<BenchmarkTestCase> {
        vec![
            BenchmarkTestCase {
                name: "arithmetic_operations".to_string(),
                code: "let sum = 0; for(let i = 0; i < 1000; i++) { sum += i; }".to_string(),
                expected_execution_time: Some(Duration::from_millis(10)),
                expected_memory_usage: Some(1024),
                iterations: 100,
                warmup_iterations: 10,
                category: BenchmarkCategory::Arithmetic,
            },
            BenchmarkTestCase {
                name: "string_operations".to_string(),
                code: "let str = ''; for(let i = 0; i < 100; i++) { str += 'a'; }".to_string(),
                expected_execution_time: Some(Duration::from_millis(5)),
                expected_memory_usage: Some(512),
                iterations: 50,
                warmup_iterations: 5,
                category: BenchmarkCategory::String,
            },
            BenchmarkTestCase {
                name: "object_property_access".to_string(),
                code: "let obj = {a: 1, b: 2, c: 3}; for(let i = 0; i < 1000; i++) { let val = obj.a + obj.b + obj.c; }".to_string(),
                expected_execution_time: Some(Duration::from_millis(8)),
                expected_memory_usage: Some(2048),
                iterations: 100,
                warmup_iterations: 10,
                category: BenchmarkCategory::ObjectAccess,
            },
        ]
    }
}

/// Benchmark comparison with baseline
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    /// Test name
    pub test_name: String,
    /// Execution time improvement percentage
    pub execution_time_improvement: f64,
    /// Memory usage improvement percentage
    pub memory_usage_improvement: f64,
    /// Performance score improvement
    pub performance_score_improvement: f64,
    /// Overall improvement
    pub overall_improvement: f64,
}

impl Default for BenchmarkingSuite {
    fn default() -> Self {
        Self::new(
            BenchmarkSuiteConfig::default(),
            IntegrationConfig::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmarking_suite_creation() {
        let config = BenchmarkSuiteConfig::default();
        let integration_config = IntegrationConfig::default();
        let suite = BenchmarkingSuite::new(config, integration_config);

        assert_eq!(suite.test_cases.len(), 0);
        assert_eq!(suite.results.len(), 0);
    }

    #[test]
    fn test_add_test_cases() {
        let mut suite = BenchmarkingSuite::new(
            BenchmarkSuiteConfig::default(),
            IntegrationConfig::default(),
        );

        let test_cases = BenchmarkingSuite::create_default_test_cases();
        suite.add_test_cases(test_cases.clone());

        assert_eq!(suite.test_cases.len(), test_cases.len());
    }

    #[test]
    fn test_duration_calculations() {
        let suite = BenchmarkingSuite::new(
            BenchmarkSuiteConfig::default(),
            IntegrationConfig::default(),
        );

        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
        ];

        let average = suite.calculate_average_duration(&durations);
        assert_eq!(average, Duration::from_millis(20));

        let std_dev = suite.calculate_std_dev_duration(&durations, average);
        assert!(std_dev > Duration::ZERO);
    }

    #[test]
    fn test_performance_score_calculation() {
        let suite = BenchmarkingSuite::new(
            BenchmarkSuiteConfig::default(),
            IntegrationConfig::default(),
        );

        let score = suite.calculate_performance_score(Duration::from_millis(50), 1024, 0, 100);

        assert!(score >= 0.0 && score <= 100.0);
    }

    #[test]
    fn test_benchmark_validation() {
        let suite = BenchmarkingSuite::new(
            BenchmarkSuiteConfig::default(),
            IntegrationConfig::default(),
        );

        let test_case = BenchmarkTestCase {
            name: "test".to_string(),
            code: "1 + 1".to_string(),
            expected_execution_time: Some(Duration::from_millis(10)),
            expected_memory_usage: Some(100),
            iterations: 1,
            warmup_iterations: 0,
            category: BenchmarkCategory::Arithmetic,
        };

        // Test with values within tolerance
        let passed = suite.validate_benchmark_result(
            &test_case,
            Duration::from_millis(12), // Within 20% tolerance
            120,                       // Within 30% tolerance
        );
        assert!(passed);

        // Test with values outside tolerance
        let failed = suite.validate_benchmark_result(
            &test_case,
            Duration::from_millis(50), // Outside tolerance
            500,                       // Outside tolerance
        );
        assert!(!failed);
    }

    #[test]
    fn test_report_generation() {
        let mut suite = BenchmarkingSuite::new(
            BenchmarkSuiteConfig::default(),
            IntegrationConfig::default(),
        );

        // Add a test case and run it
        let test_cases = BenchmarkingSuite::create_default_test_cases();
        suite.add_test_cases(test_cases);

        // Generate reports in different formats
        let human_report = suite.generate_report();
        assert!(human_report.contains("BENCHMARK SUITE REPORT"));

        // Test other formats
        let mut json_suite = suite.clone();
        json_suite.config.output_format = OutputFormat::Json;
        let json_report = json_suite.generate_report();
        assert!(json_report.contains("{"));

        let mut csv_suite = suite.clone();
        csv_suite.config.output_format = OutputFormat::Csv;
        let csv_report = csv_suite.generate_report();
        assert!(csv_report.contains("Test Name"));

        let mut md_suite = suite.clone();
        md_suite.config.output_format = OutputFormat::Markdown;
        let md_report = md_suite.generate_report();
        assert!(md_report.contains("# Benchmark Suite Report"));
    }

    #[test]
    fn test_improvement_calculation() {
        let suite = BenchmarkingSuite::new(
            BenchmarkSuiteConfig::default(),
            IntegrationConfig::default(),
        );

        let improvement = suite.calculate_improvement(100.0, 80.0);
        assert_eq!(improvement, 20.0); // 20% improvement

        let regression = suite.calculate_improvement(100.0, 120.0);
        assert_eq!(regression, -20.0); // 20% regression
    }
}

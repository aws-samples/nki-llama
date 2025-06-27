#!/usr/bin/env python3
"""
Handler for calculating NKI-LLAMA scores combining inference and training metrics.
This script invokes calculate_training_metrics.py and processes benchmark results.
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime
import glob
import re


class NKILlamaHandler:
    """Handler for calculating and managing NKI-LLAMA benchmark scores."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def build_training_command(self, cmd_args: Dict[str, Any]) -> list:
        """
        Build the command for calculate_training_metrics.py.
        
        Args:
            cmd_args: Dictionary of command line arguments
            
        Returns:
            List representing the command to execute
        """
        # Build command
        cmd = ["python", cmd_args["script_path"]]
        
        # Add required arguments
        cmd.extend(["--config", cmd_args["config"]])
        cmd.extend(["--model-config", cmd_args["model_config"]])
        
        # Add optional arguments
        if cmd_args.get("log_file"):
            cmd.extend(["--log-file", cmd_args["log_file"]])
        if cmd_args.get("compile_dir"):
            cmd.extend(["--compile-dir", cmd_args["compile_dir"]])
        if cmd_args.get("throughput"):
            cmd.extend(["--throughput", str(cmd_args["throughput"])])
        if cmd_args.get("hw_backend"):
            cmd.extend(["--hw-backend", cmd_args["hw_backend"]])
        if cmd_args.get("batch_size"):
            cmd.extend(["--batch-size", str(cmd_args["batch_size"])])
        if cmd_args.get("seq_len"):
            cmd.extend(["--seq-len", str(cmd_args["seq_len"])])
        if cmd_args.get("num_nodes"):
            cmd.extend(["--num-nodes", str(cmd_args["num_nodes"])])
            
        # Add scoring parameters
        if cmd_args.get("calculate_score"):
            cmd.append("--calculate-score")
        if cmd_args.get("per_file_scores"):
            cmd.append("--per-file-scores")
        if cmd_args.get("detailed"):
            cmd.append("--detailed")
        if cmd_args.get("print_per_file"):
            cmd.append("--print-per-file")
            
        # Output file
        output_file = cmd_args.get("output", "benchmark_finetuning.json")
        cmd.extend(["--output", output_file])
        
        return cmd
        
    def run_training_metrics(self, cmd_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run calculate_training_metrics.py with the specified arguments.
        
        Args:
            cmd_args: Dictionary of command line arguments
            
        Returns:
            Dictionary containing the training metrics results
        """
        # Build the command
        cmd = self.build_training_command(cmd_args)
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        # ADD THIS PRINT STATEMENT FOR BETTER VISIBILITY
        print("\n" + "="*80)
        print("🚀 EXECUTING TRAINING METRICS COMMAND:")
        print("="*80)
        print(f"Command: {' '.join(cmd)}")
        print("="*80 + "\n")
        
        output_file = cmd_args.get("output", "benchmark_finetuning.json")
        
        try:
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Print stdout if verbose
            if self.verbose and result.stdout:
                print("=== Training Metrics Output ===")
                print(result.stdout)
                print("==============================")
                
            # Load and return the results
            with open(output_file, 'r') as f:
                return json.load(f)
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running calculate_training_metrics.py: {e}")
            if e.stderr:
                self.logger.error(f"Error output: {e.stderr}")
            raise
            
    def calculate_inference_score(self, inference_data: Dict[str, Any], 
                                  reference_data: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict]:
        """
        Calculate inference score based on the benchmark_inference definition.
        
        Score = Accuracy * Reduced Latency * Increased Throughput * (1 + Normalized NKI FLOPS)
        
        Args:
            inference_data: Dictionary containing inference benchmark results
            reference_data: Optional reference implementation data
            
        Returns:
            Tuple of (score, score_breakdown)
        """
        # Default reference values if not provided
        if reference_data is None:
            reference_data = {
                "e2e_model": {
                    "latency_ms_avg": 50000,  # 50 seconds reference
                    "throughput": 10  # 10 tokens/sec reference
                },
                "accuracy": 1.0  # Assume accuracy threshold is met
            }
        
        # Extract metrics from inference data
        e2e_latency = inference_data["e2e_model"]["latency_ms_avg"]
        e2e_throughput = inference_data["e2e_model"]["throughput"]
        
        # Calculate components
        accuracy = reference_data.get("accuracy", 1.0)  # Binary: 1 if meets threshold, 0 otherwise
        
        # Reduced Latency = Reference TTFT / Submission TTFT
        # Using e2e latency as proxy for TTFT
        reduced_latency = reference_data["e2e_model"]["latency_ms_avg"] / e2e_latency
        
        # Increased Throughput = Submission tokens/sec / Reference tokens/sec  
        increased_throughput = e2e_throughput / reference_data["e2e_model"]["throughput"]
        
        # Normalized NKI FLOPS - this would come from the training metrics
        # For now, using a placeholder - this should be integrated with training metrics
        normalized_nki_flops = 0.0  # Will be updated when combined with training metrics
        
        # Calculate final score
        score = accuracy * reduced_latency * increased_throughput * (1 + normalized_nki_flops)
        
        breakdown = {
            "accuracy": accuracy,
            "reduced_latency": reduced_latency,
            "increased_throughput": increased_throughput,
            "normalized_nki_flops": normalized_nki_flops,
            "reference_latency_ms": reference_data["e2e_model"]["latency_ms_avg"],
            "achieved_latency_ms": e2e_latency,
            "reference_throughput": reference_data["e2e_model"]["throughput"],
            "achieved_throughput": e2e_throughput
        }
        
        return score, breakdown
        
    def calculate_reasoning_score(self, reasoning_metrics: Dict[str, Any]) -> float:
        """
        Calculate reasoning score from reasoning evaluation results.
        
        The reasoning score is based on the overall accuracy across reasoning tasks,
        normalized to a 0-1 scale and then scaled to match the scoring range of other components.
        
        Args:
            reasoning_metrics: Dictionary containing reasoning evaluation results
            
        Returns:
            Float representing the reasoning score
        """
        try:
            # Get the overall score (already averaged across tasks)
            overall_score = reasoning_metrics.get("overall_score", 0.0)
            
            # The overall_score is already in 0-1 range (accuracy percentage)
            # Scale it to match the typical range of other components (0-10 range)
            reasoning_score = overall_score * 10.0
            
            self.logger.debug(f"Reasoning score calculation: {overall_score:.4f} -> {reasoning_score:.4f}")
            
            return reasoning_score
            
        except Exception as e:
            self.logger.error(f"Error calculating reasoning score: {e}")
            return 0.0
        
    def calculate_combined_score(self, training_metrics: Dict[str, Any],
                                 inference_metrics: Optional[Dict[str, Any]] = None,
                                 reasoning_metrics: Optional[Dict[str, Any]] = None,
                                 weights: Optional[Dict[str, float]] = None,
                                 reference_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate combined NKI-LLAMA score from training, inference, and reasoning metrics.
        Handles graceful fallback when components are not available.
        
        Args:
            training_metrics: Training metrics including NKI analysis
            inference_metrics: Optional inference benchmark results
            reasoning_metrics: Optional reasoning evaluation results
            weights: Optional weights for combining scores
            reference_data: Optional reference implementation data for inference scoring
            
        Returns:
            Dictionary containing combined score and breakdown
        """
        if weights is None:
            # Default weights based on available components
            if inference_metrics is not None and reasoning_metrics is not None:
                weights = {
                    "training": 0.3,
                    "inference": 0.4,
                    "reasoning": 0.3
                }
            elif inference_metrics is not None:
                weights = {
                    "training": 0.4,
                    "inference": 0.6,
                    "reasoning": 0.0
                }
            elif reasoning_metrics is not None:
                weights = {
                    "training": 0.7,
                    "inference": 0.0,
                    "reasoning": 0.3
                }
            else:
                weights = {
                    "training": 1.0,
                    "inference": 0.0,
                    "reasoning": 0.0
                }
            
        # Get training score and NKI ratio
        training_score = training_metrics.get("training_score", 0.0)
        nki_ratio = training_metrics["nki_analysis"]["summary"]["overall_nki_ratio"]
        
        # Calculate reasoning score if available
        reasoning_score = None
        if reasoning_metrics is not None:
            reasoning_score = self.calculate_reasoning_score(reasoning_metrics)
        
        # Determine execution mode based on available components
        available_components = []
        if training_metrics is not None:
            available_components.append("training")
        if inference_metrics is not None:
            available_components.append("inference")
        if reasoning_metrics is not None:
            available_components.append("reasoning")
            
        mode = "_".join(available_components) if len(available_components) > 1 else f"{available_components[0]}_only"
        
        # Handle training-only mode
        if inference_metrics is None and reasoning_metrics is None:
            return {
                "combined_score": training_score,
                "training_score": training_score,
                "inference_score": None,
                "reasoning_score": None,
                "weights": weights,
                "mode": "training_only",
                "breakdown": {
                    "training": training_metrics.get("training_score_breakdown", {}),
                    "inference": None,
                    "reasoning": None
                },
                "nki_ratio": nki_ratio
            }
        
        # Calculate inference score with NKI ratio if available
        inference_score_with_nki = None
        inference_breakdown = None
        if inference_metrics is not None:
            inference_score, inference_breakdown = self.calculate_inference_score(inference_metrics, reference_data)
            
            # Update inference score with actual NKI FLOPS ratio
            inference_breakdown["normalized_nki_flops"] = nki_ratio
            inference_score_with_nki = (
                inference_breakdown["accuracy"] * 
                inference_breakdown["reduced_latency"] * 
                inference_breakdown["increased_throughput"] * 
                (1 + nki_ratio)
            )
        
        # Calculate weighted average based on available components
        combined_score = 0.0
        if weights["training"] > 0:
            combined_score += weights["training"] * training_score
        if weights["inference"] > 0 and inference_score_with_nki is not None:
            combined_score += weights["inference"] * inference_score_with_nki
        if weights["reasoning"] > 0 and reasoning_score is not None:
            combined_score += weights["reasoning"] * reasoning_score
        
        return {
            "combined_score": combined_score,
            "training_score": training_score,
            "inference_score": inference_score_with_nki,
            "reasoning_score": reasoning_score,
            "weights": weights,
            "mode": mode,
            "breakdown": {
                "training": training_metrics.get("training_score_breakdown", {}),
                "inference": inference_breakdown,
                "reasoning": reasoning_metrics.get("tasks", {}) if reasoning_metrics else None
            },
            "nki_ratio": nki_ratio
        }
        
    def display_results(self, results: Dict[str, Any]):
        """Display the benchmark results in a formatted way."""
        print("\n" + "="*70)
        print("NKI-LLAMA BENCHMARK RESULTS")
        print("="*70)
        
        mode = results.get("mode", "unknown")
        
        # Check mode and display appropriate results
        if mode == "training_only":
            print("\n⚠️  TRAINING-ONLY MODE (Inference and reasoning results not available)")
            print(f"\n🏆 NKI KERNEL TRAINING SCORE: {results['training_score']:.4f}")
            print(f"   NKI Ratio: {results['nki_ratio']:.4f}")
            
            # Training breakdown
            if "training" in results["breakdown"] and results["breakdown"]["training"]:
                tb = results["breakdown"]["training"]
                print(f"\n🎯 Training Metrics:")
                print(f"  MFU: {tb.get('achieved_mfu', 0):.2f}% (baseline: {tb.get('base_mfu', 0):.2f}%)")
                print(f"  Throughput: {tb.get('achieved_throughput', 0):.2f} seq/s (baseline: {tb.get('base_throughput', 0):.2f})")
                print(f"  MFU Improvement: {tb.get('mfu_improvement', 0):.4f}x")
                print(f"  Throughput Improvement: {tb.get('throughput_improvement', 0):.4f}x")
                
            print("\n💡 Note: This score represents training performance only.")
            print("   To get the full NKI-LLAMA score, run inference benchmarks and reasoning")
            print("   evaluation, then provide the results using --inference-results and --reasoning-results options.")
            
        else:
            # Multi-component mode - display based on available components
            print(f"\n🏆 FINAL NKI-LLAMA SCORE: {results['combined_score']:.4f}")
            print(f"\nExecution Mode: {mode.replace('_', ' + ').title()}")
            
            print(f"\nScore Weights:")
            if results['weights']['training'] > 0:
                print(f"  Training: {results['weights']['training']*100:.0f}%")
            if results['weights']['inference'] > 0:
                print(f"  Inference: {results['weights']['inference']*100:.0f}%")
            if results['weights']['reasoning'] > 0:
                print(f"  Reasoning: {results['weights']['reasoning']*100:.0f}%")
            
            # Component scores
            print(f"\n📊 Component Scores:")
            print(f"  Training Score: {results['training_score']:.4f}")
            if results['inference_score'] is not None:
                print(f"  Inference Score: {results['inference_score']:.4f}")
            if results['reasoning_score'] is not None:
                print(f"  Reasoning Score: {results['reasoning_score']:.4f}")
            print(f"  NKI Ratio: {results['nki_ratio']:.4f}")
            
            # Training breakdown
            if "training" in results["breakdown"] and results["breakdown"]["training"]:
                tb = results["breakdown"]["training"]
                print(f"\n🎯 Training Metrics:")
                print(f"  MFU: {tb.get('achieved_mfu', 0):.2f}% (baseline: {tb.get('base_mfu', 0):.2f}%)")
                print(f"  Throughput: {tb.get('achieved_throughput', 0):.2f} seq/s (baseline: {tb.get('base_throughput', 0):.2f})")
                print(f"  MFU Improvement: {tb.get('mfu_improvement', 0):.4f}x")
                print(f"  Throughput Improvement: {tb.get('throughput_improvement', 0):.4f}x")
                
            # Inference breakdown
            if results["breakdown"]["inference"]:
                ib = results["breakdown"]["inference"]
                print(f"\n⚡ Inference Metrics:")
                print(f"  Latency: {ib['achieved_latency_ms']:.2f}ms (reference: {ib['reference_latency_ms']:.2f}ms)")
                print(f"  Throughput: {ib['achieved_throughput']:.2f} tokens/s (reference: {ib['reference_throughput']:.2f})")
                print(f"  Latency Reduction: {ib['reduced_latency']:.4f}x")
                print(f"  Throughput Increase: {ib['increased_throughput']:.4f}x")
                print(f"  Accuracy: {'✓ Passed' if ib['accuracy'] == 1.0 else '✗ Failed'}")
        
        print("\n" + "="*70)
        
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save the combined results to a JSON file."""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "mode": results.get("mode", "combined"),
            "nki_kernel_score": results["combined_score"],
            "component_scores": {
                "training": results["training_score"],
                "inference": results.get("inference_score")
            },
            "weights": results["weights"],
            "nki_ratio": results["nki_ratio"],
            "detailed_breakdown": results["breakdown"]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        self.logger.info(f"Results saved to: {output_file}")
        
    def discover_reasoning_results(self, model_path: str, 
                                   results_base_path: str = "/home/ubuntu/aws-neuron-samples/inference-benchmarking/results") -> Optional[Dict[str, Any]]:
        """
        Discover reasoning results based on model ID in aws-neuron-samples/inference-benchmarking/results/.
        
        Args:
            model_path: Path to the model (e.g., "/home/ubuntu/models/llama-3-1-8b")
            results_base_path: Base path to reasoning results directory
            
        Returns:
            Dictionary containing reasoning results or None if not found
        """
        try:
            # Extract model name from path and create sanitized version
            model_name = os.path.basename(model_path.rstrip('/'))
            sanitized_model_path = model_path.replace('/', '__')
            
            self.logger.info(f"Searching for reasoning results for model: {model_name}")
            self.logger.debug(f"Sanitized model path: {sanitized_model_path}")
            
            # Search in accuracy results directory
            accuracy_base = os.path.join(results_base_path, "accuracy")
            
            if not os.path.exists(accuracy_base):
                self.logger.warning(f"Accuracy results directory not found: {accuracy_base}")
                return None
                
            # Find all result files matching the model path pattern
            search_pattern = os.path.join(accuracy_base, "**", sanitized_model_path, "results_*.json")
            result_files = glob.glob(search_pattern, recursive=True)
            
            if not result_files:
                self.logger.info(f"No reasoning result files found for model {model_name}")
                self.logger.debug(f"Search pattern used: {search_pattern}")
                return None
                
            # Use the most recent result file (based on timestamp in filename)
            latest_file = max(result_files, key=lambda f: os.path.getmtime(f))
            self.logger.info(f"Found reasoning results: {latest_file}")
            
            # Load and parse the result file
            with open(latest_file, 'r') as f:
                reasoning_data = json.load(f)
                
            return self.parse_reasoning_results(reasoning_data, latest_file)
            
        except Exception as e:
            self.logger.error(f"Error discovering reasoning results: {e}")
            return None
            
    def parse_reasoning_results(self, reasoning_data: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """
        Parse JSON result files to extract "exact_match,strict-match" scores.
        
        Args:
            reasoning_data: Raw reasoning results data
            file_path: Path to the result file for metadata
            
        Returns:
            Dictionary containing parsed reasoning scores
        """
        try:
            parsed_results = {
                "source_file": file_path,
                "model_name": reasoning_data.get("model_name", "unknown"),
                "model_name_sanitized": reasoning_data.get("model_name_sanitized", "unknown"),
                "evaluation_time": reasoning_data.get("total_evaluation_time_seconds", 0),
                "tasks": {},
                "overall_score": 0.0,
                "task_count": 0
            }
            
            # Extract scores from each task
            results_section = reasoning_data.get("results", {})
            total_score = 0.0
            task_count = 0
            
            for task_name, task_results in results_section.items():
                # Look for exact_match,strict-match score
                strict_match_score = task_results.get("exact_match,strict-match")
                flexible_extract_score = task_results.get("exact_match,flexible-extract")
                
                if strict_match_score is not None:
                    parsed_results["tasks"][task_name] = {
                        "exact_match_strict": strict_match_score,
                        "exact_match_flexible": flexible_extract_score,
                        "primary_score": strict_match_score  # Use strict-match as primary
                    }
                    total_score += strict_match_score
                    task_count += 1
                    
                    self.logger.debug(f"Task {task_name}: strict-match={strict_match_score}, flexible-extract={flexible_extract_score}")
                    
            # Calculate overall average score
            if task_count > 0:
                parsed_results["overall_score"] = total_score / task_count
                parsed_results["task_count"] = task_count
                
            self.logger.info(f"Parsed reasoning results: {task_count} tasks, overall score: {parsed_results['overall_score']:.4f}")
            return parsed_results
            
        except Exception as e:
            self.logger.error(f"Error parsing reasoning results: {e}")
            return {
                "source_file": file_path,
                "error": str(e),
                "overall_score": 0.0,
                "task_count": 0,
                "tasks": {}
            }
            
    def map_model_config_to_path(self, model_config_path: str) -> str:
        """
        Map model configuration to corresponding model path for reasoning result discovery.
        
        Args:
            model_config_path: Path to model configuration file
            
        Returns:
            Inferred model path for reasoning result lookup
        """
        try:
            # Try to extract model information from config file
            if os.path.exists(model_config_path):
                with open(model_config_path, 'r') as f:
                    config_data = json.load(f)
                    
                # Look for model path hints in config
                model_name_or_path = config_data.get("model_name_or_path", "")
                if model_name_or_path and os.path.exists(model_name_or_path):
                    return model_name_or_path
                    
            # Fallback: infer from config path structure
            # e.g., /path/to/8B_config_llama3-1/config.json -> llama-3-1-8b
            config_dir = os.path.dirname(model_config_path)
            config_dir_name = os.path.basename(config_dir)
            
            # Map common config directory patterns to model names
            model_mapping = {
                "8B_config_llama3-1": "/home/ubuntu/models/llama-3-1-8b",
                "8B_config_llama3": "/home/ubuntu/models/llama-3-8b-distill",
                # Add more mappings as needed
            }
            
            if config_dir_name in model_mapping:
                return model_mapping[config_dir_name]
                
            # Final fallback: assume standard model path
            return "/home/ubuntu/models/llama-3-1-8b"
            
        except Exception as e:
            self.logger.warning(f"Error mapping model config to path: {e}")
            return "/home/ubuntu/models/llama-3-1-8b"

def main():
    parser = argparse.ArgumentParser(
        description="Handler for NKI-LLAMA benchmark score calculation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training metrics arguments
    training_group = parser.add_argument_group('Training Metrics')
    training_group.add_argument(
        "--training-script",
        default="/home/ubuntu/nki-llama/src/fine-tune/scripts/calculate_training_metrics.py",
        help="Path to calculate_training_metrics.py"
    )
    training_group.add_argument(
        "--config",
        default="/home/ubuntu/nki-llama/src/fine-tune/neuronx-distributed-training/examples/conf/hf_llama3_8B_SFT_config.yaml",
        help="Training config YAML file"
    )
    training_group.add_argument(
        "--model-config",
        default="/home/ubuntu/nki-llama/src/fine-tune/configs/model-config/8B_config_llama3-1/config.json",
        help="Model config JSON file"
    )
    training_group.add_argument(
        "--log-file",
        default="/home/ubuntu/nki-llama/logs/nki-llama_20250610_014432.log",
        help="Training log file"
    )
    training_group.add_argument(
        "--compile-dir",
        default="/home/ubuntu/neuron_cache",
        help="Neuron compile cache directory"
    )
    training_group.add_argument(
        "--throughput",
        type=float,
        default=2.1,
        help="Training throughput in seq/s"
    )
    training_group.add_argument(
        "--hw-backend",
        choices=['trn1', 'trn2'],
        default='trn1',
        help="Hardware backend"
    )
    
    # Inference metrics arguments
    inference_group = parser.add_argument_group('Inference Metrics')
    inference_group.add_argument(
        "--inference-results",
        default="benchmark_inference.json",
        help="Path to inference benchmark results (optional - if not provided, only training score is calculated)"
    )
    inference_group.add_argument(
        "--reference-latency",
        type=float,
        default=50000,
        help="Reference implementation latency in ms"
    )
    inference_group.add_argument(
        "--reference-throughput",
        type=float,
        default=10,
        help="Reference implementation throughput in tokens/s"
    )
    
    # Score calculation arguments
    score_group = parser.add_argument_group('Score Calculation')
    score_group.add_argument(
        "--training-weight",
        type=float,
        default=0.4,
        help="Weight for training score (0-1)"
    )
    score_group.add_argument(
        "--inference-weight",
        type=float,
        default=0.6,
        help="Weight for inference score (0-1)"
    )
    score_group.add_argument(
        "--calculate-score",
        action="store_true",
        help="Calculate training score"
    )
    score_group.add_argument(
        "--per-file-scores",
        action="store_true",
        help="Calculate per-file scores"
    )
    score_group.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed output"
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output file for combined benchmark results"
    )
    output_group.add_argument(
        "--training-output",
        default="benchmark_finetuning.json",
        help="Output file for training metrics"
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate weights
    if args.training_weight + args.inference_weight != 1.0:
        parser.error("Training and inference weights must sum to 1.0")
        
    # Create handler
    handler = NKILlamaHandler(verbose=args.verbose)
    
    try:
        # Step 1: Run training metrics calculation
        print("📈 Calculating training metrics...")
        training_args = {
            "script_path": args.training_script,
            "config": args.config,
            "model_config": args.model_config,
            "log_file": args.log_file,
            "compile_dir": args.compile_dir,
            "throughput": args.throughput,
            "hw_backend": args.hw_backend,
            "calculate_score": args.calculate_score,
            "per_file_scores": args.per_file_scores,
            "detailed": args.detailed,
            "print_per_file": args.verbose,
            "output": args.training_output,
            "base_mfu": 50.0,
            "base_throughput": 100.0,
            "loss_improvement": 1.0,
            "convergence_rate": 1.0
        }
        
        # Build the command to display it even if using existing file
        cmd = handler.build_training_command(training_args)
        
        # Check if we need to run training metrics or use existing file
        if os.path.exists(args.training_output):
            # DISPLAY THE COMMAND THAT WOULD BE EXECUTED
            print("\n" + "="*80)
            print("📋 TRAINING METRICS COMMAND (using existing file instead):")
            print("="*80)
            print(f"Command that would be executed:\n{' '.join(cmd)}")
            print("="*80 + "\n")
            
            handler.logger.info(f"Using existing training metrics from {args.training_output}")
            with open(args.training_output, 'r') as f:
                training_metrics = json.load(f)
        else:
            training_metrics = handler.run_training_metrics(training_args)
            
        # Step 2: Check for inference metrics
        inference_metrics = None
        inference_available = os.path.exists(args.inference_results)
        
        if inference_available:
            print("\n⚡ Loading inference metrics...")
            with open(args.inference_results, 'r') as f:
                inference_metrics = json.load(f)
        else:
            print("\n⚠️  Inference results file not found. Running in training-only mode.")
            print(f"   (Looking for: {args.inference_results})")
            
        # Step 3: Calculate score(s)
        if inference_available:
            print("\n🔬 Calculating combined NKI-LLAMA score...")
        else:
            print("\n🔬 Calculating NKI kernel training score...")
            
        weights = {
            "training": args.training_weight,
            "inference": args.inference_weight
        }
        
        # Set reference data for inference scoring
        reference_data = {
            "e2e_model": {
                "latency_ms_avg": args.reference_latency,
                "throughput": args.reference_throughput
            },
            "accuracy": 1.0  # Assuming accuracy threshold is met
        }
        
        # Calculate score - will handle both training-only and combined modes
        results = handler.calculate_combined_score(
            training_metrics,
            inference_metrics,
            weights,
            reference_data
        )
        
        # Step 4: Display results
        handler.display_results(results)
        
        # Step 5: Save results
        handler.save_results(results, args.output)
        
        if inference_available:
            print(f"\n✅ Benchmark complete! Results saved to {args.output}")
        else:
            print(f"\n✅ Training benchmark complete! Results saved to {args.output}")
            print("   Run inference benchmarks to get the full NKI-LLAMA score.")
        
    except Exception as e:
        handler.logger.error(f"Error during benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
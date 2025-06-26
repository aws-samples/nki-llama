#!/usr/bin/env python3
"""Calculate training metrics including MFU, NKI score, and final training score from training logs and HLO files."""

import argparse
import json
import os
import re
import glob
import base64
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from neuronx_distributed_training.utils.llama_perf_estimate import calculate_mfu
from torch_neuronx.pyhlo.hlo_pb2 import HloModuleProto


def calculate_training_score(
    base_mfu,
    base_throughput,
    mfu,
    throughput,
    nki_flop_ratio,
    loss_improvement=1.0,
    convergence_rate=1.0
):
    """
    Calculate training score similar to inference calculate_score function.
    
    Args:
        base_mfu: Baseline Model FLOPs Utilization percentage
        base_throughput: Baseline throughput in sequences/second
        mfu: Achieved Model FLOPs Utilization percentage
        throughput: Achieved throughput in sequences/second
        nki_flop_ratio: Ratio of NKI MACs to total MACs
        loss_improvement: Optional - ratio of baseline loss to achieved loss at same step
        convergence_rate: Optional - ratio of steps to reach target loss (baseline/achieved)
    
    Returns:
        float: Final training score
    """
    
    # Calculate improvement ratios
    mfu_improvement = mfu / base_mfu if base_mfu > 0 else 1.0
    throughput_improvement = throughput / base_throughput if base_throughput > 0 else 1.0
    
    # Combine metrics into final score
    # Similar formula to inference but adapted for training metrics
    final_score = mfu_improvement * throughput_improvement * loss_improvement * convergence_rate * (0.1 + nki_flop_ratio)
    
    return {
        'score': final_score,
        'mfu_improvement': mfu_improvement,
        'throughput_improvement': throughput_improvement,
        'loss_improvement': loss_improvement,
        'convergence_rate': convergence_rate,
        'nki_flop_ratio': nki_flop_ratio
    }


def calculate_per_file_training_score(
    file_metrics: Dict,
    base_mfu: float,
    base_throughput: float,
    achieved_mfu: float,
    achieved_throughput: float,
    loss_improvement: float = 1.0,
    convergence_rate: float = 1.0
) -> Dict:
    """
    Calculate training score for a single file based on its NKI ratio.
    
    Args:
        file_metrics: Dictionary containing file analysis results
        base_mfu: Baseline MFU percentage
        base_throughput: Baseline throughput
        achieved_mfu: Achieved MFU percentage
        achieved_throughput: Achieved throughput
        loss_improvement: Loss improvement ratio
        convergence_rate: Convergence rate improvement
    
    Returns:
        Dictionary with score details
    """
    if file_metrics['status'] != 'success':
        return {
            'score': 0.0,
            'error': file_metrics.get('error', 'File analysis failed')
        }
    
    nki_ratio = file_metrics['nki_ratio']
    score_details = calculate_training_score(
        base_mfu=base_mfu,
        base_throughput=base_throughput,
        mfu=achieved_mfu,
        throughput=achieved_throughput,
        nki_flop_ratio=nki_ratio,
        loss_improvement=loss_improvement,
        convergence_rate=convergence_rate
    )
    
    return score_details


def parse_hlo_file(hlo_file_path: str) -> HloModuleProto:
    """Parse HLO file and return protobuf."""
    # Check if file exists
    if not os.path.exists(hlo_file_path):
        raise FileNotFoundError(f"HLO file not found: {hlo_file_path}")
    
    with open(hlo_file_path, 'rb') as f:
        hlo_data = f.read()
    
    hlo_proto = HloModuleProto()
    hlo_proto.ParseFromString(hlo_data)
    return hlo_proto


def count_mac_operations(hlo_proto: HloModuleProto) -> Tuple[int, int]:
    """Count MAC operations in HLO proto.
    
    Returns:
        Tuple of (total_mac_count, nki_mac_count)
    """
    nki_mac = 0
    hlo_mac = 0
    
    for computation in hlo_proto.computations:
        instruction_map = {instr.id: instr for instr in computation.instructions}
        
        for instruction in computation.instructions:
            # Finding NKI ops
            if instruction.opcode == "custom-call":
                if instruction.custom_call_target == 'AwsNeuronCustomNativeKernel':
                    try:
                        backend_config = instruction.backend_config
                        config = json.loads(base64.b64decode(backend_config))
                        mac_count = int(config.get('mac_count', 0))
                    except Exception as e:
                        print(f"Warning: Failed to parse NKI backend config: {e}")
                        mac_count = 0
                    
                    nki_mac += mac_count
                    hlo_mac += mac_count
            elif instruction.opcode == "dot":
                # Get dot dimension numbers
                dnums = instruction.dot_dimension_numbers
                
                # Get shapes of operands using operand_ids
                try:
                    lhs_shape = instruction_map[instruction.operand_ids[0]].shape
                    rhs_shape = instruction_map[instruction.operand_ids[1]].shape
                    
                    # Initialize counters
                    lhs_batch = 1
                    lhs_contracting_size = 1
                    lhs_non_contracting_size = 1
                    rhs_non_contracting_size = 1
                    
                    # Process LHS shape
                    for i in range(len(lhs_shape.dimensions)):
                        if i in dnums.lhs_contracting_dimensions:
                            lhs_contracting_size *= lhs_shape.dimensions[i]
                        elif i in dnums.lhs_batch_dimensions:
                            lhs_batch *= lhs_shape.dimensions[i]
                        else:
                            lhs_non_contracting_size *= lhs_shape.dimensions[i]
                    
                    # Process RHS shape
                    for i in range(len(rhs_shape.dimensions)):
                        if i not in dnums.rhs_contracting_dimensions and \
                           i not in dnums.rhs_batch_dimensions:
                            rhs_non_contracting_size *= rhs_shape.dimensions[i]
                    
                    mac_count = (lhs_batch * lhs_non_contracting_size *
                                lhs_contracting_size * rhs_non_contracting_size)
                    hlo_mac += mac_count
                except Exception as e:
                    print(f"Warning: Failed to process dot operation: {e}")
    
    return hlo_mac, nki_mac


def find_all_hlo_files(compile_dir: str) -> List[str]:
    """Find all HLO module files in the neuron cache directory."""
    hlo_files = []
    
    # Convert to Path object for easier manipulation
    base_path = Path(compile_dir)
    
    # Find all .hlo_module.pb files recursively
    hlo_patterns = [
        #"*.hlo",
        #"graph.hlo",
        #"model.hlo_module.pb",
        "**/*.hlo_module.pb",
        "**/model.hlo_module.pb",
        "**/*.hlo",
        "**/graph.hlo"
    ]
    
    for pattern in hlo_patterns:
        found_files = list(base_path.glob(pattern))
        hlo_files.extend([str(f) for f in found_files])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in hlo_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    return unique_files


def get_module_info(hlo_file_path: str) -> Dict[str, str]:
    """Extract module information from HLO file path."""
    path_parts = Path(hlo_file_path).parts
    module_info = {
        'file_path': hlo_file_path,
        'module_name': None,
        'neuronxcc_version': None
    }
    
    # Extract module name and neuronxcc version from path
    for i, part in enumerate(path_parts):
        if part.startswith('MODULE_'):
            module_info['module_name'] = part
        elif part.startswith('neuronxcc-'):
            module_info['neuronxcc_version'] = part
    
    return module_info


def parse_training_logs(log_file: str) -> Dict:
    """Parse training logs to extract throughput, loss, and timestamp information."""
    metrics = {
        'steps': [],
        'step_times': [],
        'throughputs': [],
        'losses': [],
        'timestamps': [],
        'first_timestamp': None,
        'last_timestamp': None,
        'log_file_path': log_file,
        'epochs': [],
        'learning_rates': [],
        'gradient_norms': [],
        'consumed_samples': [],
        'throughput_peaks': []
    }
    
    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} not found")
        return metrics
    
    # Get file modification time as a fallback timestamp
    file_stat = os.stat(log_file)
    file_mod_time = datetime.fromtimestamp(file_stat.st_mtime)
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Updated patterns for the specific log format
    patterns = {
        # Timestamp pattern: [2025-06-10 09:37:36.116: or [2025-06-10 09:37:36,288]
        'timestamp': r'\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})[.,]\d+[:\]]',
        # Progress bar pattern - matches the actual format in the logs
        'progress_bar': r'Epoch\s+(\d+):\s*\d+%\|[█▌\s]+\|\s*\d+/\d+.*?reduced_train_loss=([\d.]+).*?lr=([\d.]+).*?global_step=([\d.]+).*?consumed_samples=([\d.]+).*?throughput=([\d.]+).*?throughput_peak=([\d.]+).*?gradient_norm=([\d.]+)',
        # Alternative individual patterns
        'epoch_alt': r'Epoch\s+(\d+):',
        'throughput': r'throughput=([\d.]+)',
        'loss': r'reduced_train_loss=([\d.]+)',
        'global_step': r'global_step=([\d.]+)',
        'learning_rate': r'lr=([\d.]+)',
        'gradient_norm': r'gradient_norm=([\d.]+)',
        'consumed_samples': r'consumed_samples=([\d.]+)',
        'throughput_peak': r'throughput_peak=([\d.]+)',
        # Step time if present
        'step_time': r'step_time[:\s]+([\d.]+)'
    }
    
    for line in lines:
        # Check for timestamps
        timestamp_match = re.search(patterns['timestamp'], line)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            if timestamp not in metrics['timestamps']:
                metrics['timestamps'].append(timestamp)
                if metrics['first_timestamp'] is None:
                    metrics['first_timestamp'] = timestamp
                metrics['last_timestamp'] = timestamp
        
        # Try to extract from progress bar format first
        progress_match = re.search(patterns['progress_bar'], line)
        if progress_match:
            epoch = int(progress_match.group(1))
            loss = float(progress_match.group(2))
            lr = float(progress_match.group(3))
            step = int(float(progress_match.group(4)))
            samples = float(progress_match.group(5))
            throughput = float(progress_match.group(6))
            throughput_peak = float(progress_match.group(7))
            grad_norm = float(progress_match.group(8))
            
            # Only add unique values
            if epoch not in metrics['epochs']:
                metrics['epochs'].append(epoch)
            if step not in metrics['steps']:
                metrics['steps'].append(step)
            
            # For metrics that can vary, we want all values
            metrics['losses'].append(loss)
            metrics['learning_rates'].append(lr)
            metrics['consumed_samples'].append(samples)
            metrics['throughputs'].append(throughput)
            metrics['throughput_peaks'].append(throughput_peak)
            metrics['gradient_norms'].append(grad_norm)
        else:
            # Fall back to individual pattern matching if progress bar doesn't match
            
            # Epoch (alternative pattern)
            epoch_match = re.search(patterns['epoch_alt'], line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                if epoch not in metrics['epochs']:
                    metrics['epochs'].append(epoch)
            
            # Look for individual metrics in lines that might not have the full progress bar
            # Only process lines that seem to contain metrics (avoid duplicate processing)
            if 'reduced_train_loss=' in line:
                # Loss
                loss_match = re.search(patterns['loss'], line)
                if loss_match:
                    loss = float(loss_match.group(1))
                    metrics['losses'].append(loss)
                
                # Learning rate
                lr_match = re.search(patterns['learning_rate'], line)
                if lr_match:
                    lr = float(lr_match.group(1))
                    metrics['learning_rates'].append(lr)
                
                # Global step
                step_match = re.search(patterns['global_step'], line)
                if step_match:
                    step = int(float(step_match.group(1)))
                    if step not in metrics['steps']:
                        metrics['steps'].append(step)
                
                # Consumed samples
                samples_match = re.search(patterns['consumed_samples'], line)
                if samples_match:
                    samples = float(samples_match.group(1))
                    metrics['consumed_samples'].append(samples)
                
                # Throughput
                throughput_match = re.search(patterns['throughput'], line)
                if throughput_match:
                    throughput = float(throughput_match.group(1))
                    metrics['throughputs'].append(throughput)
                
                # Throughput peak
                peak_match = re.search(patterns['throughput_peak'], line)
                if peak_match:
                    peak = float(peak_match.group(1))
                    metrics['throughput_peaks'].append(peak)
                
                # Gradient norm
                grad_match = re.search(patterns['gradient_norm'], line)
                if grad_match:
                    grad_norm = float(grad_match.group(1))
                    metrics['gradient_norms'].append(grad_norm)
            
            # Step time (if present)
            step_time_match = re.search(patterns['step_time'], line)
            if step_time_match:
                step_time = float(step_time_match.group(1))
                metrics['step_times'].append(step_time)
    
    # Remove duplicates from lists while preserving order for some metrics
    # For epochs and steps, we want unique values
    metrics['epochs'] = sorted(list(set(metrics['epochs'])))
    metrics['steps'] = sorted(list(set(metrics['steps'])))
    
    # For other metrics, remove consecutive duplicates but keep the progression
    def remove_consecutive_duplicates(lst):
        if not lst:
            return []
        result = [lst[0]]
        for i in range(1, len(lst)):
            if lst[i] != lst[i-1]:
                result.append(lst[i])
        return result
    
    metrics['losses'] = remove_consecutive_duplicates(metrics['losses'])
    metrics['learning_rates'] = remove_consecutive_duplicates(metrics['learning_rates'])
    metrics['consumed_samples'] = remove_consecutive_duplicates(metrics['consumed_samples'])
    metrics['throughputs'] = remove_consecutive_duplicates(metrics['throughputs'])
    metrics['throughput_peaks'] = remove_consecutive_duplicates(metrics['throughput_peaks'])
    metrics['gradient_norms'] = remove_consecutive_duplicates(metrics['gradient_norms'])
    
    # If no timestamps found in logs, use file modification time
    if not metrics['timestamps']:
        metrics['file_modification_time'] = file_mod_time.strftime('%Y-%m-%d %H:%M:%S')
    
    return metrics


def display_log_metrics(log_metrics: Dict) -> None:
    """Display parsed log metrics in a formatted way."""
    print("\n" + "="*60)
    print("LOG FILE METRICS SUMMARY")
    print("="*60)
    
    if log_metrics['log_file_path']:
        print(f"Log file: {log_metrics['log_file_path']}")
    
    # Display timestamp information
    if log_metrics['first_timestamp'] or log_metrics['last_timestamp']:
        print(f"\nTraining Timeline:")
        if log_metrics['first_timestamp']:
            print(f"  First timestamp: {log_metrics['first_timestamp']}")
        if log_metrics['last_timestamp']:
            print(f"  Last timestamp: {log_metrics['last_timestamp']}")
        if log_metrics['first_timestamp'] and log_metrics['last_timestamp']:
            # Try to calculate duration
            try:
                first_dt = datetime.strptime(log_metrics['first_timestamp'], '%Y-%m-%d %H:%M:%S')
                last_dt = datetime.strptime(log_metrics['last_timestamp'], '%Y-%m-%d %H:%M:%S')
                duration = last_dt - first_dt
                print(f"  Duration: {duration}")
            except:
                pass
    elif 'file_modification_time' in log_metrics:
        print(f"\nLog file last modified: {log_metrics['file_modification_time']}")
    
    # Display epoch information
    if log_metrics['epochs']:
        print(f"\nEpoch Information:")
        print(f"  Epochs recorded: {len(log_metrics['epochs'])}")
        print(f"  First epoch: {min(log_metrics['epochs'])}")
        print(f"  Last epoch: {max(log_metrics['epochs'])}")
    
    # Display step information
    if log_metrics['steps']:
        print(f"\nTraining Steps:")
        print(f"  Total steps recorded: {len(log_metrics['steps'])}")
        print(f"  First step: {min(log_metrics['steps'])}")
        print(f"  Last step: {max(log_metrics['steps'])}")
    
    # Display consumed samples
    if log_metrics['consumed_samples']:
        print(f"\nConsumed Samples:")
        print(f"  First: {min(log_metrics['consumed_samples']):,.0f}")
        print(f"  Last: {max(log_metrics['consumed_samples']):,.0f}")
        print(f"  Total processed: {max(log_metrics['consumed_samples']) - min(log_metrics['consumed_samples']):,.0f}")
    
    # Display step time statistics
    if log_metrics['step_times']:
        avg_step_time = sum(log_metrics['step_times']) / len(log_metrics['step_times'])
        print(f"\nStep Time Statistics:")
        print(f"  Average: {avg_step_time:.3f} seconds")
        print(f"  Min: {min(log_metrics['step_times']):.3f} seconds")
        print(f"  Max: {max(log_metrics['step_times']):.3f} seconds")
        print(f"  Number of measurements: {len(log_metrics['step_times'])}")
    
    # Display throughput statistics
    if log_metrics['throughputs']:
        avg_throughput = sum(log_metrics['throughputs']) / len(log_metrics['throughputs'])
        print(f"\nThroughput Statistics:")
        print(f"  Average: {avg_throughput:.3f} seq/s")
        print(f"  Min: {min(log_metrics['throughputs']):.3f} seq/s")
        print(f"  Max: {max(log_metrics['throughputs']):.3f} seq/s")
        print(f"  Number of measurements: {len(log_metrics['throughputs'])}")
    
    # Display throughput peak statistics
    if log_metrics['throughput_peaks']:
        avg_peak = sum(log_metrics['throughput_peaks']) / len(log_metrics['throughput_peaks'])
        print(f"\nThroughput Peak Statistics:")
        print(f"  Average: {avg_peak:.3f} seq/s")
        print(f"  Min: {min(log_metrics['throughput_peaks']):.3f} seq/s")
        print(f"  Max: {max(log_metrics['throughput_peaks']):.3f} seq/s")
    
    # Display loss statistics
    if log_metrics['losses']:
        avg_loss = sum(log_metrics['losses']) / len(log_metrics['losses'])
        print(f"\nLoss Statistics:")
        print(f"  Average: {avg_loss:.4f}")
        print(f"  Min: {min(log_metrics['losses']):.4f}")
        print(f"  Max: {max(log_metrics['losses']):.4f}")
        print(f"  First loss: {log_metrics['losses'][0]:.4f}")
        print(f"  Last loss: {log_metrics['losses'][-1]:.4f}")
        print(f"  Number of measurements: {len(log_metrics['losses'])}")
        
        # Check if loss is decreasing
        if len(log_metrics['losses']) > 1:
            loss_trend = "decreasing" if log_metrics['losses'][-1] < log_metrics['losses'][0] else "increasing"
            print(f"  Loss trend: {loss_trend}")
    
    # Display learning rate statistics
    if log_metrics['learning_rates']:
        print(f"\nLearning Rate Statistics:")
        print(f"  First: {log_metrics['learning_rates'][0]:.6f}")
        print(f"  Last: {log_metrics['learning_rates'][-1]:.6f}")
        print(f"  Min: {min(log_metrics['learning_rates']):.6f}")
        print(f"  Max: {max(log_metrics['learning_rates']):.6f}")
    
    # Display gradient norm statistics
    if log_metrics['gradient_norms']:
        avg_grad = sum(log_metrics['gradient_norms']) / len(log_metrics['gradient_norms'])
        print(f"\nGradient Norm Statistics:")
        print(f"  Average: {avg_grad:.4f}")
        print(f"  Min: {min(log_metrics['gradient_norms']):.4f}")
        print(f"  Max: {max(log_metrics['gradient_norms']):.4f}")
    
    print("="*60)


def analyze_hlo_file(hlo_file: str) -> Dict:
    """Analyze a single HLO file and return its metrics."""
    try:
        module_info = get_module_info(hlo_file)
        
        # Parse the HLO file
        hlo_proto = parse_hlo_file(hlo_file)
        
        # Count MAC operations
        hlo_mac, nki_mac = count_mac_operations(hlo_proto)
        
        # Calculate NKI ratio for this file
        nki_ratio = nki_mac / hlo_mac if hlo_mac > 0 else 0.0
        
        return {
            'status': 'success',
            'file_path': hlo_file,
            'file_name': os.path.basename(hlo_file),
            'module_name': module_info['module_name'],
            'neuronxcc_version': module_info['neuronxcc_version'],
            'hlo_macs': hlo_mac,
            'nki_macs': nki_mac,
            'nki_ratio': nki_ratio
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'file_path': hlo_file,
            'file_name': os.path.basename(hlo_file),
            'error': str(e),
            'hlo_macs': 0,
            'nki_macs': 0,
            'nki_ratio': 0.0
        }


def analyze_all_hlo_files(hlo_files: List[str], score_params: Dict = None) -> Dict:
    """Analyze all HLO files individually and return per-file metrics with optional scoring."""
    per_file_metrics = []
    successful_analyses = 0
    
    for hlo_file in hlo_files:
        print(f"Analyzing: {hlo_file}")
        file_metrics = analyze_hlo_file(hlo_file)
        
        # Calculate per-file score if parameters provided
        if score_params and file_metrics['status'] == 'success':
            file_score = calculate_per_file_training_score(
                file_metrics=file_metrics,
                base_mfu=score_params['base_mfu'],
                base_throughput=score_params['base_throughput'],
                achieved_mfu=score_params['achieved_mfu'],
                achieved_throughput=score_params['achieved_throughput'],
                loss_improvement=score_params.get('loss_improvement', 1.0),
                convergence_rate=score_params.get('convergence_rate', 1.0)
            )
            file_metrics['training_score'] = file_score
        
        per_file_metrics.append(file_metrics)
        
        if file_metrics['status'] == 'success':
            successful_analyses += 1
        else:
            print(f"  Error: {file_metrics['error']}")
    
    # Calculate summary statistics
    successful_files = [m for m in per_file_metrics if m['status'] == 'success']
    
    if successful_files:
        total_hlo_macs = sum(m['hlo_macs'] for m in successful_files)
        total_nki_macs = sum(m['nki_macs'] for m in successful_files)
        overall_nki_ratio = total_nki_macs / total_hlo_macs if total_hlo_macs > 0 else 0.0
        
        nki_ratios = [m['nki_ratio'] for m in successful_files if m['nki_ratio'] > 0]
        avg_nki_ratio = sum(nki_ratios) / len(nki_ratios) if nki_ratios else 0.0
        min_nki_ratio = min(nki_ratios) if nki_ratios else 0.0
        max_nki_ratio = max(nki_ratios) if nki_ratios else 0.0
        
        # Calculate score statistics if scores exist
        if score_params:
            scores = [m['training_score']['score'] for m in successful_files if 'training_score' in m]
            if scores:
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
            else:
                avg_score = min_score = max_score = 0.0
        else:
            avg_score = min_score = max_score = None
    else:
        total_hlo_macs = 0
        total_nki_macs = 0
        overall_nki_ratio = 0.0
        avg_nki_ratio = 0.0
        min_nki_ratio = 0.0
        max_nki_ratio = 0.0
        avg_score = min_score = max_score = None
    
    summary = {
        'total_files': len(hlo_files),
        'successful_analyses': successful_analyses,
        'failed_analyses': len(hlo_files) - successful_analyses,
        'total_hlo_macs': total_hlo_macs,
        'total_nki_macs': total_nki_macs,
        'overall_nki_ratio': overall_nki_ratio,
        'average_nki_ratio': avg_nki_ratio,
        'min_nki_ratio': min_nki_ratio,
        'max_nki_ratio': max_nki_ratio
    }
    
    if score_params:
        summary['average_score'] = avg_score
        summary['min_score'] = min_score
        summary['max_score'] = max_score
    
    return {
        'per_file_metrics': per_file_metrics,
        'summary': summary
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate training metrics post-training")
    parser.add_argument("--config", required=True, help="Path to the training config YAML file")
    parser.add_argument("--model-config", required=True, help="Path to the model config.json file")
    parser.add_argument("--compile-dir", default="/home/ubuntu/neuron_cache", help="Neuron compile cache directory")
    parser.add_argument("--log-file", help="Training log file to parse for metrics")
    parser.add_argument("--batch-size", type=int, help="Global batch size (overrides config)")
    parser.add_argument("--seq-len", type=int, help="Sequence length (overrides config)")
    parser.add_argument("--throughput", type=float, help="Throughput in sequences/second (if known)")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--hw-backend", choices=['trn1', 'trn2'], default='trn1', help="Hardware backend")
    parser.add_argument("--output", default="training_metrics.json", help="Output metrics file")
    parser.add_argument("--detailed", action="store_true", help="Include detailed per-file metrics in output")
    parser.add_argument("--print-per-file", action="store_true", help="Print per-file metrics to console")
    
    # Training score parameters
    score_group = parser.add_argument_group('Training Score Parameters')
    score_group.add_argument(
        "--base-mfu",
        type=float,
        default=50.0,
        help="Baseline MFU percentage for score calculation"
    )
    score_group.add_argument(
        "--base-throughput",
        type=float,
        default=100.0,
        help="Baseline throughput (seq/s) for score calculation"
    )
    score_group.add_argument(
        "--loss-improvement",
        type=float,
        default=1.0,
        help="Loss improvement ratio (baseline_loss/achieved_loss at same step)"
    )
    score_group.add_argument(
        "--convergence-rate",
        type=float,
        default=1.0,
        help="Convergence rate improvement (baseline_steps/achieved_steps to target loss)"
    )
    score_group.add_argument(
        "--calculate-score",
        action="store_true",
        help="Calculate final training score"
    )
    score_group.add_argument(
        "--per-file-scores",
        action="store_true",
        help="Calculate training scores for each file individually"
    )
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override config values if provided
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = cfg['data']['global_batch_size']
    
    if args.seq_len:
        seq_len = args.seq_len
    else:
        seq_len = cfg['data']['seq_length']
    
    # Parse log file and display metrics if provided
    log_metrics = None
    if args.log_file:
        log_metrics = parse_training_logs(args.log_file)
        display_log_metrics(log_metrics)
    
    # Determine throughput
    if args.throughput:
        throughput = args.throughput
    elif args.log_file and log_metrics and log_metrics['throughputs']:
        throughput = sum(log_metrics['throughputs']) / len(log_metrics['throughputs'])
        print(f"\nUsing average throughput from log file: {throughput:.2f} seq/s")
    else:
        print("Warning: No throughput information provided, using default")
        throughput = 100.0  # Default estimate
    
    print(f"\nCalculating metrics with:")
    print(f"  Model config: {args.model_config}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Throughput: {throughput} seq/s")
    print(f"  Hardware: {args.hw_backend}")
    print(f"  Nodes: {args.num_nodes}")
    
    # Calculate MFU
    mfu, seq_per_second, throughput_per_node, tflops_per_second, time_per_batch = calculate_mfu(
        config_path=args.model_config,
        batch_size=batch_size,
        throughput=throughput,
        num_nodes=args.num_nodes,
        seq_len=seq_len,
        hw_backend=args.hw_backend
    )
    
    # Find and analyze all HLO files
    print(f"\nSearching for HLO files in {args.compile_dir}...")
    hlo_files = find_all_hlo_files(args.compile_dir)
    print(f"Found {len(hlo_files)} HLO files")
    
    # Prepare score parameters if needed
    score_params = None
    if args.calculate_score and args.per_file_scores:
        score_params = {
            'base_mfu': args.base_mfu,
            'base_throughput': args.base_throughput,
            'achieved_mfu': mfu,
            'achieved_throughput': throughput,
            'loss_improvement': args.loss_improvement,
            'convergence_rate': args.convergence_rate
        }
    
    # Analyze HLO files individually
    hlo_analysis = analyze_all_hlo_files(hlo_files, score_params)
    
    # Print per-file information if requested
    if args.print_per_file:
        print("\nPer-file NKI analysis:")
        print("-" * 120)
        if args.per_file_scores and args.calculate_score:
            print(f"{'File Name':<40} {'Module':<20} {'HLO MACs':>15} {'NKI MACs':>15} {'NKI Ratio':>10} {'Score':>10}")
        else:
            print(f"{'File Name':<40} {'Module':<20} {'HLO MACs':>15} {'NKI MACs':>15} {'NKI Ratio':>10}")
        print("-" * 120)
        
        for file_metrics in hlo_analysis['per_file_metrics']:
            if file_metrics['status'] == 'success':
                base_info = (f"{file_metrics['file_name']:<40} "
                           f"{(file_metrics['module_name'] or 'N/A'):<20} "
                           f"{file_metrics['hlo_macs']:>15,} "
                           f"{file_metrics['nki_macs']:>15,} "
                           f"{file_metrics['nki_ratio']:>10.4f}")
                
                if 'training_score' in file_metrics:
                    print(f"{base_info} {file_metrics['training_score']['score']:>10.4f}")
                else:
                    print(base_info)
            else:
                print(f"{file_metrics['file_name']:<40} ERROR: {file_metrics['error']}")
        print("-" * 120)
        
        # Print score statistics if available
        if args.per_file_scores and args.calculate_score and hlo_analysis['summary'].get('average_score') is not None:
            print(f"\nPer-file Score Statistics:")
            print(f"  Average Score: {hlo_analysis['summary']['average_score']:.4f}")
            print(f"  Min Score: {hlo_analysis['summary']['min_score']:.4f}")
            print(f"  Max Score: {hlo_analysis['summary']['max_score']:.4f}")
    
    # Compile metrics
    metrics = {
        "model_config": args.model_config,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "num_nodes": args.num_nodes,
        "hardware_backend": args.hw_backend,
        "throughput_seq_per_sec": throughput,
        "mfu_percent": mfu,
        "tflops_per_second": tflops_per_second,
        "throughput_tflops_per_node": throughput_per_node,
        "seq_per_second_per_node": seq_per_second,
        "time_per_batch_seconds": time_per_batch,
        "nki_analysis": {
            "summary": hlo_analysis['summary']
        }
    }
    
    # Add log metrics to output if available
    if log_metrics:
        metrics["log_metrics"] = {
            "log_file": log_metrics['log_file_path'],
            "timestamps": {
                "first": log_metrics['first_timestamp'],
                "last": log_metrics['last_timestamp'],
                "file_modification": log_metrics.get('file_modification_time')
            },
            "steps": {
                "count": len(log_metrics['steps']),
                "first": min(log_metrics['steps']) if log_metrics['steps'] else None,
                "last": max(log_metrics['steps']) if log_metrics['steps'] else None
            },
            "step_times": {
                "average": sum(log_metrics['step_times']) / len(log_metrics['step_times']) if log_metrics['step_times'] else None,
                "min": min(log_metrics['step_times']) if log_metrics['step_times'] else None,
                "max": max(log_metrics['step_times']) if log_metrics['step_times'] else None,
                "count": len(log_metrics['step_times'])
            },
            "throughput": {
                "average": sum(log_metrics['throughputs']) / len(log_metrics['throughputs']) if log_metrics['throughputs'] else None,
                "min": min(log_metrics['throughputs']) if log_metrics['throughputs'] else None,
                "max": max(log_metrics['throughputs']) if log_metrics['throughputs'] else None,
                "count": len(log_metrics['throughputs'])
            },
            "losses": {
                "average": sum(log_metrics['losses']) / len(log_metrics['losses']) if log_metrics['losses'] else None,
                "min": min(log_metrics['losses']) if log_metrics['losses'] else None,
                "max": max(log_metrics['losses']) if log_metrics['losses'] else None,
                "first": log_metrics['losses'][0] if log_metrics['losses'] else None,
                "last": log_metrics['losses'][-1] if log_metrics['losses'] else None,
                "count": len(log_metrics['losses'])
            }
        }
    
    # Add detailed per-file metrics if requested
    if args.detailed:
        metrics["nki_analysis"]["per_file_metrics"] = hlo_analysis['per_file_metrics']
    
    # Calculate overall training score if requested
    if args.calculate_score:
        nki_ratio = hlo_analysis['summary']['overall_nki_ratio']
        
        print('\nTraining Score Breakdown:')
        print('=' * 50)
        print(f'MFU improvement: {mfu/args.base_mfu:.4f} ({args.base_mfu:.2f}% → {mfu:.2f}%)')
        print(f'Throughput improvement: {throughput/args.base_throughput:.4f} ({args.base_throughput:.2f} → {throughput:.2f} seq/s)')
        print(f'Loss improvement: {args.loss_improvement:.4f}')
        print(f'Convergence rate improvement: {args.convergence_rate:.4f}')
        print(f'NKI flop ratio: {nki_ratio:.4f}')
        print('=' * 50)
        
        score_details = calculate_training_score(
            base_mfu=args.base_mfu,
            base_throughput=args.base_throughput,
            mfu=mfu,
            throughput=throughput,
            nki_flop_ratio=nki_ratio,
            loss_improvement=args.loss_improvement,
            convergence_rate=args.convergence_rate
        )
        
        score = score_details['score']
        print(f'Final Training Score: {score:.4f}')
        
        metrics['training_score'] = score
        metrics['training_score_breakdown'] = {
            'base_mfu': args.base_mfu,
            'base_throughput': args.base_throughput,
            'achieved_mfu': mfu,
            'achieved_throughput': throughput,
            'mfu_improvement': score_details['mfu_improvement'],
            'throughput_improvement': score_details['throughput_improvement'],
            'nki_flop_ratio': nki_ratio,
            'loss_improvement': args.loss_improvement,
            'convergence_rate': args.convergence_rate
        }
    
    # Save metrics
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING METRICS SUMMARY")
    print("="*50)
    print(f"MFU: {mfu:.2f}%")
    print(f"TFLOPs/second: {tflops_per_second:.2f}")
    print(f"\nNKI Analysis Summary:")
    print(f"  Files analyzed: {hlo_analysis['summary']['successful_analyses']}/{hlo_analysis['summary']['total_files']}")
    print(f"  Overall NKI Ratio: {hlo_analysis['summary']['overall_nki_ratio']:.4f}")
    print(f"  Average NKI Ratio: {hlo_analysis['summary']['average_nki_ratio']:.4f}")
    print(f"  Min NKI Ratio: {hlo_analysis['summary']['min_nki_ratio']:.4f}")
    print(f"  Max NKI Ratio: {hlo_analysis['summary']['max_nki_ratio']:.4f}")
    print(f"  Total HLO MACs: {hlo_analysis['summary']['total_hlo_macs']:,}")
    print(f"  Total NKI MACs: {hlo_analysis['summary']['total_nki_macs']:,}")
    print(f"\nThroughput: {throughput:.2f} seq/s")
    print(f"Throughput per node: {throughput_per_node:.2f} TFLOP/s")
    print(f"\nMetrics saved to: {args.output}")
    
    return metrics


if __name__ == "__main__":
    main()
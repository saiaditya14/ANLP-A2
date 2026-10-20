# -*- coding: utf-8 -*-

import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


def evaluate_with_profiling(model, dataset, batch_size=32, device='cuda'):
    """
    Drop-in replacement for your evaluate_model() function.
    Adds GPU memory tracking and inference latency measurement.
    
    Args:
        model: PyTorch model to evaluate
        dataset: Dataset with 'input_ids', 'attention_mask', and 'labels'
        batch_size: Batch size for evaluation
        device: Device for inference (will be converted to torch.device)
        
    Returns:
        dict: Metrics including accuracy, report, confusion_matrix, 
              gpu_memory_mb, and inference_latency_ms
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []
    batch_times = []
    
    # Track GPU memory before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        for batch in dl:
            # Move batch to device
            b = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Time the inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            # Run inference (supports both CausalLM and SequenceClassification)
            outputs = model(input_ids=b['input_ids'], attention_mask=b['attention_mask'])
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_times.append(end_time - start_time)
            
            # Extract predictions
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = b['labels'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate memory usage
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = 0.0
    
    # Calculate latency metrics
    total_time = sum(batch_times)
    total_samples = len(all_labels)
    avg_latency_per_sample_ms = (total_time / total_samples) * 1000  # Convert to ms
    
    # Calculate accuracy and other metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': float(acc),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'num_samples': int(total_samples),
        'gpu_memory_mb': float(peak_memory_mb),
        'inference_latency_ms': float(avg_latency_per_sample_ms),
        'total_inference_time_s': float(total_time)
    }


def print_performance_summary(metrics, model_name="Model"):
    """
    Print a nicely formatted performance summary.
    
    Args:
        metrics: Dictionary returned by evaluate_with_profiling()
        model_name: Name to display for this model
    """
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY: {model_name}")
    print(f"{'='*70}")
    print(f"{'Metric':<35} {'Value':<20}")
    print(f"{'-'*70}")
    print(f"{'Accuracy':<35} {metrics['accuracy']:<20.4f}")
    print(f"{'GPU Memory Usage (MB)':<35} {metrics['gpu_memory_mb']:<20.2f}")
    print(f"{'Inference Latency (ms/sample)':<35} {metrics['inference_latency_ms']:<20.3f}")
    print(f"{'Total Samples':<35} {metrics['num_samples']:<20}")
    print(f"{'Total Inference Time (s)':<35} {metrics['total_inference_time_s']:<20.2f}")
    print(f"{'='*70}\n")


def create_comparison_table(results_dict, output_file=None):
    """
    Create a comparison table from multiple model results.
    
    Args:
        results_dict: Dict of {model_name: metrics_dict}
        output_file: Optional path to save the table as text
        
    Example:
        results = {
            "FP32 Baseline": metrics_fp32,
            "INT8 Scratch": metrics_int8_scratch,
            "INT8 BnB": metrics_int8_bnb,
            "NF4 BnB": metrics_nf4_bnb
        }
        create_comparison_table(results, "comparison_table.txt")
    """
    header = f"{'Model':<25} {'GPU Memory (MB)':<20} {'Latency (ms)':<20} {'Accuracy':<15}"
    separator = "="*80
    
    lines = [separator, header, separator]
    
    for model_name, metrics in results_dict.items():
        line = (f"{model_name:<25} "
                f"{metrics['gpu_memory_mb']:<20.2f} "
                f"{metrics['inference_latency_ms']:<20.3f} "
                f"{metrics['accuracy']:<15.4f}")
        lines.append(line)
    
    lines.append(separator)
    
    table_str = "\n".join(lines)
    print("\n" + table_str)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(table_str)
        print(f"\n✅ Comparison table saved to {output_file}")
    
    return table_str


# Example usage within your existing code:
if __name__ == "__main__":
    print("""Hi""")
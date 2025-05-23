import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import json
from typing import Dict, List, Tuple
import seaborn as sns

class DetailedKVAnalysis:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B"):
        """Initialize with Llama 1B model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model loaded with {self.num_layers} layers")
    
    def custom_quantize_tensor(self, tensor: torch.Tensor, bits: int = 4, group_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Custom quantization function"""
        # Reshape for group quantization
        original_shape = tensor.shape
        flat_tensor = tensor.view(-1, group_size)
        
        # Calculate per-group scales and zero points
        max_vals = torch.max(flat_tensor, dim=1, keepdim=True)[0]
        min_vals = torch.min(flat_tensor, dim=1, keepdim=True)[0]
        
        # Quantization parameters
        qmax = 2**bits - 1
        scale = (max_vals - min_vals) / qmax
        zero_point = -min_vals / scale
        
        # Quantize
        quantized = torch.round(flat_tensor / scale + zero_point)
        quantized = torch.clamp(quantized, 0, qmax)
        
        return quantized.view(original_shape), scale.view(-1), zero_point.view(-1)
    
    def dequantize_tensor(self, quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, original_shape: tuple, group_size: int = 64) -> torch.Tensor:
        """Dequantize tensor"""
        flat_quantized = quantized.view(-1, group_size)
        scale_expanded = scale.unsqueeze(1).expand(-1, group_size)
        zero_point_expanded = zero_point.unsqueeze(1).expand(-1, group_size)
        
        dequantized = (flat_quantized - zero_point_expanded) * scale_expanded
        return dequantized.view(original_shape)
    
    def extract_kv_during_generation(self, prompt: str, max_new_tokens: int = 20) -> Dict:
        """Extract KV cache states during generation"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Storage for KV states
        layer_kv_states = {i: {"keys": [], "values": [], "step": []} for i in range(self.num_layers)}
        
        # Use dynamic cache to capture states
        past_key_values = DynamicCache()
        
        generated_tokens = []
        current_inputs = inputs
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    **current_inputs,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
            
            # Extract KV states from cache
            for layer_idx in range(self.num_layers):
                if layer_idx < len(past_key_values.key_cache):
                    keys = past_key_values.key_cache[layer_idx].clone().detach()
                    values = past_key_values.value_cache[layer_idx].clone().detach()
                    
                    layer_kv_states[layer_idx]["keys"].append(keys)
                    layer_kv_states[layer_idx]["values"].append(values)
                    layer_kv_states[layer_idx]["step"].append(step)
            
            # Get next token
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            
            # Prepare next input
            current_inputs = {"input_ids": next_token.unsqueeze(0)}
            
            # Break if EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Generate final text
        full_sequence = torch.cat([inputs.input_ids[0], torch.tensor(generated_tokens).to(self.device)])
        generated_text = self.tokenizer.decode(full_sequence, skip_special_tokens=True)
        
        return {
            "kv_states": layer_kv_states,
            "generated_text": generated_text,
            "generated_tokens": generated_tokens
        }
    
    def calculate_quantization_errors_per_layer(self, kv_data: Dict, bits_list: List[int] = [2, 4, 8]) -> Dict:
        """Calculate quantization errors for each layer, each bit setting, and K vs V"""
        results = {
            "layer_analysis": {},
            "summary_stats": {},
            "bit_comparison": {}
        }
        
        for layer_idx in range(self.num_layers):
            layer_results = {
                "key_errors": {bits: [] for bits in bits_list},
                "value_errors": {bits: [] for bits in bits_list},
                "key_stats": {},
                "value_stats": {}
            }
            
            # Get final KV states for this layer
            if layer_idx in kv_data["kv_states"] and len(kv_data["kv_states"][layer_idx]["keys"]) > 0:
                final_keys = kv_data["kv_states"][layer_idx]["keys"][-1]  # Last step
                final_values = kv_data["kv_states"][layer_idx]["values"][-1]
                
                # Analyze different bit settings
                for bits in bits_list:
                    # Quantize keys
                    quant_keys, key_scale, key_zero = self.custom_quantize_tensor(final_keys, bits)
                    dequant_keys = self.dequantize_tensor(quant_keys, key_scale, key_zero, final_keys.shape)
                    
                    # Quantize values  
                    quant_values, val_scale, val_zero = self.custom_quantize_tensor(final_values, bits)
                    dequant_values = self.dequantize_tensor(quant_values, val_scale, val_zero, final_values.shape)
                    
                    # Calculate errors
                    key_mse = torch.nn.functional.mse_loss(final_keys.float(), dequant_keys.float()).item()
                    key_mae = torch.nn.functional.l1_loss(final_keys.float(), dequant_keys.float()).item()
                    
                    value_mse = torch.nn.functional.mse_loss(final_values.float(), dequant_values.float()).item()
                    value_mae = torch.nn.functional.l1_loss(final_values.float(), dequant_values.float()).item()
                    
                    layer_results["key_errors"][bits] = {"mse": key_mse, "mae": key_mae}
                    layer_results["value_errors"][bits] = {"mse": value_mse, "mae": value_mae}
                
                # Calculate statistics for this layer
                layer_results["key_stats"] = {
                    "mean": final_keys.mean().item(),
                    "std": final_keys.std().item(),
                    "min": final_keys.min().item(),
                    "max": final_keys.max().item(),
                    "shape": list(final_keys.shape)
                }
                
                layer_results["value_stats"] = {
                    "mean": final_values.mean().item(),
                    "std": final_values.std().item(),
                    "min": final_values.min().item(),
                    "max": final_values.max().item(),
                    "shape": list(final_values.shape)
                }
            
            results["layer_analysis"][layer_idx] = layer_results
        
        # Calculate summary statistics
        for bits in bits_list:
            key_mses = [results["layer_analysis"][i]["key_errors"][bits]["mse"] 
                       for i in range(self.num_layers) 
                       if bits in results["layer_analysis"][i]["key_errors"]]
            
            value_mses = [results["layer_analysis"][i]["value_errors"][bits]["mse"] 
                         for i in range(self.num_layers) 
                         if bits in results["layer_analysis"][i]["value_errors"]]
            
            results["summary_stats"][f"{bits}bit"] = {
                "avg_key_mse": np.mean(key_mses) if key_mses else 0,
                "avg_value_mse": np.mean(value_mses) if value_mses else 0,
                "max_key_mse": np.max(key_mses) if key_mses else 0,
                "max_value_mse": np.max(value_mses) if value_mses else 0,
                "std_key_mse": np.std(key_mses) if key_mses else 0,
                "std_value_mse": np.std(value_mses) if value_mses else 0
            }
        
        return results
    
    def create_detailed_visualizations(self, analysis_results: Dict, prompt_info: Dict, save_prefix: str = "detailed_analysis"):
        """Create comprehensive visualizations"""
        
        # Setup the plotting
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # Create a 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Extract data for plotting
        bits_list = [2, 4, 8]
        layer_indices = list(range(self.num_layers))
        
        # 1. Layer-wise Key MSE comparison
        ax1 = fig.add_subplot(gs[0, 0])
        for bits in bits_list:
            key_mses = [analysis_results["layer_analysis"][i]["key_errors"][bits]["mse"] 
                       for i in layer_indices if bits in analysis_results["layer_analysis"][i]["key_errors"]]
            ax1.plot(layer_indices[:len(key_mses)], key_mses, marker='o', label=f'{bits}-bit', linewidth=2)
        ax1.set_title('Key Cache MSE by Layer', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('MSE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Layer-wise Value MSE comparison  
        ax2 = fig.add_subplot(gs[0, 1])
        for bits in bits_list:
            value_mses = [analysis_results["layer_analysis"][i]["value_errors"][bits]["mse"] 
                         for i in layer_indices if bits in analysis_results["layer_analysis"][i]["value_errors"]]
            ax2.plot(layer_indices[:len(value_mses)], value_mses, marker='s', label=f'{bits}-bit', linewidth=2)
        ax2.set_title('Value Cache MSE by Layer', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('MSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Keys vs Values comparison (bar chart)
        ax3 = fig.add_subplot(gs[0, 2])
        x = np.arange(len(bits_list))
        width = 0.35
        
        key_avgs = [analysis_results["summary_stats"][f"{bits}bit"]["avg_key_mse"] for bits in bits_list]
        value_avgs = [analysis_results["summary_stats"][f"{bits}bit"]["avg_value_mse"] for bits in bits_list]
        
        ax3.bar(x - width/2, key_avgs, width, label='Keys', alpha=0.8, color='skyblue')
        ax3.bar(x + width/2, value_avgs, width, label='Values', alpha=0.8, color='lightcoral')
        
        ax3.set_title('Average MSE: Keys vs Values', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Quantization Bits')
        ax3.set_ylabel('Average MSE')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{bits}-bit' for bits in bits_list])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Heatmap of errors across layers and bits
        ax4 = fig.add_subplot(gs[1, :])
        
        # Create error matrix [bits, layers]
        key_error_matrix = np.zeros((len(bits_list), self.num_layers))
        value_error_matrix = np.zeros((len(bits_list), self.num_layers))
        
        for i, bits in enumerate(bits_list):
            for layer_idx in range(self.num_layers):
                if bits in analysis_results["layer_analysis"][layer_idx]["key_errors"]:
                    key_error_matrix[i, layer_idx] = analysis_results["layer_analysis"][layer_idx]["key_errors"][bits]["mse"]
                    value_error_matrix[i, layer_idx] = analysis_results["layer_analysis"][layer_idx]["value_errors"][bits]["mse"]
        
        # Combined heatmap
        combined_matrix = key_error_matrix + value_error_matrix
        
        im = ax4.imshow(combined_matrix, cmap='YlOrRd', aspect='auto')
        ax4.set_title('Combined Key + Value MSE Heatmap (by Layer and Quantization)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Layer Index')
        ax4.set_ylabel('Quantization Bits')
        ax4.set_yticks(range(len(bits_list)))
        ax4.set_yticklabels([f'{bits}-bit' for bits in bits_list])
        plt.colorbar(im, ax=ax4, label='Combined MSE')
        
        # 5. Distribution of errors
        ax5 = fig.add_subplot(gs[2, 0])
        all_key_errors = []
        all_value_errors = []
        for layer_idx in range(self.num_layers):
            for bits in bits_list:
                if bits in analysis_results["layer_analysis"][layer_idx]["key_errors"]:
                    all_key_errors.append(analysis_results["layer_analysis"][layer_idx]["key_errors"][bits]["mse"])
                    all_value_errors.append(analysis_results["layer_analysis"][layer_idx]["value_errors"][bits]["mse"])
        
        ax5.hist(all_key_errors, bins=20, alpha=0.7, label='Keys', color='skyblue')
        ax5.hist(all_value_errors, bins=20, alpha=0.7, label='Values', color='lightcoral')
        ax5.set_title('Distribution of MSE Errors', fontsize=14, fontweight='bold')
        ax5.set_xlabel('MSE Value')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Error by quantization bits
        ax6 = fig.add_subplot(gs[2, 1])
        
        box_data_keys = []
        box_data_values = []
        labels = []
        
        for bits in bits_list:
            key_errors_for_bits = [analysis_results["layer_analysis"][i]["key_errors"][bits]["mse"] 
                                  for i in range(self.num_layers) 
                                  if bits in analysis_results["layer_analysis"][i]["key_errors"]]
            value_errors_for_bits = [analysis_results["layer_analysis"][i]["value_errors"][bits]["mse"] 
                                    for i in range(self.num_layers) 
                                    if bits in analysis_results["layer_analysis"][i]["value_errors"]]
            
            box_data_keys.extend(key_errors_for_bits)
            box_data_values.extend(value_errors_for_bits)
            labels.extend([f'K-{bits}bit'] * len(key_errors_for_bits))
            labels.extend([f'V-{bits}bit'] * len(value_errors_for_bits))
        
        # Combine for box plot
        combined_data = box_data_keys + box_data_values
        
        # Create box plot data
        bp_data = []
        bp_labels = []
        for bits in bits_list:
            key_data = [analysis_results["layer_analysis"][i]["key_errors"][bits]["mse"] 
                       for i in range(self.num_layers) 
                       if bits in analysis_results["layer_analysis"][i]["key_errors"]]
            value_data = [analysis_results["layer_analysis"][i]["value_errors"][bits]["mse"] 
                         for i in range(self.num_layers) 
                         if bits in analysis_results["layer_analysis"][i]["value_errors"]]
            
            bp_data.extend([key_data, value_data])
            bp_labels.extend([f'K-{bits}bit', f'V-{bits}bit'])
        
        ax6.boxplot(bp_data, labels=bp_labels)
        ax6.set_title('Error Distribution by Quantization', fontsize=14, fontweight='bold')
        ax6.set_ylabel('MSE')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # 7. Summary statistics table
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        # Create summary table
        summary_data = []
        for bits in bits_list:
            stats = analysis_results["summary_stats"][f"{bits}bit"]
            summary_data.append([
                f"{bits}-bit",
                f"{stats['avg_key_mse']:.6f}",
                f"{stats['avg_value_mse']:.6f}",
                f"{stats['max_key_mse']:.6f}",
                f"{stats['max_value_mse']:.6f}"
            ])
        
        table = ax7.table(cellText=summary_data,
                         colLabels=['Bits', 'Avg Key MSE', 'Avg Value MSE', 'Max Key MSE', 'Max Value MSE'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax7.set_title('Summary Statistics', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'KV Cache Quantization Analysis\nPrompt: "{prompt_info["prompt"][:50]}..."', 
                    fontsize=16, fontweight='bold')
        
        plt.savefig(f"{save_prefix}_complete_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self, prompts: List[str] = None) -> Dict:
        """Run complete analysis on given prompts"""
        if prompts is None:
            prompts = [
                "The future of artificial intelligence is",
                "In a world where technology advances rapidly, we must consider"
            ]
        
        all_results = {}
        
        for i, prompt in enumerate(prompts):
            print(f"\n=== Analyzing Prompt {i+1} ===")
            print(f"Prompt: {prompt}")
            
            # Extract KV data during generation
            print("Extracting KV cache data during generation...")
            kv_data = self.extract_kv_during_generation(prompt)
            
            print(f"Generated text: {kv_data['generated_text']}")
            
            # Analyze quantization errors
            print("Calculating quantization errors...")
            analysis_results = self.calculate_quantization_errors_per_layer(kv_data)
            
            # Create visualizations
            print("Creating visualizations...")
            self.create_detailed_visualizations(
                analysis_results, 
                {"prompt": prompt, "generated_text": kv_data['generated_text']},
                f"prompt_{i+1}_analysis"
            )
            
            # Print summary
            print("\n=== SUMMARY ===")
            for bits in [2, 4, 8]:
                stats = analysis_results["summary_stats"][f"{bits}bit"]
                print(f"{bits}-bit quantization:")
                print(f"  Average Key MSE: {stats['avg_key_mse']:.8f}")
                print(f"  Average Value MSE: {stats['avg_value_mse']:.8f}")
                print(f"  Max Key MSE: {stats['max_key_mse']:.8f}")
                print(f"  Max Value MSE: {stats['max_value_mse']:.8f}")
            
            all_results[f"prompt_{i+1}"] = {
                "prompt": prompt,
                "generated_text": kv_data['generated_text'], 
                "kv_data": kv_data,
                "analysis": analysis_results
            }
        
        # Save complete results
        with open("complete_kv_analysis_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        return all_results

def main():
    """Run the detailed KV analysis"""
    analyzer = DetailedKVAnalysis()
    
    # Define test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly, we must consider"
    ]
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(test_prompts)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Files generated:")
    print("- prompt_*_analysis_complete_analysis.png (visualizations)")
    print("- complete_kv_analysis_results.json (detailed results)")
    
    return results

if __name__ == "__main__":
    results = main() 

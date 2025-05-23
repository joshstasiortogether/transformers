import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantizedCacheConfig
from transformers.cache_utils import DynamicCache, QuantoQuantizedCache
import seaborn as sns
from typing import Dict, List, Tuple
import json

class KVQuantizationAnalyzer:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B"):
        """Initialize the analyzer with a Llama 1B model"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model loaded with {self.num_layers} layers")
        
    def prepare_prompts(self) -> List[str]:
        """Prepare test prompts for analysis"""
        prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly, we must consider"
        ]
        return prompts
    
    def run_with_cache_type(self, prompt: str, cache_type: str = "dynamic", cache_config=None, max_new_tokens: int = 50):
        """Run inference with specified cache type and capture KV states"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Configure cache
        if cache_type == "quantized":
            if cache_config is None:
                cache_config = QuantizedCacheConfig(
                    backend="quanto",
                    nbits=4,
                    q_group_size=64,
                    residual_length=32,
                    device=self.device
                )
            past_key_values = QuantoQuantizedCache(cache_config)
        else:
            past_key_values = DynamicCache()
        
        # Track KV states at each layer
        kv_states = {i: {"keys": [], "values": []} for i in range(self.num_layers)}
        
        # Hook to capture KV states
        def capture_kv_hook(layer_idx):
            def hook(module, input, output):
                if hasattr(output, 'past_key_values') or len(output) > 2:
                    # Extract past_key_values from output
                    if hasattr(output, 'past_key_values'):
                        pkv = output.past_key_values
                    elif len(output) > 2:
                        pkv = output[2]  # past_key_values is usually the 3rd element
                    else:
                        return
                    
                    if pkv is not None and len(pkv) > layer_idx:
                        if hasattr(pkv, 'key_cache') and hasattr(pkv, 'value_cache'):
                            # DynamicCache or QuantizedCache
                            if len(pkv.key_cache) > layer_idx:
                                kv_states[layer_idx]["keys"].append(pkv.key_cache[layer_idx].clone().detach())
                                kv_states[layer_idx]["values"].append(pkv.value_cache[layer_idx].clone().detach())
                        elif isinstance(pkv, (list, tuple)) and len(pkv) > layer_idx:
                            # Legacy cache format
                            layer_cache = pkv[layer_idx]
                            if isinstance(layer_cache, (list, tuple)) and len(layer_cache) >= 2:
                                kv_states[layer_idx]["keys"].append(layer_cache[0].clone().detach())
                                kv_states[layer_idx]["values"].append(layer_cache[1].clone().detach())
            return hook
        
        # Register hooks
        hooks = []
        for i, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(capture_kv_hook(i))
            hooks.append(hook)
        
        try:
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_hidden_states=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return {
            "kv_states": kv_states,
            "generated_text": generated_text,
            "cache": outputs.past_key_values if hasattr(outputs, 'past_key_values') else past_key_values
        }
    
    def calculate_quantization_error(self, original_kv: Dict, quantized_kv: Dict) -> Dict:
        """Calculate quantization error metrics for each layer and K/V"""
        errors = {
            "layer_errors": {},
            "summary": {"key_errors": [], "value_errors": []}
        }
        
        for layer_idx in range(self.num_layers):
            layer_errors = {"key_mse": [], "key_mae": [], "value_mse": [], "value_mae": []}
            
            # Get the final KV states for this layer
            orig_keys = original_kv[layer_idx]["keys"]
            orig_values = original_kv[layer_idx]["values"]
            quant_keys = quantized_kv[layer_idx]["keys"]
            quant_values = quantized_kv[layer_idx]["values"]
            
            if len(orig_keys) > 0 and len(quant_keys) > 0:
                # Use the last (final) state for comparison
                orig_k = orig_keys[-1]
                orig_v = orig_values[-1]
                quant_k = quant_keys[-1] if len(quant_keys) > 0 else orig_k
                quant_v = quant_values[-1] if len(quant_values) > 0 else orig_v
                
                # Handle quantized cache format (might need dequantization)
                if hasattr(quant_k, 'dequantize'):
                    quant_k = quant_k.dequantize()
                if hasattr(quant_v, 'dequantize'):
                    quant_v = quant_v.dequantize()
                
                # Ensure same shape and device
                if orig_k.shape == quant_k.shape:
                    # Calculate errors for keys
                    key_mse = F.mse_loss(orig_k.float(), quant_k.float()).item()
                    key_mae = F.l1_loss(orig_k.float(), quant_k.float()).item()
                    
                    layer_errors["key_mse"].append(key_mse)
                    layer_errors["key_mae"].append(key_mae)
                    errors["summary"]["key_errors"].append(key_mse)
                
                if orig_v.shape == quant_v.shape:
                    # Calculate errors for values
                    value_mse = F.mse_loss(orig_v.float(), quant_v.float()).item()
                    value_mae = F.l1_loss(orig_v.float(), quant_v.float()).item()
                    
                    layer_errors["value_mse"].append(value_mse)
                    layer_errors["value_mae"].append(value_mae)
                    errors["summary"]["value_errors"].append(value_mse)
            
            # Average errors for this layer
            errors["layer_errors"][layer_idx] = {
                "key_mse": np.mean(layer_errors["key_mse"]) if layer_errors["key_mse"] else 0,
                "key_mae": np.mean(layer_errors["key_mae"]) if layer_errors["key_mae"] else 0,
                "value_mse": np.mean(layer_errors["value_mse"]) if layer_errors["value_mse"] else 0,
                "value_mae": np.mean(layer_errors["value_mae"]) if layer_errors["value_mae"] else 0,
            }
        
        return errors
    
    def analyze_quantization_impact(self, prompts: List[str], quantization_configs: List[Dict] = None):
        """Complete analysis of quantization impact"""
        if quantization_configs is None:
            quantization_configs = [
                {"backend": "quanto", "nbits": 2, "q_group_size": 64, "residual_length": 32},
                {"backend": "quanto", "nbits": 4, "q_group_size": 64, "residual_length": 32},
                {"backend": "quanto", "nbits": 8, "q_group_size": 64, "residual_length": 32},
            ]
        
        results = {}
        
        for i, prompt in enumerate(prompts):
            print(f"\nAnalyzing prompt {i+1}: '{prompt[:50]}...'")
            
            # Run with original (unquantized) cache
            print("Running with original cache...")
            original_result = self.run_with_cache_type(prompt, "dynamic")
            
            prompt_results = {
                "original_text": original_result["generated_text"],
                "quantization_results": {}
            }
            
            for config in quantization_configs:
                config_name = f"{config['backend']}_{config['nbits']}bit"
                print(f"Running with {config_name} quantization...")
                
                cache_config = QuantizedCacheConfig(**config, device=self.device)
                quantized_result = self.run_with_cache_type(prompt, "quantized", cache_config)
                
                # Calculate errors
                errors = self.calculate_quantization_error(
                    original_result["kv_states"], 
                    quantized_result["kv_states"]
                )
                
                prompt_results["quantization_results"][config_name] = {
                    "generated_text": quantized_result["generated_text"],
                    "errors": errors,
                    "config": config
                }
            
            results[f"prompt_{i+1}"] = prompt_results
        
        return results
    
    def create_error_visualizations(self, results: Dict, save_path: str = "quantization_analysis"):
        """Create comprehensive visualizations of quantization errors"""
        # Extract data for plotting
        configs = list(list(results.values())[0]["quantization_results"].keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("KV Cache Quantization Error Analysis", fontsize=16)
        
        # Plot 1: Layer-wise Key Errors
        for config in configs:
            key_errors = []
            for layer_idx in range(self.num_layers):
                # Average across prompts
                errors = [results[f"prompt_{i+1}"]["quantization_results"][config]["errors"]["layer_errors"][layer_idx]["key_mse"] 
                         for i in range(len(results))]
                key_errors.append(np.mean(errors))
            
            axes[0,0].plot(range(self.num_layers), key_errors, marker='o', label=config)
        
        axes[0,0].set_title("Key Cache MSE by Layer")
        axes[0,0].set_xlabel("Layer Index")
        axes[0,0].set_ylabel("MSE")
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Plot 2: Layer-wise Value Errors
        for config in configs:
            value_errors = []
            for layer_idx in range(self.num_layers):
                errors = [results[f"prompt_{i+1}"]["quantization_results"][config]["errors"]["layer_errors"][layer_idx]["value_mse"] 
                         for i in range(len(results))]
                value_errors.append(np.mean(errors))
            
            axes[0,1].plot(range(self.num_layers), value_errors, marker='s', label=config)
        
        axes[0,1].set_title("Value Cache MSE by Layer")
        axes[0,1].set_xlabel("Layer Index")
        axes[0,1].set_ylabel("MSE")
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Plot 3: Overall Key vs Value Error Comparison
        key_totals = []
        value_totals = []
        config_labels = []
        
        for config in configs:
            key_total = []
            value_total = []
            for prompt_key in results.keys():
                errors = results[prompt_key]["quantization_results"][config]["errors"]["summary"]
                key_total.extend(errors["key_errors"])
                value_total.extend(errors["value_errors"])
            
            key_totals.append(np.mean(key_total))
            value_totals.append(np.mean(value_total))
            config_labels.append(config)
        
        x = np.arange(len(config_labels))
        width = 0.35
        
        axes[1,0].bar(x - width/2, key_totals, width, label='Keys', alpha=0.8)
        axes[1,0].bar(x + width/2, value_totals, width, label='Values', alpha=0.8)
        
        axes[1,0].set_title("Average MSE: Keys vs Values")
        axes[1,0].set_xlabel("Quantization Config")
        axes[1,0].set_ylabel("MSE")
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(config_labels, rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Plot 4: Error Distribution Heatmap
        error_matrix = np.zeros((len(configs), self.num_layers))
        for i, config in enumerate(configs):
            for layer_idx in range(self.num_layers):
                total_error = 0
                count = 0
                for prompt_key in results.keys():
                    layer_errors = results[prompt_key]["quantization_results"][config]["errors"]["layer_errors"][layer_idx]
                    total_error += layer_errors["key_mse"] + layer_errors["value_mse"]
                    count += 1
                error_matrix[i, layer_idx] = total_error / count if count > 0 else 0
        
        im = axes[1,1].imshow(error_matrix, cmap='YlOrRd', aspect='auto')
        axes[1,1].set_title("Combined K+V Error Heatmap")
        axes[1,1].set_xlabel("Layer Index")
        axes[1,1].set_ylabel("Quantization Config")
        axes[1,1].set_yticks(range(len(configs)))
        axes[1,1].set_yticklabels(config_labels)
        plt.colorbar(im, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig(f"{save_path}_visualizations.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results: Dict, save_path: str = "quantization_report"):
        """Generate a comprehensive report"""
        report = {
            "model": self.model_name,
            "num_layers": self.num_layers,
            "analysis_summary": {},
            "detailed_results": results
        }
        
        # Create summary statistics
        for prompt_idx, (prompt_key, prompt_data) in enumerate(results.items()):
            print(f"\n=== PROMPT {prompt_idx + 1} ANALYSIS ===")
            print(f"Original text: {prompt_data['original_text']}")
            print()
            
            for config_name, config_data in prompt_data["quantization_results"].items():
                print(f"--- {config_name.upper()} ---")
                print(f"Generated text: {config_data['generated_text']}")
                
                # Calculate summary stats
                errors = config_data["errors"]
                avg_key_error = np.mean(errors["summary"]["key_errors"]) if errors["summary"]["key_errors"] else 0
                avg_value_error = np.mean(errors["summary"]["value_errors"]) if errors["summary"]["value_errors"] else 0
                
                print(f"Average Key MSE: {avg_key_error:.6f}")
                print(f"Average Value MSE: {avg_value_error:.6f}")
                
                # Find layers with highest errors
                layer_errors = [(i, e["key_mse"] + e["value_mse"]) for i, e in errors["layer_errors"].items()]
                layer_errors.sort(key=lambda x: x[1], reverse=True)
                
                print(f"Top 3 layers with highest errors: {layer_errors[:3]}")
                print()
        
        # Save detailed report
        with open(f"{save_path}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report

def main():
    """Main analysis function"""
    # Initialize analyzer
    analyzer = KVQuantizationAnalyzer()
    
    # Prepare prompts
    prompts = analyzer.prepare_prompts()
    
    # Define quantization configurations to test
    quant_configs = [
        {"backend": "quanto", "nbits": 2, "q_group_size": 64, "residual_length": 32},
        {"backend": "quanto", "nbits": 4, "q_group_size": 64, "residual_length": 32},
    ]
    
    # Run analysis
    print("Starting quantization analysis...")
    results = analyzer.analyze_quantization_impact(prompts, quant_configs)
    
    # Generate visualizations
    print("Creating visualizations...")
    analyzer.create_error_visualizations(results)
    
    # Generate report
    print("Generating report...")
    report = analyzer.generate_report(results)
    
    print("Analysis complete! Check the generated files for detailed results.")
    
    return results, report

if __name__ == "__main__":
    results, report = main() 
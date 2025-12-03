import itertools
import json
import os
import random
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from trainAndEval import ExperimentRunner


class HyperparameterSweep:
    """Manages hyperparameter sweep experiments"""
    def __init__(self, dataset_loader, model_class, device='cpu', output_dir='./results'):
        self.dataset_loader = dataset_loader
        self.model_class = model_class
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def define_search_space(self):
        return {
            'layer_type': ['f', 'g'],
            'activation': ['relu', 'tanh'],
            'pooling': ['mean', 'sum'],
            'num_layers': [2, 3],
            'hidden_dim': [64, 128],
            'dropout': [0.5],
            'lr': [0.01],
        }
    
    def generate_configs(self, search_space, max_configs=None):
        training_keys = ['lr']
        model_keys = [k for k in search_space.keys() if k not in training_keys]
        model_keys_sorted = sorted(model_keys)
        model_combinations = list(itertools.product(
            *[search_space[k] for k in model_keys_sorted]
        ))
        
        configs = []
        for combo in model_combinations:
            model_params = dict(zip(model_keys_sorted, combo))
            for lr in search_space['lr']:
                config = {
                    'model_params': model_params,
                    'training_params': {'lr': lr, 'epochs': 200, 'patience': 50}
                }
                configs.append(config)
        
        if max_configs and len(configs) > max_configs:
            print(f"Total configs ({len(configs)}) exceeds max ({max_configs}). Stratified sampling by layer_type...")
            f_configs = [c for c in configs if c['model_params'].get('layer_type') == 'f']
            g_configs = [c for c in configs if c['model_params'].get('layer_type') == 'g']
            
            random.shuffle(f_configs)
            random.shuffle(g_configs)
            
            half = max_configs // 2
            sampled = f_configs[:half] + g_configs[:half]
            
            if len(sampled) < max_configs:
                remaining = [c for c in configs if c not in sampled]
                random.shuffle(remaining)
                sampled.extend(remaining[:max_configs - len(sampled)])
            
            configs = sampled
    
        return configs
    
    def config_to_name(self, config):
        mp = config['model_params']
        tp = config['training_params']
        name = (f"{mp['layer_type']}_"
                f"{mp['activation']}_"
                f"{mp['pooling']}_"
                f"L{mp['num_layers']}_"
                f"H{mp['hidden_dim']}_"
                f"D{mp['dropout']}_"
                f"lr{tp['lr']}")
        return name
    
    def run_sweep(self, train_loader, val_loader, test_loader, 
                  search_space, seeds=[42, 123, 456], 
                  max_configs=None, dataset_name="dataset"):
        runner = ExperimentRunner(self.model_class, device=self.device)
        configs = self.generate_configs(search_space, max_configs)
        
        print(f"\n{'#'*70}")
        print(f"HYPERPARAMETER SWEEP: {dataset_name}")
        print(f"{'#'*70}")
        print(f"Total configurations: {len(configs)}")
        print(f"Seeds per config: {len(seeds)}")
        print(f"Total experiments: {len(configs) * len(seeds)}")
        print(f"{'#'*70}\n")
        
        for i, config in enumerate(configs, 1):
            config_name = self.config_to_name(config)
            print(f"\n[{i}/{len(configs)}] Testing: {config_name}")
            
            model_params = config['model_params'].copy()
            model_params['input_dim'] = self.dataset_loader.dataset.num_node_features
            model_params['output_dim'] = self.dataset_loader.dataset.num_classes
            
            try:
                runner.run_multiple_seeds(
                    model_params=model_params,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    training_params=config['training_params'],
                    seeds=seeds,
                    config_name=config_name
                )
            except Exception as e:
                print(f"Error in config {config_name}: {e}")
                continue
        
        return runner
    
    def save_results(self, runner, dataset_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path = os.path.join(self.output_dir, f"{dataset_name}_results_{timestamp}.json")
        results_dict = {
            'dataset': dataset_name,
            'timestamp': timestamp,
            'results': {}
        }
        
        for config_name, result in runner.results.items():
            result_copy = result.copy()
            result_copy['model_params'] = str(result_copy['model_params'])
            results_dict['results'][config_name] = result_copy
        
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {json_path}")
        
        csv_path = os.path.join(self.output_dir, f"{dataset_name}_summary_{timestamp}.csv")
        rows = []
        for config_name, result in runner.results.items():
            mp = result['model_params']
            row = {
                'config': config_name,
                'layer_type': mp['layer_type'],
                'activation': mp['activation'],
                'pooling': mp['pooling'],
                'num_layers': mp['num_layers'],
                'hidden_dim': mp['hidden_dim'],
                'dropout': mp['dropout'],
                'val_acc_mean': result['val_acc_mean'],
                'val_acc_std': result['val_acc_std'],
                'test_acc_mean': result['test_acc_mean'],
                'test_acc_std': result['test_acc_std'],
                'test_f1_mean': result['test_f1_mean'],
                'test_f1_std': result['test_f1_std'],
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('val_acc_mean', ascending=False)
        df.to_csv(csv_path, index=False)
        
        print(f"Summary saved to: {csv_path}")
        return json_path, csv_path
    
    def plot_results(self, runner, dataset_name, save=True):
        results = runner.results
        
        if not results:
            print("No results to plot")
            return
        
        configs = list(results.keys())
        test_accs = [results[c]['test_acc_mean'] for c in configs]
        test_stds = [results[c]['test_acc_std'] for c in configs]
        
        sorted_indices = sorted(range(len(test_accs)), 
                              key=lambda i: test_accs[i], 
                              reverse=True)
        configs = [configs[i] for i in sorted_indices]
        test_accs = [test_accs[i] for i in sorted_indices]
        test_stds = [test_stds[i] for i in sorted_indices]
        
        top_n = min(10, len(configs))
        configs = configs[:top_n]
        test_accs = test_accs[:top_n]
        test_stds = test_stds[:top_n]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = range(len(configs))
        
        ax.bar(x_pos, test_accs, yerr=test_stds, capsize=5, alpha=0.7)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'Top {top_n} Configurations - {dataset_name}', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.output_dir, f"{dataset_name}_top_configs.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
        
        plt.show()
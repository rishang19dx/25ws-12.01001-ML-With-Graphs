import torch
import argparse

from model import GNN
from dataLoader import TUDatasetLoader
from hyperparamSweep import HyperparameterSweep


def main():
    parser = argparse.ArgumentParser(description='Run GNN hyperparameter sweep experiments')
    parser.add_argument('--dataset', type=str, default='MUTAG', 
                       help='Dataset name (e.g., MUTAG, IMDB-BINARY)')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='Random seeds for experiments')
    parser.add_argument('--max_configs', type=int, default=None,
                       help='Maximum number of configurations to test (None for all)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda, auto-detect if not specified)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip plotting results')
    
    args = parser.parse_args()
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("GNN HYPERPARAMETER SWEEP EXPERIMENT")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seeds: {args.seeds}")
    print(f"Max configs: {args.max_configs if args.max_configs else 'All'}")
    print("="*70 + "\n")
    
    print("\n[Step 1/4] Loading dataset...")
    print("-" * 70)
    loader = TUDatasetLoader(args.dataset, root=args.data_root, seed=42)
    
    print("\n[Step 2/4] Creating data splits and loaders...")
    print("-" * 70)
    train_dataset, val_dataset, test_dataset = loader.get_splits()
    train_loader, val_loader, test_loader = loader.get_loaders(
        train_dataset, val_dataset, test_dataset, 
        batch_size=args.batch_size
    )
    
    print("\n[Step 3/4] Running hyperparameter sweep...")
    print("-" * 70)
    sweep = HyperparameterSweep(
        dataset_loader=loader,
        model_class=GNN,
        device=device,
        output_dir=args.output_dir
    )
    
    search_space = sweep.define_search_space()
    runner = sweep.run_sweep(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        search_space=search_space,
        seeds=args.seeds,
        max_configs=args.max_configs,
        dataset_name=args.dataset
    )
    
    print("\n[Step 4/4] Saving results and generating plots...")
    print("-" * 70)
    runner.print_all_results()
    sweep.save_results(runner, args.dataset)
    
    if not args.no_plot:
        sweep.plot_results(runner, args.dataset, save=True)
    
    best_name, best_result = runner.get_best_config()
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Best Configuration: {best_name}")
    print(f"  Validation Accuracy: {best_result['val_acc_mean']:.4f} ± {best_result['val_acc_std']:.4f}")
    print(f"  Test Accuracy: {best_result['test_acc_mean']:.4f} ± {best_result['test_acc_std']:.4f}")
    print(f"  Test F1-Score: {best_result['test_f1_mean']:.4f} ± {best_result['test_f1_std']:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


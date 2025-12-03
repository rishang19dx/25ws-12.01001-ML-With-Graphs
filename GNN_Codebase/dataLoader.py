import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant
import numpy as np
from sklearn.model_selection import train_test_split


class TUDatasetLoader:
    """Loader for TUDataset with train/val/test splits and statistics"""
    def __init__(self, dataset_name, root='./data', seed=42):
        self.dataset_name = dataset_name
        self.seed = seed
        
        print(f"\n{'='*60}")
        print(f"Loading dataset: {dataset_name}")
        print(f"{'='*60}")
        
        self.dataset = TUDataset(root=root, name=dataset_name)
        if self.dataset.num_node_features == 0:
            print("Dataset has no node features. Adding constant features...")
            self.dataset.transform = Constant(value=1.0)
        
        self._print_statistics()
        
    def _print_statistics(self):
        data = self.dataset[0]
        
        print(f"\nDataset Statistics:")
        print(f"  Number of graphs: {len(self.dataset)}")
        print(f"  Number of classes: {self.dataset.num_classes}")
        print(f"  Number of node features: {self.dataset.num_node_features}")

        num_nodes = [data.num_nodes for data in self.dataset]
        num_edges = [data.num_edges for data in self.dataset]
        
        print(f"\nGraph Size Statistics:")
        print(f"  Avg nodes per graph: {np.mean(num_nodes):.2f} ± {np.std(num_nodes):.2f}")
        print(f"  Min/Max nodes: {np.min(num_nodes)} / {np.max(num_nodes)}")
        print(f"  Avg edges per graph: {np.mean(num_edges):.2f} ± {np.std(num_edges):.2f}")
        print(f"  Min/Max edges: {np.min(num_edges)} / {np.max(num_edges)}")
        
        if hasattr(data, 'y'):
            labels = [data.y.item() for data in self.dataset]
            unique, counts = np.unique(labels, return_counts=True)
            print(f"\nClass Distribution:")
            for label, count in zip(unique, counts):
                print(f"  Class {label}: {count} ({100*count/len(labels):.1f}%)")
    
    def get_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        labels = np.array([data.y.item() for data in self.dataset])
        indices = np.arange(len(self.dataset))
        
        n_total = len(self.dataset)
        
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            stratify=labels,
            random_state=self.seed
        )
        train_val_labels = labels[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio/(train_ratio + val_ratio),
            stratify=train_val_labels,
            random_state=self.seed
        )
        train_dataset = self.dataset[train_idx.tolist()]
        val_dataset = self.dataset[val_idx.tolist()]
        test_dataset = self.dataset[test_idx.tolist()]
        
        print(f"\nData Splits:")
        print(f"  Train: {len(train_dataset)} graphs ({100*len(train_dataset)/n_total:.1f}%)")
        print(f"  Val:   {len(val_dataset)} graphs ({100*len(val_dataset)/n_total:.1f}%)")
        print(f"  Test:  {len(test_dataset)} graphs ({100*len(test_dataset)/n_total:.1f}%)")
        
        return train_dataset, val_dataset, test_dataset
    
    def get_loaders(self, train_dataset, val_dataset, test_dataset, 
                    batch_size=32, shuffle=True):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\nDataLoaders Created:")
        print(f"  Batch size: {batch_size}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        return train_loader, val_loader, test_loader
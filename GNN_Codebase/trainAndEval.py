import torch
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import time
import copy
from collections import defaultdict


class Trainer:
    """Trainer for GNN models with validation-based early stopping"""
    def __init__(self, model, device='cpu', patience=50):
        self.model = model.to(device)
        self.device = device
        self.patience = patience
        self.best_model_state = None
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
        
        avg_loss = total_loss / len(loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return avg_loss, accuracy, f1
    
    def train(self, train_loader, val_loader, lr=0.01, epochs=200, verbose=True):
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        best_val_acc = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, val_f1 = self.evaluate(val_loader, criterion)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if verbose and (epoch % 20 == 0 or epoch == 1):
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

            if patience_counter >= self.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        if verbose:
            print(f"\nTraining completed in {training_time:.2f}s")
            print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
        
        return {
            'history': history,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'training_time': training_time,
            'total_epochs': epoch
        }
    
    def test(self, test_loader, verbose=True):
        criterion = torch.nn.CrossEntropyLoss()
        test_loss, test_acc, test_f1 = self.evaluate(test_loader, criterion)
        
        if verbose:
            print(f"\nTest Results:")
            print(f"   Loss: {test_loss:.4f}")
            print(f"   Accuracy: {test_acc:.4f}")
            print(f"   F1-Score: {test_f1:.4f}")
        
        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1
        }


class ExperimentRunner:
    """Run experiments across multiple seeds and hyperparameters"""
    def __init__(self, model_class, device='cpu'):
        self.model_class = model_class
        self.device = device
        self.results = defaultdict(list)
    
    def run_single_experiment(self, model_params, train_loader, val_loader, 
                            test_loader, training_params, seed, verbose=False):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Seed {seed}")
            print(f"{'='*60}")
        
        model = self.model_class(**model_params)
        trainer = Trainer(model, device=self.device, patience=training_params.get('patience', 50))
        train_results = trainer.train(
            train_loader, val_loader,
            lr=training_params.get('lr', 0.01),
            epochs=training_params.get('epochs', 200),
            verbose=verbose
        )
        test_results = trainer.test(test_loader, verbose=verbose)

        return {
            'seed': seed,
            'val_acc': train_results['best_val_acc'],
            'test_acc': test_results['test_acc'],
            'test_f1': test_results['test_f1'],
            'best_epoch': train_results['best_epoch'],
            'training_time': train_results['training_time']
        }
    
    def run_multiple_seeds(self, model_params, train_loader, val_loader, 
                          test_loader, training_params, seeds=[42, 123, 456], 
                          config_name="experiment"):
        print(f"\n{'#'*60}")
        print(f"Running: {config_name}")
        print(f"{'#'*60}")
        print(f"Model params: {model_params}")
        print(f"Seeds: {seeds}")
        
        seed_results = []
        for seed in seeds:
            result = self.run_single_experiment(
                model_params, train_loader, val_loader, test_loader,
                training_params, seed, verbose=True
            )
            seed_results.append(result)
        
        test_accs = [r['test_acc'] for r in seed_results]
        test_f1s = [r['test_f1'] for r in seed_results]
        val_accs = [r['val_acc'] for r in seed_results]
        
        aggregated = {
            'config_name': config_name,
            'model_params': model_params,
            'seed_results': seed_results,
            'test_acc_mean': np.mean(test_accs),
            'test_acc_std': np.std(test_accs),
            'test_f1_mean': np.mean(test_f1s),
            'test_f1_std': np.std(test_f1s),
            'val_acc_mean': np.mean(val_accs),
            'val_acc_std': np.std(val_accs),
            'best_val_acc': np.max(val_accs)
        }
        
        self.results[config_name] = aggregated
        
        print(f"\n{'='*60}")
        print(f"Summary for {config_name}")
        print(f"{'='*60}")
        print(f"Val Accuracy:  {aggregated['val_acc_mean']:.4f} ± {aggregated['val_acc_std']:.4f}")
        print(f"Test Accuracy: {aggregated['test_acc_mean']:.4f} ± {aggregated['test_acc_std']:.4f}")
        print(f"Test F1:       {aggregated['test_f1_mean']:.4f} ± {aggregated['test_f1_std']:.4f}")
        
        return aggregated
    
    def get_best_config(self, metric='val_acc_mean'):
        if not self.results:
            return None
        best_config = max(self.results.items(), key=lambda x: x[1][metric])
        return best_config
    
    def print_all_results(self):
        if not self.results:
            print("No results to display")
            return
        
        print(f"\n{'='*80}")
        print("ALL EXPERIMENTAL RESULTS")
        print(f"{'='*80}")
        print(f"{'Configuration':<30} {'Val Acc':<20} {'Test Acc':<20} {'Test F1':<20}")
        print(f"{'-'*80}")
        
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['val_acc_mean'], 
                              reverse=True)
        
        for config_name, results in sorted_results:
            print(f"{config_name:<30} "
                  f"{results['val_acc_mean']:.4f}±{results['val_acc_std']:.4f}      "
                  f"{results['test_acc_mean']:.4f}±{results['test_acc_std']:.4f}      "
                  f"{results['test_f1_mean']:.4f}±{results['test_f1_std']:.4f}")
        
        print(f"{'='*80}\n")
        
        best_name, best_results = sorted_results[0]
        print(f"Best Configuration: {best_name}")
        print(f"   Val Acc:  {best_results['val_acc_mean']:.4f} ± {best_results['val_acc_std']:.4f}")
        print(f"   Test Acc: {best_results['test_acc_mean']:.4f} ± {best_results['test_acc_std']:.4f}")
        print(f"   Test F1:  {best_results['test_f1_mean']:.4f} ± {best_results['test_f1_std']:.4f}")
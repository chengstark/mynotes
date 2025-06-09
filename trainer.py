import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from typing import Dict
import quantization


class QuantAwareTrainer:
    def __init__(self, loaders, epochs=10, patience=3, device=None, multiclass=False):
        self.loaders = loaders
        self.epochs = epochs
        self.patience = patience
        self.multiclass = multiclass
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        total_loss, correct, total = 0, 0, 0
        predictions, labels, probs = [], [], []
        
        for batch in loader:
            X, y = batch['data'].to(self.device), batch['label'].to(self.device)
            
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            predictions.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(torch.softmax(out, dim=-1).cpu().numpy())
        
        acc = correct / total
        f1 = f1_score(labels, predictions, average='macro' if self.multiclass else 'binary')
        
        try:
            if self.multiclass:
                auroc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            else:
                auroc = roc_auc_score(labels, np.array(probs)[:, 1])
        except:
            auroc = acc
            
        return total_loss / len(loader), acc, f1, auroc

    def eval_epoch(self, model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        predictions, labels, probs = [], [], []
        
        with torch.no_grad():
            for batch in loader:
                X, y = batch['data'].to(self.device), batch['label'].to(self.device)
                out = model(X)
                loss = criterion(out, y)
                
                total_loss += loss.item()
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
                predictions.extend(pred.cpu().numpy())
                labels.extend(y.cpu().numpy())
                probs.extend(torch.softmax(out, dim=-1).cpu().numpy())
        
        acc = correct / total
        f1 = f1_score(labels, predictions, average='macro' if self.multiclass else 'binary')
        
        try:
            if self.multiclass:
                auroc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            else:
                auroc = roc_auc_score(labels, np.array(probs)[:, 1])
        except:
            auroc = acc
            
        return total_loss / len(loader), acc, f1, auroc

    def train_and_evaluate(self, model, nn_module=None, qasm_config=None, desc="") -> Dict:
        model = model.to(self.device)
        
        config = quantization.load_config("app_quant_config.yaml")
        calibration_data = next(iter(self.loaders['train_loader']))['data']
        qmodel = quantization.prepare_qat(model, config, calibration_data)
        print("[INFO] Using quantization-aware training")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(qmodel.parameters(), lr=0.001)
        
        best_val_acc = 0
        patience_counter = 0
        best_weights = None
        
        train_metrics = {'losses': [], 'accs': [], 'f1s': [], 'aurocs': []}
        val_metrics = {'losses': [], 'accs': [], 'f1s': [], 'aurocs': []}
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc, train_f1, train_auroc = self.train_epoch(
                qmodel, self.loaders['train_loader'], criterion, optimizer
            )
            train_metrics['losses'].append(train_loss)
            train_metrics['accs'].append(train_acc)
            train_metrics['f1s'].append(train_f1)
            train_metrics['aurocs'].append(train_auroc)

            quantization.convert(qmodel, config)
            
            # Validate
            val_loss, val_acc, val_f1, val_auroc = self.eval_epoch(
                qmodel, self.loaders['val_loader'], criterion
            )
            val_metrics['losses'].append(val_loss)
            val_metrics['accs'].append(val_acc)
            val_metrics['f1s'].append(val_f1)
            val_metrics['aurocs'].append(val_auroc)
            
            print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_weights = qmodel.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        # Load best weights and test
        if best_weights:
            qmodel.load_state_dict(best_weights)
        
        test_loss, test_acc, test_f1, test_auroc = self.eval_epoch(
            qmodel, self.loaders['test_loader'], criterion
        )
        
        score_dict = {
            'acc': test_acc,
            'f1': test_f1,
            'auroc': test_auroc,
            'train_losses': train_metrics['losses'],
            'val_losses': val_metrics['losses'],
            'train_accs': train_metrics['accs'],
            'val_accs': val_metrics['accs'],
            'train_f1s': train_metrics['f1s'],
            'val_f1s': val_metrics['f1s'],
            'train_aurocs': train_metrics['aurocs'],
            'val_aurocs': val_metrics['aurocs'],
            'train_master_scores': [self._calculate_master_metric({'acc': a, 'f1': f, 'auroc': au}) 
                                  for a,f,au in zip(train_metrics['accs'], train_metrics['f1s'], train_metrics['aurocs'])],
            'val_master_scores': [self._calculate_master_metric({'acc': a, 'f1': f, 'auroc': au}) 
                                for a,f,au in zip(val_metrics['accs'], val_metrics['f1s'], val_metrics['aurocs'])]
        }
        score_dict['master'] = self._calculate_master_metric(score_dict)
        return score_dict

    def _calculate_master_metric(self, score_dict):
        """Calculate master metric combining acc, f1, and auroc"""
        return (score_dict['acc'] + score_dict['f1'] + score_dict['auroc']) / 3
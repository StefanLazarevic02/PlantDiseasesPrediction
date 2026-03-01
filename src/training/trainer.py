import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import time
import copy
from pathlib import Path


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        valid_loader,
        config: dict,
        device: str = None
    ):
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        
        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        
        # Loss function (sa class weights za imbalanced dataset)
        class_weights = train_loader.dataset.get_class_weights().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Tracking
        self.history = {
            "train_loss": [], "train_acc": [],
            "valid_loss": [], "valid_acc": [],
            "lr": []
        }
        self.best_acc = 0.0
        self.best_model = None
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.valid_loader, desc="Validation", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, save_dir: str = "outputs"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting training")
        print(f"{'='*60}")
        print(f"Epoch: {self.epochs}")
        print(f"Learning rate: {self.lr}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Train samples: {len(self.train_loader.dataset):,}")
        print(f"Valid samples: {len(self.valid_loader.dataset):,}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            valid_loss, valid_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(valid_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["valid_loss"].append(valid_loss)
            self.history["valid_acc"].append(valid_acc)
            self.history["lr"].append(current_lr)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%")
            print(f"LR: {current_lr:.6f}")
            
            # Save best model
            if valid_acc > self.best_acc:
                self.best_acc = valid_acc
                self.best_model = copy.deepcopy(self.model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': self.best_acc,
                    'config': self.config
                }, save_path / "best_model.pth")
                print(f"New best model saved! (Acc: {valid_acc:.2f}%)")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_acc:.2f}%")
        print(f"{'='*60}")
        
        return self.history
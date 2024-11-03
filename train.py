import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ChineseSentimentClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout=0.3):
        super(ChineseSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        dropout_output = self.dropout(pooled_output)
        return self.linear(dropout_output)

class SentimentTrainer:
    def __init__(self, model_params=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # Default model parameters
        self.model_params = {
            'n_classes': 3,
            'dropout': 0.3,
            'batch_size': 32,
            'max_length': 128,
            'learning_rate': 2e-5,
            'epochs': 5
        }
        
        if model_params:
            self.model_params.update(model_params)
            
        self.model = ChineseSentimentClassifier(
            n_classes=self.model_params['n_classes'],
            dropout=self.model_params['dropout']
        ).to(self.device)
        
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='training.log'
        )
        
    def prepare_data(self, texts, labels):
        """Prepare data for training"""
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = SentimentDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            self.model_params['max_length']
        )
        
        val_dataset = SentimentDataset(
            val_texts,
            val_labels,
            self.tokenizer,
            self.model_params['max_length']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_params['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.model_params['batch_size']
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate the model"""
        self.model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        return (
            val_loss / len(val_loader),
            classification_report(true_labels, predictions, digits=4)
        )
    
    def train(self, texts, labels):
        """Complete training pipeline"""
        train_loader, val_loader = self.prepare_data(texts, labels)
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.model_params['learning_rate']
        )
        
        total_steps = len(train_loader) * self.model_params['epochs']
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=total_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_reports': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(self.model_params['epochs']):
            logging.info(f"Epoch {epoch + 1}/{self.model_params['epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Evaluate
            val_loss, val_report = self.evaluate(val_loader)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_reports'].append(val_report)
            
            logging.info(f"Train Loss: {train_loss:.4f}")
            logging.info(f"Val Loss: {val_loss:.4f}")
            logging.info(f"Validation Report:\n{val_report}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
                logging.info("Saved best model")
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()
    
    def predict(self, texts):
        """Predict sentiment for new texts"""
        self.model.eval()
        predictions = []
        
        for text in texts:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.model_params['max_length'],
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.append(preds.item())
        
        return predictions

def main():
    # Example usage
    
    # 1. Prepare your data
    # This is just example data - replace with your actual labeled dataset
    data = {
        'text': [
            "这个产品非常好用，很满意",
            "服务态度需要改善",
            "一般般，没什么特别的"
        ],
        'label': [2, 0, 1]  # 2: positive, 1: neutral, 0: negative
    }
    df = pd.DataFrame(data)
    
    # 2. Initialize trainer
    trainer = SentimentTrainer(model_params={
        'n_classes': 3,
        'epochs': 3,
        'batch_size': 16
    })
    
    # 3. Train model
    history = trainer.train(df['text'].values, df['label'].values)
    
    # 4. Plot training history
    trainer.plot_training_history(history)
    
    # 5. Make predictions
    new_texts = ["这个新功能很有创意", "质量太差了"]
    predictions = trainer.predict(new_texts)
    
    return predictions

if __name__ == "__main__":
    predictions = main()
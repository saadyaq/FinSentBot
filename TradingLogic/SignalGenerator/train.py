import pandas as pd 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

#Dataset Pytorch

class TradingDataset(Dataset):
    def __init__(self,df):
        self.X=df[['sentiment_score','price_now','price_future','variation']].values.astype(np.float32)
        self.y=df["action"].values

        #Encoder les labels

        self.label_encoder=LabelEncoder()
        self.y=self.label_encoder.fit_tranform(df["action"])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return torch.tensor(self.X[idx],dtype=torch.float32),torch.tensor(self.y[idx],dtype=torch.long)

# Modèle MLP

class TradingMLP(nn.Module):
    def __init__(self,input_dim, hidden_dim=16,output_dim=3):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim)
        )
    def forward (self,x):
        return self.net(x)

#Boucle d'entrainement

def train_model(model ,dataloader,criterion, optimizer, device):
    model.train()
    
    for epoch in range(10):
        total_loss=0
        for x_batch , y_batch in dataloader :
            x_batch, y_batch=x_batch.to(device),y_batch.to(device)

            optimizer.zero_grad()
            outputs=model(x_batch)
            loss=criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()

        print(f"Epoch {epoch +1}, Loss: {total_loss:.4f}")

def evaluate_model(model,dataloader,criterion,device):
    model.eval()
    total_loss=0
    correct=0
    total=0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch,y_batch=x_batch.to(device), y_batch.to(device)
            outputs=model(x_batch)
            loss=criterion(outputs,y_batch)
            total_loss+=loss.item()

            _,predicted=torch.max(outputs.data,1)
            total +=y_batch.size(0)
            correct+=(predicted==y_batch).sum().item()
    
    accuracy=correct/total if total>0 else 0
    avg_loss=total_loss/len(dataloader)

    print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy : {accuracy:.2%}")
    return avg_loss, accuracy
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from .models import LSTMSignalGenerator, SimpleMLP


class TradingDataset(Dataset):
    """Dataset Pytorch pour les données de trading """
    
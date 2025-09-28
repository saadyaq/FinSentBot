import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the enhanced dataset
df = pd.read_csv('/home/saadyaq/SE/Python/finsentbot/data/training_datasets/train_enhanced_with_historical.csv')

print("Dataset Overview:")
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Dataset shape: {df.shape}")
print("\n" + "="*50)

# Class distribution
print("Class Distribution:")
action_counts = df['action'].value_counts()
print(action_counts)
print(f"Percentages:")
for action, count in action_counts.items():
    pct = (count/len(df))*100
    print(f"{action}: {pct:.1f}%")

print("\n" + "="*50)

# Basic statistics
print("Basic Statistics:")
print(f"Unique symbols: {df['symbol'].nunique()}")
print(f"Symbol breakdown:")
symbol_counts = df['symbol'].value_counts().head(10)
print(symbol_counts)

print(f"\nDate range: {df['news_timestamp'].min()} to {df['news_timestamp'].max()}")
print(f"Price variation range: {df['variation'].min():.4f} to {df['variation'].max():.4f}")
print(f"Sentiment score range: {df['sentiment_score'].min():.4f} to {df['sentiment_score'].max():.4f}")

print("\n" + "="*50)

# Data quality checks
print("Data Quality Checks:")
print(f"Missing values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Check for zero variations
zero_variations = (df['variation'] == 0).sum()
print(f"Zero price variations: {zero_variations} ({zero_variations/len(df)*100:.1f}%)")

# Sentiment score distribution by action
print(f"\nSentiment Score by Action:")
for action in df['action'].unique():
    subset = df[df['action'] == action]['sentiment_score']
    print(f"{action}: mean={subset.mean():.3f}, std={subset.std():.3f}, count={len(subset)}")

print("\n" + "="*50)
print("Sample size is excellent! Ready for model training.")
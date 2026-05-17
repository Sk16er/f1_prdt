import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

class F1Dataset(Dataset):
    def __init__(self, X_cat, X_num, y, padding_masks):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.padding_masks = torch.tensor(padding_masks, dtype=torch.bool)

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        return {
            'cat_features': self.X_cat[idx],
            'num_features': self.X_num[idx],
            'labels': self.y[idx],
            'padding_mask': self.padding_masks[idx]
        }

def load_and_preprocess_data(data_dir='archive', max_drivers=55, save_encoders=True):
    """Loads CSVs, merges them, engineers features, and creates padded sequences."""
    try:
        races = pd.read_csv(os.path.join(data_dir, 'races.csv'))
        results = pd.read_csv(os.path.join(data_dir, 'results.csv'))
        drivers = pd.read_csv(os.path.join(data_dir, 'drivers.csv'))
        constructors = pd.read_csv(os.path.join(data_dir, 'constructors.csv'))
        driver_standings = pd.read_csv(os.path.join(data_dir, 'driver_standings.csv'))
        constructor_standings = pd.read_csv(os.path.join(data_dir, 'constructor_standings.csv'))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

    # Merge basic info
    df = pd.merge(results, races[['raceId', 'year', 'circuitId', 'date']], on='raceId', how='left')
    df = pd.merge(df, drivers[['driverId', 'nationality', 'dob']], on='driverId', how='left')
    df = pd.merge(df, constructors[['constructorId', 'nationality']], on='constructorId', how='left', suffixes=('_driver', '_constructor'))

    # Calculate driver age
    df['date'] = pd.to_datetime(df['date'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25

    # Championship standings before each race
    driver_standings_pre_race = driver_standings.rename(columns={'points': 'driver_points', 'position': 'driver_standings_pos'})
    constructor_standings_pre_race = constructor_standings.rename(columns={'points': 'constructor_points', 'position': 'constructor_standings_pos'})

    df = pd.merge(df, driver_standings_pre_race[['raceId', 'driverId', 'driver_points', 'driver_standings_pos']], on=['raceId', 'driverId'], how='left')
    df = pd.merge(df, constructor_standings_pre_race[['raceId', 'constructorId', 'constructor_points', 'constructor_standings_pos']], on=['raceId', 'constructorId'], how='left')

    # Fill NaNs
    df = df.fillna(0)

    # Target
    df['is_winner'] = (df['positionOrder'] == 1).astype(int)

    # Label Encoders
    categorical_features = ['driverId', 'constructorId', 'circuitId']
    encoders = {}
    vocab_sizes = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        # Add a special "Unknown" category (index 0)
        unique_vals = list(df[col].unique())
        le.fit(['<UNK>'] + [str(v) for v in unique_vals])
        df[col] = le.transform(df[col].astype(str))
        encoders[col] = le
        vocab_sizes[col] = len(le.classes_)

    if save_encoders:
        os.makedirs('models', exist_ok=True)
        with open('models/encoders.pkl', 'wb') as f:
            pickle.dump({'encoders': encoders, 'vocab_sizes': vocab_sizes, 'max_drivers': max_drivers}, f)

    numeric_features = ['grid', 'driver_age', 'driver_points', 'driver_standings_pos', 'constructor_points', 'constructor_standings_pos']

    # Group by race
    races_grouped = df.groupby('raceId')
    
    X_cat, X_num, y = [], [], []
    
    for name, group in races_grouped:
        group = group.sort_values('grid')
        
        cat_vals = group[categorical_features].values
        num_vals = group[numeric_features].values
        target_vals = group['is_winner'].values
        
        # Pad sequences up to max_drivers
        num_drivers_in_race = len(group)
        if num_drivers_in_race > max_drivers:
            # truncate if somehow more than max_drivers
            cat_vals = cat_vals[:max_drivers]
            num_vals = num_vals[:max_drivers]
            target_vals = target_vals[:max_drivers]
            num_drivers_in_race = max_drivers
            
        pad_len = max_drivers - num_drivers_in_race
        
        if pad_len > 0:
            cat_pad = np.zeros((pad_len, len(categorical_features)))
            num_pad = np.zeros((pad_len, len(numeric_features)))
            target_pad = np.zeros(pad_len)
            
            cat_vals = np.vstack([cat_vals, cat_pad])
            num_vals = np.vstack([num_vals, num_pad])
            target_vals = np.concatenate([target_vals, target_pad])
            
        X_cat.append(cat_vals)
        X_num.append(num_vals)
        y.append(target_vals)

    X_cat = np.array(X_cat)
    X_num = np.array(X_num)
    y = np.array(y)
    
    # padding_mask: True if it is padding, False if it is a real driver
    # Assuming index 0 for driverId is '<UNK>' which could be padding, or we just check if it's all zeros.
    # Actually, let's use the fact that padding has 0 in driverId.
    padding_masks = (X_cat[:, :, 0] == 0)

    # Split
    indices = np.arange(len(X_cat))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_dataset = F1Dataset(X_cat[idx_train], X_num[idx_train], y[idx_train], padding_masks[idx_train])
    test_dataset = F1Dataset(X_cat[idx_test], X_num[idx_test], y[idx_test], padding_masks[idx_test])

    return train_dataset, test_dataset, vocab_sizes, len(numeric_features)

def get_dataloaders(batch_size=32, data_dir='archive'):
    train_dataset, test_dataset, vocab_sizes, num_features = load_and_preprocess_data(data_dir=data_dir)
    if train_dataset is None:
        return None, None, None, None
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, vocab_sizes, num_features

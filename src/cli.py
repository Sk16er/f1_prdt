import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

from src.data import get_dataloaders
from src.model import F1Predictor
from src.visualize import plot_attention

console = Console()

def train(args):
    console.print(Panel.fit("[bold green]Starting F1 Predictor Training (PyTorch)[/bold green]"))
    
    with console.status("[bold blue]Loading data and preprocessing...[/bold blue]"):
        train_loader, test_loader, vocab_sizes, num_features = get_dataloaders(batch_size=args.batch_size)
    
    if train_loader is None:
        console.print("[bold red]Failed to load data. Make sure 'archive' folder exists with CSV files.[/bold red]")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"Using device: [bold cyan]{device}[/bold cyan]")
    
    model = F1Predictor(vocab_sizes, num_features, embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers)
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss(reduction='none') # reduction='none' to apply padding mask
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_loss = float('inf')
    os.makedirs('models', exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        epoch_task = progress.add_task("[cyan]Training Epochs...", total=args.epochs)
        
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                cat_features = batch['cat_features'].to(device)
                num_features_b = batch['num_features'].to(device)
                labels = batch['labels'].to(device)
                padding_mask = batch['padding_mask'].to(device)
                
                optimizer.zero_grad()
                logits = model(cat_features, num_features_b, padding_mask)
                
                loss = criterion(logits, labels)
                
                # Mask out padding elements from loss
                mask = ~padding_mask.unsqueeze(-1)
                loss = (loss * mask).sum() / mask.sum()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    cat_features = batch['cat_features'].to(device)
                    num_features_b = batch['num_features'].to(device)
                    labels = batch['labels'].to(device)
                    padding_mask = batch['padding_mask'].to(device)
                    
                    logits = model(cat_features, num_features_b, padding_mask)
                    loss = criterion(logits, labels)
                    
                    mask = ~padding_mask.unsqueeze(-1)
                    loss = (loss * mask).sum() / mask.sum()
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(test_loader)
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), 'models/model.pt')
            
            progress.update(epoch_task, advance=1, description=f"[cyan]Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    console.print(Panel.fit(f"[bold green]Training Complete! Best Val Loss: {best_loss:.4f}[/bold green]\nModel saved to models/model.pt"))

def predict(args):
    if not os.path.exists('models/model.pt') or not os.path.exists('models/encoders.pkl'):
        console.print("[bold red]Model files not found. Please run 'python -m src.cli train' first.[/bold red]")
        return
        
    with open('models/encoders.pkl', 'rb') as f:
        encoder_data = pickle.load(f)
        encoders = encoder_data['encoders']
        vocab_sizes = encoder_data['vocab_sizes']
        max_drivers = encoder_data['max_drivers']
        
    # Load raw data for this race
    races = pd.read_csv('archive/races.csv')
    
    if args.year is None or args.name is None:
        console.print("[bold cyan]--- Interactive Race Selection ---[/bold cyan]")
        args.year = IntPrompt.ask("Enter the race year (e.g. 2022)")
        
        available_races = races[races['year'] == args.year].sort_values('round')
        if available_races.empty:
            console.print(f"[bold red]No races found for year {args.year}.[/bold red]")
            return
            
        console.print(f"\n[bold green]Available races in {args.year}:[/bold green]")
        for idx, (_, row) in enumerate(available_races.iterrows()):
            console.print(f"  [cyan]{idx + 1}.[/cyan] {row['name']}")
            
        selection = IntPrompt.ask("\nSelect a race number", choices=[str(i+1) for i in range(len(available_races))])
        
        selected_race = available_races.iloc[selection - 1]
        args.name = selected_race['name']
        console.print(f"\n[bold magenta]Predicting for: {args.name} {args.year}[/bold magenta]\n")

    target_race = races[(races['year'] == args.year) & (races['name'] == args.name)]
    if target_race.empty:
        console.print(f"[bold red]Race '{args.name}' in year {args.year} not found in dataset.[/bold red]")
        return
        
    target_race_id = target_race['raceId'].iloc[0]
    
    results = pd.read_csv('archive/results.csv')
    drivers = pd.read_csv('archive/drivers.csv')
    constructors = pd.read_csv('archive/constructors.csv')
    driver_standings = pd.read_csv('archive/driver_standings.csv')
    constructor_standings = pd.read_csv('archive/constructor_standings.csv')
    
    # Feature engineering for target race
    df = pd.merge(results[results['raceId'] == target_race_id], races[['raceId', 'year', 'circuitId', 'date']], on='raceId', how='left')
    df = pd.merge(df, drivers[['driverId', 'nationality', 'dob', 'driverRef', 'forename', 'surname']], on='driverId', how='left')
    df = pd.merge(df, constructors[['constructorId', 'nationality']], on='constructorId', how='left', suffixes=('_driver', '_constructor'))
    
    df['date'] = pd.to_datetime(df['date'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25
    
    d_standings = driver_standings.rename(columns={'points': 'driver_points', 'position': 'driver_standings_pos'})
    c_standings = constructor_standings.rename(columns={'points': 'constructor_points', 'position': 'constructor_standings_pos'})
    
    df = pd.merge(df, d_standings[['raceId', 'driverId', 'driver_points', 'driver_standings_pos']], on=['raceId', 'driverId'], how='left')
    df = pd.merge(df, c_standings[['raceId', 'constructorId', 'constructor_points', 'constructor_standings_pos']], on=['raceId', 'constructorId'], how='left')
    
    df = df.fillna(0)
    df = df.sort_values('grid')
    
    driver_names = (df['forename'] + " " + df['surname']).tolist()
    
    # Encode
    for col in ['driverId', 'constructorId', 'circuitId']:
        le = encoders[col]
        # map unknown to '<UNK>' (index 0)
        df[col] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        
    num_features_list = ['grid', 'driver_age', 'driver_points', 'driver_standings_pos', 'constructor_points', 'constructor_standings_pos']
    
    cat_vals = df[['driverId', 'constructorId', 'circuitId']].values
    num_vals = df[num_features_list].values
    
    # Pad
    num_drivers_in_race = len(df)
    if num_drivers_in_race > max_drivers:
        cat_vals = cat_vals[:max_drivers]
        num_vals = num_vals[:max_drivers]
        driver_names = driver_names[:max_drivers]
        num_drivers_in_race = max_drivers
        
    pad_len = max_drivers - num_drivers_in_race
    
    if pad_len > 0:
        cat_vals = np.vstack([cat_vals, np.zeros((pad_len, 3))])
        num_vals = np.vstack([num_vals, np.zeros((pad_len, len(num_features_list)))])
        driver_names.extend(['<UNK>'] * pad_len)
        
    X_cat = torch.tensor(cat_vals, dtype=torch.long).unsqueeze(0)
    X_num = torch.tensor(num_vals, dtype=torch.float32).unsqueeze(0)
    padding_mask = (X_cat[:, :, 0] == 0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = F1Predictor(vocab_sizes, len(num_features_list), embed_dim=16) 
    model.load_state_dict(torch.load('models/model.pt', map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        logits, attentions = model(X_cat.to(device), X_num.to(device), padding_mask.to(device), return_attention=True)
        probs = torch.sigmoid(logits).cpu().numpy()[0, :, 0]
        
        # attentions is a list of [batch, num_heads, seq_len, seq_len] or similar, actually PyTorch nn.MultiheadAttention returns (batch, seq_len, seq_len) if average_attn_weights=True
        last_layer_attn = attentions[-1][0].cpu().numpy() # shape (seq_len, seq_len)
        
    # Plot attention
    img_path = plot_attention(last_layer_attn, driver_names, args.name, args.year)
    
    # Display results
    table = Table(title=f"Predicted Probabilities: {args.name} {args.year}")
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    table.add_column("Driver", style="magenta")
    table.add_column("Probability", justify="right", style="green")
    
    results_list = []
    for i, (name, prob) in enumerate(zip(driver_names, probs)):
        if name != '<UNK>':
            results_list.append((name, prob))
            
    results_list.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, prob) in enumerate(results_list[:10]):
        table.add_row(str(i+1), name, f"{prob:.2%}")
        
    console.print(table)
    if img_path:
         console.print(f"[bold yellow]Attention visualization saved to:[/bold yellow] {img_path}")

def main():
    parser = argparse.ArgumentParser(description="F1 Predictor CLI")
    subparsers = parser.add_subparsers(dest='command')
    
    train_parser = subparsers.add_parser('train', help="Train the PyTorch Transformer model")
    train_parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    train_parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    train_parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument('--embed-dim', type=int, default=16, help="Embedding dimension")
    train_parser.add_argument('--num-heads', type=int, default=4, help="Number of attention heads")
    train_parser.add_argument('--num-layers', type=int, default=2, help="Number of transformer layers")
    
    predict_parser = subparsers.add_parser('predict', help="Predict the winner of a specific race")
    predict_parser.add_argument('--year', type=int, help="Race year (optional, will prompt if omitted)")
    predict_parser.add_argument('--name', type=str, help="Race name (optional, will prompt if omitted)")
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

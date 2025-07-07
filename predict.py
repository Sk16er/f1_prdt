import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# --- Configuration ---
MODEL_PATH = 'f1_winner_model_v2.keras'
RACE_YEAR = 2022
RACE_NAME = 'Italian Grand Prix'
MAX_DRIVERS = 55 # This must be the same as used in training

# --- Load Data and Model ---
try:
    model = load_model(MODEL_PATH)
    races = pd.read_csv('archive/races.csv')
    results = pd.read_csv('archive/results.csv')
    drivers = pd.read_csv('archive/drivers.csv')
    constructors = pd.read_csv('archive/constructors.csv')
    driver_standings = pd.read_csv('archive/driver_standings.csv')
    constructor_standings = pd.read_csv('archive/constructor_standings.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure all CSV files are in the 'archive' directory and the model file is present.")
    exit()

# --- Preprocessing ---

# 1. Find the race to predict
target_race = races[(races['year'] == RACE_YEAR) & (races['name'] == RACE_NAME)]
if target_race.empty:
    print(f"Error: Could not find the {RACE_NAME} in year {RACE_YEAR}.")
    exit()
target_race_id = target_race['raceId'].iloc[0]

# 2. Build the full feature set (similar to training)
df = pd.merge(results, races[['raceId', 'year', 'circuitId', 'date']], on='raceId', how='left')
df = pd.merge(df, drivers, on='driverId', how='left')
df = pd.merge(df, constructors, on='constructorId', how='left', suffixes=('_driver', '_constructor'))

df['date'] = pd.to_datetime(df['date'])
df['dob'] = pd.to_datetime(df['dob'])
df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25

driver_standings_pre_race = driver_standings.rename(columns={'points': 'driver_points', 'position': 'driver_standings_pos'})
constructor_standings_pre_race = constructor_standings.rename(columns={'points': 'constructor_points', 'position': 'constructor_standings_pos'})

df = pd.merge(df, driver_standings_pre_race[['raceId', 'driverId', 'driver_points', 'driver_standings_pos']], on=['raceId', 'driverId'], how='left')
df = pd.merge(df, constructor_standings_pre_race[['raceId', 'constructorId', 'constructor_points', 'constructor_standings_pos']], on=['raceId', 'constructorId'], how='left')
df.fillna(0, inplace=True)

# 3. Fit label encoders on the *entire* dataset to ensure consistency
categorical_features = ['driverId', 'constructorId', 'circuitId']
encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 4. Isolate and preprocess the single race for prediction
race_data = df[df['raceId'] == target_race_id].sort_values('grid')

if race_data.empty:
    print(f"No data available for the selected race ({RACE_NAME} {RACE_YEAR}).")
    exit()

# Get original driver names for the final output
original_driver_info = race_data[['driverRef', 'forename', 'surname']].copy()

# Transform the categorical features for the prediction race
for col in categorical_features:
    # Create a mapping from original label to encoded label
    # Handle unseen labels by mapping them to a default value (e.g., 0)
    # This assumes 0 is a valid encoded value or can be treated as 'unknown'
    mapping = {label: encoded for encoded, label in enumerate(encoders[col].classes_)}
    race_data[col] = race_data[col].map(mapping).fillna(0).astype(int)

numeric_features = ['grid', 'driver_age', 'driver_points', 'driver_standings_pos', 'constructor_points', 'constructor_standings_pos']

X_cat_race = race_data[categorical_features].values
X_num_race = race_data[numeric_features].values

# 5. Pad the sequence
X_cat_padded = pad_sequences([X_cat_race], maxlen=MAX_DRIVERS, padding='post')
X_num_padded = pad_sequences([X_num_race], maxlen=MAX_DRIVERS, padding='post', dtype='float32')

# --- Prediction ---
prediction_input = [
    X_cat_padded[:, :, 0],
    X_cat_padded[:, :, 1],
    X_cat_padded[:, :, 2],
    X_num_padded
]

probabilities = model.predict(prediction_input)[0] # Get probs for the first (and only) race

# Find predicted winner
predicted_winner_index = np.argmax(probabilities)

# Ensure the predicted_winner_index is within the bounds of actual drivers
if predicted_winner_index >= len(original_driver_info):
    # This means the model predicted a padded position as winner.
    # This indicates a potential issue with the model or data.
    # For now, we'll set it to the driver with the highest probability among actual drivers.
    actual_driver_probabilities = probabilities[:len(original_driver_info)]
    predicted_winner_index = np.argmax(actual_driver_probabilities)

predicted_driver_ref = original_driver_info.iloc[predicted_winner_index]['driverRef']
predicted_driver_name = f"{original_driver_info.iloc[predicted_winner_index]['forename']} {original_driver_info.iloc[predicted_winner_index]['surname']}"

# --- Get Actual Winner ---
actual_winner_info = results[(results['raceId'] == target_race_id) & (results['positionOrder'] == 1)]
actual_driver_id = actual_winner_info['driverId'].iloc[0]
actual_driver_info = drivers[drivers['driverId'] == actual_driver_id]
actual_driver_name = f"{actual_driver_info['forename'].iloc[0]} {actual_driver_info['surname'].iloc[0]}"

# --- Display Results ---
print(f"\n--- F1 Race Winner Prediction ---")
print(f"Race: {RACE_NAME} {RACE_YEAR}")
print(f"\nPredicted Winner: {predicted_driver_name} (Prob: {probabilities[predicted_winner_index][0]:.2%})")
print(f"Actual Winner:    {actual_driver_name}")

# Optional: Print top 5 predictions
print("\nTop 5 Predictions:")
# Filter top_5_indices to only include indices corresponding to actual drivers
valid_top_5_indices = [i for i in np.argsort(probabilities.flatten())[::-1] if i < len(original_driver_info)]
for i in valid_top_5_indices[:5]: # Take only up to 5 valid indices
    driver_name = f"{original_driver_info.iloc[i]['forename']} {original_driver_info.iloc[i]['surname']}"
    prob = probabilities[i][0]
    print(f"  - {driver_name}: {prob:.2%}")

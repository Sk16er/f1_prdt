
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load the datasets
try:
    races = pd.read_csv('archive/races.csv')
    results = pd.read_csv('archive/results.csv')
    drivers = pd.read_csv('archive/drivers.csv')
    constructors = pd.read_csv('archive/constructors.csv')
    driver_standings = pd.read_csv('archive/driver_standings.csv')
    constructor_standings = pd.read_csv('archive/constructor_standings.csv')
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please make sure the 'archive' directory with the CSV files is in the same directory as the script.")
    exit()

# --- Feature Engineering ---

# Merge basic info
df = pd.merge(results, races[['raceId', 'year', 'circuitId', 'date']], on='raceId', how='left')
df = pd.merge(df, drivers[['driverId', 'nationality', 'dob']], on='driverId', how='left')
df = pd.merge(df, constructors[['constructorId', 'nationality']], on='constructorId', how='left', suffixes=('_driver', '_constructor'))

# Calculate driver age
df['date'] = pd.to_datetime(df['date'])
df['dob'] = pd.to_datetime(df['dob'])
df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25

# Get championship standings before each race
driver_standings_pre_race = driver_standings.rename(columns={'points': 'driver_points', 'position': 'driver_standings_pos'})
constructor_standings_pre_race = constructor_standings.rename(columns={'points': 'constructor_points', 'position': 'constructor_standings_pos'})

df = pd.merge(df, driver_standings_pre_race[['raceId', 'driverId', 'driver_points', 'driver_standings_pos']], on=['raceId', 'driverId'], how='left')
df = pd.merge(df, constructor_standings_pre_race[['raceId', 'constructorId', 'constructor_points', 'constructor_standings_pos']], on=['raceId', 'constructorId'], how='left')

# Fill NaNs for standings (e.g., first race of a season)
df.fillna(0, inplace=True)

# --- Preprocessing ---

# Target variable
df['is_winner'] = (df['positionOrder'] == 1).astype(int)

# Label encode IDs for embedding layers
categorical_features = ['driverId', 'constructorId', 'circuitId']
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Define feature columns
numeric_features = ['grid', 'driver_age', 'driver_points', 'driver_standings_pos', 'constructor_points', 'constructor_standings_pos']

# Group data into sequences per race
races_grouped = df.groupby('raceId')

X_cat = []
X_num = []
y = []

for name, group in races_grouped:
    group = group.sort_values('grid')
    X_cat.append(group[categorical_features].values)
    X_num.append(group[numeric_features].values)
    y.append(group['is_winner'].values)

# Pad sequences
max_drivers = max(len(seq) for seq in X_cat)
X_cat_padded = pad_sequences(X_cat, maxlen=max_drivers, padding='post')
X_num_padded = pad_sequences(X_num, maxlen=max_drivers, padding='post', dtype='float32')
y_padded = pad_sequences(y, maxlen=max_drivers, padding='post', value=0).reshape(-1, max_drivers, 1)

# Create sample weights to ignore padding
sample_weights = (X_cat_padded[:, :, 0] != 0).astype(float)

# Split data
(X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test, sw_train, sw_test) = train_test_split(
    X_cat_padded, X_num_padded, y_padded, sample_weights, test_size=0.2, random_state=42
)

# --- Transformer Model ---

# Embedding layers
input_driver = Input(shape=(max_drivers,), name='input_driverId')
input_constructor = Input(shape=(max_drivers,), name='input_constructorId')
input_circuit = Input(shape=(max_drivers,), name='input_circuitId')
input_numeric = Input(shape=(max_drivers, len(numeric_features)), name='input_numeric')

emb_driver = Embedding(input_dim=df['driverId'].max() + 1, output_dim=10)(input_driver)
emb_constructor = Embedding(input_dim=df['constructorId'].max() + 1, output_dim=10)(input_constructor)
emb_circuit = Embedding(input_dim=df['circuitId'].max() + 1, output_dim=10)(input_circuit)

# Concatenate features
concatenated_features = Concatenate()([emb_driver, emb_constructor, emb_circuit, input_numeric])

# Transformer Blocks
num_heads = 4
d_model = 64

def transformer_encoder(inputs, num_heads, d_model):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn_output = Dense(d_model * 2, activation='relu')(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2

x = transformer_encoder(concatenated_features, num_heads, d_model)
x = transformer_encoder(x, num_heads, d_model) # Second transformer block

# Output layer
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[input_driver, input_constructor, input_circuit, input_numeric], outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('f1_winner_model_v2.keras', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    [X_cat_train[:,:,0], X_cat_train[:,:,1], X_cat_train[:,:,2], X_num_train],
    y_train,
    sample_weight=sw_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate
loss, accuracy = model.evaluate(
    [X_cat_test[:,:,0], X_cat_test[:,:,1], X_cat_test[:,:,2], X_num_test],
    y_test,
    sample_weight=sw_test
)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

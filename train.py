

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
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please make sure the 'archive' directory with the CSV files is in the same directory as the script.")
    exit()

# Merge the datasets
df = pd.merge(results, races[['raceId', 'year', 'circuitId']], on='raceId', how='left')
df = pd.merge(df, drivers[['driverId', 'nationality']], on='driverId', how='left')
df = pd.merge(df, constructors[['constructorId', 'nationality']], on='constructorId', how='left', suffixes=('_driver', '_constructor'))

# Feature selection
features = ['raceId', 'driverId', 'constructorId', 'grid', 'year', 'circuitId', 'nationality_driver', 'nationality_constructor', 'positionOrder']
df = df[features]

# Create the target variable
df['is_winner'] = (df['positionOrder'] == 1).astype(int)

# Label encode categorical features
categorical_features = ['driverId', 'constructorId', 'circuitId', 'nationality_driver', 'nationality_constructor']
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Group data into sequences per race
races_grouped = df.groupby('raceId')

X = []
y = []

for name, group in races_grouped:
    # Sort by grid position to have a consistent order
    group = group.sort_values('grid')
    
    # Drop the raceId and positionOrder columns
    X.append(group.drop(['raceId', 'positionOrder', 'is_winner'], axis=1).values)
    y.append(group['is_winner'].values)

# Pad sequences to the same length
# Find the maximum number of drivers in a race
max_drivers = max(len(seq) for seq in X)

X_padded = pad_sequences(X, maxlen=max_drivers, padding='post', dtype='float32')
y_padded = pad_sequences(y, maxlen=max_drivers, padding='post', value=-1, dtype='float32') # Use -1 for padding in labels

# Remove the padding from the labels for the loss function
y_padded = y_padded.reshape((*y_padded.shape, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)

# --- Transformer Model ---
num_drivers = X_padded.shape[1]
num_features = X_padded.shape[2]
num_heads = 4
d_model = 128
dff = 512
dropout_rate = 0.1

# Input layer
inputs = Input(shape=(num_drivers, num_features))

# Self-attention block
attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(query=inputs, value=inputs, key=inputs)
attn_output = Dropout(dropout_rate)(attn_output)
out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

# Feed-forward block
ffn_output = Dense(dff, activation='relu')(out1)
ffn_output = Dense(num_features)(ffn_output)
ffn_output = Dropout(dropout_rate)(ffn_output)
out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# Output layer
# We want a probability for each driver
outputs = Dense(1, activation='sigmoid')(out2)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('f1_winner_model.keras', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# --- Example Prediction ---
# Take a single race from the test set
sample_race_X = X_test[0:1]
sample_race_y = y_test[0:1]

# Predict the winner probabilities
predictions = model.predict(sample_race_X)

# Get the predicted winner
predicted_winner_index = np.argmax(predictions[0])
actual_winner_index = np.argmax(sample_race_y[0])

print("\n--- Example Prediction ---")
print(f"Race: First race in the test set")
print(f"Predicted winner driver index: {predicted_winner_index}")
print(f"Actual winner driver index: {actual_winner_index}")

# To get the original driverId, you would need to map the index back
# This requires saving the label encoders, which is a good practice for production.


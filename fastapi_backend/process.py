import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Flatten, RepeatVector, Permute, Multiply, Lambda, Activation, Reshape, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Using CPU.")

class SalesForecaster:
    def __init__(self, time_steps=12):
        self.time_steps = time_steps
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
    
    def preprocess_data(self, df):
        # Data cleaning and feature engineering
        df = df.copy()
        df = df.dropna()
        
        # Cap outliers in 'units_sold'
        q99 = df['units_sold'].quantile(0.99)
        df.loc[df['units_sold'] > q99, 'units_sold'] = q99
        
        # Add price difference feature
        df['price_diff'] = df['total_price'] - df['base_price']
        
        # Convert categorical features to one-hot encoding
        df['is_featured_sku'] = df['is_featured_sku'].astype(int)
        df['is_display_sku'] = df['is_display_sku'].astype(int)
        
        return df
    
    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.time_steps):
            X_seq.append(X[i:i + self.time_steps])
            y_seq.append(y[i + self.time_steps])
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        
        # Bidirectional LSTM layers
        lstm_1 = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True))(inputs)
        lstm_1 = Dropout(0.2)(lstm_1)
        
        # Self-attention mechanism
        attention = Dense(1, activation='tanh')(lstm_1)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(128)(attention)  # 128 = 64*2 (bidirectional)
        attention = Permute([2, 1])(attention)
        
        # Apply attention
        sent_representation = Multiply()([lstm_1, attention])
        sent_representation = tf.keras.layers.GlobalAveragePooling1D()(sent_representation)
        
        # Dense layers
        x = Dense(64, activation='relu')(sent_representation)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def fit(self, df, features, target='units_sold', epochs=50, batch_size=32, validation_split=0.2):
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Prepare features and target
        X = df[features]
        y = df[target]
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        # Build and train model
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        self.model = self.build_model(input_shape)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join('models', 'model.keras'),  # Changed path
            monitor='val_loss',
            save_best_only=True
        )
        
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_seq, y_seq):
        y_pred_scaled = self.model.predict(X_seq)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_actual = self.scaler_y.inverse_transform(y_seq)
        
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        return {
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'actuals': y_actual
        }
    
    def predict_next_week(self, new_week_data, X_scaled):
        """Predict sales for the next week"""
        X_last_11 = X_scaled[-11:]
        new_week_data_scaled = self.scaler_X.transform(new_week_data)
        X_new_sequence = np.concatenate([X_last_11, new_week_data_scaled], axis=0)
        X_new_sequence_reshaped = X_new_sequence.reshape((1, 12, X_new_sequence.shape[1]))
        y_pred_scaled_new = self.model.predict(X_new_sequence_reshaped)
        y_pred_new = self.scaler_y.inverse_transform(y_pred_scaled_new)
        return int(round(y_pred_new[0][0]))
    
    def save(self, path):
        """Save model and scalers"""
        os.makedirs(path, exist_ok=True)
        
        # Save model with .keras extension
        model_path = os.path.join(path, 'model.keras')
        self.model.save(model_path)
        
        # Save scalers
        with open(os.path.join(path, 'scaler_X.pkl'), 'wb') as f:
            pickle.dump(self.scaler_X, f)
        
        with open(os.path.join(path, 'scaler_y.pkl'), 'wb') as f:
            pickle.dump(self.scaler_y, f)
    
    @classmethod
    def load(cls, path):
        """Load saved model and scalers"""
        forecaster = cls()
        
        # Load model
        model_path = os.path.join(path, 'model.keras')
        forecaster.model = tf.keras.models.load_model(model_path)
        
        # Load scalers
        with open(os.path.join(path, 'scaler_X.pkl'), 'rb') as f:
            forecaster.scaler_X = pickle.load(f)
        
        with open(os.path.join(path, 'scaler_y.pkl'), 'rb') as f:
            forecaster.scaler_y = pickle.load(f)
        
        return forecaster

# Usage example
if __name__ == "__main__":
    # Load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'train_0irEZ2H.csv')
    df = pd.read_csv(data_path, index_col='record_ID')
    
    # Initialize and train model
    features = ['base_price', 'total_price', 'is_featured_sku', "is_display_sku", "sku_id"]
    forecaster = SalesForecaster()
    history = forecaster.fit(df, features, epochs=5)  # Changed epochs to 10
    
    # Save model
    model_dir = os.path.join(current_dir, 'models')
    forecaster.save(model_dir)
    
    # Make a prediction
    new_week_data = np.array([[100, 120, 1, 1, 9632]])
    X = df[features]
    X_scaled = forecaster.scaler_X.transform(X)
    prediction = forecaster.predict_next_week(new_week_data, X_scaled)
    print(f"Prediction for next week (units sold): {prediction}")
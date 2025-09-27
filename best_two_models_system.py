# Production-Ready Top 2 Inventory Forecasting Models
# =====================================================
# Optimized GRU and Bidirectional LSTM with full deployment

import pandas as pd
import numpy as np
import pickle
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l1_l2
    print(f"✓ TensorFlow version: {tf.__version__}")
    tf.random.set_seed(42)
except Exception as e:
    print(f"✗ TensorFlow not available: {e}")
    exit()

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

class ProductionInventoryForecaster:
    """Production-ready inventory forecasting with top 2 models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        self.feature_names = []
        self.sequence_length = 10
        
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive performance metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        nonzero_mask = y_true != 0
        if np.any(nonzero_mask):
            mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        else:
            mape = (mae / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else 100
        
        return {'r2': float(r2), 'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)}
    
    def load_and_prepare_data(self, file_path):
        """Load and prepare data for training"""
        print("Loading inventory data...")
        
        df = pd.read_csv(file_path)
        print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Ensure required columns exist
        required_cols = ['Name', 'Physical Storage Quantity', 'Weekend', 'Inventory_Cycle']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort and clean data
        df = df.sort_values(['Name', 'Inventory_Cycle']).reset_index(drop=True)
        df['Physical Storage Quantity'] = pd.to_numeric(df['Physical Storage Quantity'], errors='coerce')
        df = df.dropna(subset=['Physical Storage Quantity'])
        
        return df
    
    def create_features(self, df):
        """Create optimized feature set for top models"""
        print("Creating enhanced features...")
        
        df = df.copy()
        target_col = 'Physical Storage Quantity'
        
        # Core lag features - most important for time series
        for lag in [1, 2, 3, 7, 14]:
            df[f'lag_{lag}'] = df.groupby('Name')[target_col].shift(lag)
        
        # Rolling statistics - key patterns
        for window in [3, 7, 14]:
            df[f'ma_{window}'] = df.groupby('Name')[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'std_{window}'] = df.groupby('Name')[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
            ).fillna(0)
        
        # Trend indicators
        df['trend_1_3'] = df.groupby('Name')[target_col].shift(1) - df.groupby('Name')[target_col].shift(3)
        df['trend_7_14'] = df.groupby('Name')[target_col].shift(7) - df.groupby('Name')[target_col].shift(14)
        
        # Weekend and temporal features
        df['weekend'] = df['Weekend']
        df['cycle_sin'] = np.sin(2 * np.pi * df['Inventory_Cycle'] / 52)
        df['cycle_cos'] = np.cos(2 * np.pi * df['Inventory_Cycle'] / 52)
        
        # Volatility measure
        df['volatility'] = df.groupby('Name')[target_col].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
        ).fillna(0)
        
        print(f"✓ Features created. Shape: {df.shape}")
        return df
    
    def prepare_sequences(self, df, target_col='Physical Storage Quantity'):
        """Prepare sequences for LSTM/GRU models"""
        print(f"Preparing sequences (length={self.sequence_length})...")
        
        # Select key features for sequence models
        feature_columns = [
            'lag_1', 'lag_2', 'lag_3', 'ma_3', 'ma_7', 'std_3',
            'weekend', 'cycle_sin', 'cycle_cos', 'trend_1_3', 'volatility'
        ]
        
        # Filter to existing features
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_names = available_features
        
        sequences = []
        targets = []
        metadata = []
        
        for product, product_data in df.groupby('Name'):
            product_data = product_data.sort_values('Inventory_Cycle')
            
            if len(product_data) < self.sequence_length + 5:
                continue
            
            # Get features and target
            features = product_data[available_features].values
            target_values = product_data[target_col].values
            
            # Clean data
            features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
            target_values = np.nan_to_num(target_values, nan=0, posinf=0, neginf=0)
            
            # Create sequences
            for i in range(len(features) - self.sequence_length):
                seq = features[i:(i + self.sequence_length)]
                target = target_values[i + self.sequence_length]
                
                sequences.append(seq)
                targets.append(target)
                metadata.append({
                    'product': product,
                    'cycle': product_data.iloc[i + self.sequence_length]['Inventory_Cycle']
                })
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"✓ Created {len(sequences)} sequences, shape: {sequences.shape}")
        return sequences, targets, metadata
    
    def create_gru_model(self, input_shape):
        """Create optimized GRU model - best overall performer"""
        model = Sequential([
            Input(shape=input_shape),
            GRU(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            GRU(16, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            Dense(16, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01)),
            Dropout(0.3),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_bidirectional_lstm_model(self, input_shape):
        """Create Bidirectional LSTM model - best MAE performer"""
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(24, return_sequences=True, dropout=0.2)),
            BatchNormalization(),
            Bidirectional(LSTM(12, dropout=0.2)),
            BatchNormalization(),
            Dense(16, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01)),
            Dropout(0.3),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_models(self, df):
        """Train both top models"""
        print("\nTraining Top 2 Models...")
        print("=" * 50)
        
        # Prepare data
        X, y, metadata = self.prepare_sequences(df)
        
        # Setup scalers
        self.scalers['features'] = StandardScaler()
        self.scalers['targets'] = StandardScaler()
        
        # Scale features
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scalers['features'].fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)
        
        # Scale targets
        y_scaled = self.scalers['targets'].fit_transform(y.reshape(-1, 1)).flatten()
        
        # Temporal split (80% train, 20% test)
        metadata_df = pd.DataFrame(metadata)
        split_cycle = metadata_df['cycle'].quantile(0.8)
        
        train_mask = metadata_df['cycle'] <= split_cycle
        test_mask = metadata_df['cycle'] > split_cycle
        
        X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
        y_train, y_test = y_scaled[train_mask], y_scaled[test_mask]
        
        print(f"Training set: {len(X_train)} sequences")
        print(f"Test set: {len(X_test)} sequences")
        
        # Model configurations
        models_to_train = {
            'gru': {
                'model_func': self.create_gru_model,
                'epochs': 45,
                'batch_size': 32,
                'description': 'GRU - Best Overall R²'
            },
            'bidirectional_lstm': {
                'model_func': self.create_bidirectional_lstm_model,
                'epochs': 40,
                'batch_size': 32,
                'description': 'Bidirectional LSTM - Best MAE'
            }
        }
        
        results = {}
        
        for name, config in models_to_train.items():
            print(f"\n--- Training {config['description']} ---")
            
            try:
                # Create model
                model = config['model_func'](X_train.shape[1:])
                
                # Callbacks
                callbacks = [
                    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
                    ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6, monitor='val_loss')
                ]
                
                # Train
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Predictions
                y_train_pred_scaled = model.predict(X_train, verbose=0)
                y_test_pred_scaled = model.predict(X_test, verbose=0)
                
                # Inverse transform
                y_train_pred = self.scalers['targets'].inverse_transform(y_train_pred_scaled).flatten()
                y_test_pred = self.scalers['targets'].inverse_transform(y_test_pred_scaled).flatten()
                
                y_train_orig = self.scalers['targets'].inverse_transform(y_train.reshape(-1, 1)).flatten()
                y_test_orig = self.scalers['targets'].inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                train_metrics = self.calculate_metrics(y_train_orig, y_train_pred)
                test_metrics = self.calculate_metrics(y_test_orig, y_test_pred)
                
                # Store results
                self.models[name] = model
                self.model_metadata[name] = {
                    'description': config['description'],
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'epochs_trained': len(history.history['loss']),
                    'best_epoch': len(history.history['loss']) - callbacks[0].stopped_epoch if callbacks[0].stopped_epoch > 0 else len(history.history['loss'])
                }
                
                results[name] = {
                    'model': model,
                    'history': history,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'y_test_actual': y_test_orig,
                    'y_test_predicted': y_test_pred
                }
                
                print(f"✓ {config['description']}")
                print(f"  Train: R²={train_metrics['r2']:.3f}, MAE={train_metrics['mae']:.2f}")
                print(f"  Test:  R²={test_metrics['r2']:.3f}, MAE={test_metrics['mae']:.2f}")
                
            except Exception as e:
                print(f"✗ Failed to train {name}: {e}")
                continue
        
        return results
    
    def predict(self, new_data, model_name='gru'):
        """Make predictions with specified model"""
        if model_name not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Model '{model_name}' not available. Available: {available}")
        
        print(f"Making predictions with {model_name}...")
        
        # Prepare data
        df_features = self.create_features(new_data)
        X, metadata, _ = self.prepare_sequences(df_features)
        
        if len(X) == 0:
            return pd.DataFrame({"error": "No valid sequences for prediction"})
        
        # Scale features
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scalers['features'].transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)
        
        # Predict
        predictions_scaled = self.models[model_name].predict(X_scaled, verbose=0).flatten()
        predictions = self.scalers['targets'].inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        # Create results
        results = pd.DataFrame({
            'product': [m['product'] for m in metadata],
            'inventory_cycle': [m['cycle'] for m in metadata],
            'predicted_inventory': np.maximum(0, predictions),  # Ensure non-negative
            'model_used': model_name
        })
        
        return results
    
    def save_for_deployment(self, save_directory='top_models_deployment'):
        """Save complete system for production deployment"""
        print(f"\nSaving Top 2 Models for Deployment...")
        print("=" * 50)
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save individual model files
        model_files = {}
        for name, model in self.models.items():
            model_path = os.path.join(save_directory, f'{name}_model.h5')
            model.save(model_path)
            model_files[name] = model_path
            print(f"✓ Saved {name} model")
        
        # Save scalers
        scaler_path = os.path.join(save_directory, 'scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"✓ Saved scalers")
        
        # Save deployment config
        config = {
            'model_files': model_files,
            'scaler_path': scaler_path,
            'model_metadata': self.model_metadata,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'saved_date': datetime.now().isoformat(),
            'system_version': 'top_2_models_v1'
        }
        
        config_path = os.path.join(save_directory, 'deployment_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved configuration")
        
        # Save deployment system
        deployment_system = ProductionDeployment()
        deployment_system.load_from_config(config)
        
        system_path = os.path.join(save_directory, 'deployment_system.pkl')
        with open(system_path, 'wb') as f:
            pickle.dump(deployment_system, f)
        print(f"✓ Saved deployment system")
        
        # Create summary
        self._create_summary(save_directory)
        
        print(f"\n✓ Deployment package ready: {os.path.abspath(save_directory)}")
        return save_directory
    
    def _create_summary(self, save_directory):
        """Create deployment summary"""
        summary_path = os.path.join(save_directory, 'MODEL_SUMMARY.txt')
        
        with open(summary_path, 'w') as f:
            f.write("TOP 2 INVENTORY FORECASTING MODELS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models: {len(self.models)} trained\n\n")
            
            for name, metadata in self.model_metadata.items():
                f.write(f"{name.upper()} - {metadata['description']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Test R²: {metadata['test_metrics']['r2']:.3f}\n")
                f.write(f"Test MAE: {metadata['test_metrics']['mae']:.2f}\n")
                f.write(f"Test MAPE: {metadata['test_metrics']['mape']:.2f}%\n\n")
            
            f.write("USAGE:\n")
            f.write("from deployment_system import load_production_system\n")
            f.write("system = load_production_system('top_models_deployment')\n")
            f.write("predictions = system.predict(data, model='gru')\n")

class ProductionDeployment:
    """Lightweight deployment system for production use"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        self.feature_names = []
        self.sequence_length = 10
    
    def load_from_config(self, config):
        """Load configuration"""
        self.model_metadata = config['model_metadata']
        self.feature_names = config['feature_names']
        self.sequence_length = config['sequence_length']
    
    @classmethod
    def load_system(cls, save_directory):
        """Load complete deployment system"""
        system_path = os.path.join(save_directory, 'deployment_system.pkl')
        with open(system_path, 'rb') as f:
            system = pickle.load(f)
        
        # Load config
        config_path = os.path.join(save_directory, 'deployment_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load models
        for name, model_path in config['model_files'].items():
            system.models[name] = load_model(model_path)
        
        # Load scalers
        with open(config['scaler_path'], 'rb') as f:
            system.scalers = pickle.load(f)
        
        print(f"✓ Production system loaded with {len(system.models)} models")
        return system
    
    def create_features(self, df):
        """Create features for prediction (same as training)"""
        df = df.copy()
        target_col = 'Physical Storage Quantity'
        
        # Core features (same as training)
        for lag in [1, 2, 3, 7, 14]:
            df[f'lag_{lag}'] = df.groupby('Name')[target_col].shift(lag)
        
        for window in [3, 7, 14]:
            df[f'ma_{window}'] = df.groupby('Name')[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'std_{window}'] = df.groupby('Name')[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
            ).fillna(0)
        
        df['trend_1_3'] = df.groupby('Name')[target_col].shift(1) - df.groupby('Name')[target_col].shift(3)
        df['trend_7_14'] = df.groupby('Name')[target_col].shift(7) - df.groupby('Name')[target_col].shift(14)
        df['weekend'] = df['Weekend']
        df['cycle_sin'] = np.sin(2 * np.pi * df['Inventory_Cycle'] / 52)
        df['cycle_cos'] = np.cos(2 * np.pi * df['Inventory_Cycle'] / 52)
        df['volatility'] = df.groupby('Name')[target_col].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
        ).fillna(0)
        
        return df
    
    def prepare_sequences(self, df, target_col='Physical Storage Quantity'):
        """Prepare sequences for prediction"""
        sequences = []
        metadata = []
        
        for product, product_data in df.groupby('Name'):
            product_data = product_data.sort_values('Inventory_Cycle')
            
            if len(product_data) < self.sequence_length:
                continue
            
            features = product_data[self.feature_names].values
            features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
            
            for i in range(len(features) - self.sequence_length + 1):
                seq = features[i:(i + self.sequence_length)]
                sequences.append(seq)
                metadata.append({
                    'product': product,
                    'cycle': product_data.iloc[i + self.sequence_length - 1]['Inventory_Cycle']
                })
        
        return np.array(sequences), metadata
    
    def predict(self, new_data, model='gru'):
        """Make predictions"""
        if model not in self.models:
            raise ValueError(f"Model '{model}' not available. Available: {list(self.models.keys())}")
        
        # Prepare data
        df_features = self.create_features(new_data)
        X, metadata = self.prepare_sequences(df_features)
        
        if len(X) == 0:
            return pd.DataFrame({"error": "No sequences created"})
        
        # Scale and predict
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scalers['features'].transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)
        
        predictions_scaled = self.models[model].predict(X_scaled, verbose=0).flatten()
        predictions = self.scalers['targets'].inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        return pd.DataFrame({
            'product': [m['product'] for m in metadata],
            'inventory_cycle': [m['cycle'] for m in metadata],
            'predicted_inventory': np.maximum(0, predictions),
            'model_used': model
        })
    
    def get_model_info(self):
        """Get model information"""
        return {
            'available_models': list(self.models.keys()),
            'model_performance': self.model_metadata,
            'feature_count': len(self.feature_names),
            'sequence_length': self.sequence_length
        }

def load_production_system(save_directory):
    """Convenience function to load production system"""
    return ProductionDeployment.load_system(save_directory)

def main():
    """Run complete training and deployment pipeline"""
    print("PRODUCTION-READY TOP 2 INVENTORY FORECASTING MODELS")
    print("=" * 60)
    
    # Initialize system
    forecaster = ProductionInventoryForecaster()
    
    # Load and train
    df = forecaster.load_and_prepare_data('products_with_weekend.csv')
    df = forecaster.create_features(df)
    results = forecaster.train_models(df)
    
    # Save for deployment
    save_directory = forecaster.save_for_deployment()
    
    # Show results
    print(f"\n" + "=" * 60)
    print("TRAINING COMPLETE - TOP 2 MODELS READY")
    print("=" * 60)
    
    for name, metadata in forecaster.model_metadata.items():
        print(f"\n{metadata['description']}")
        print(f"  Test R²: {metadata['test_metrics']['r2']:.3f}")
        print(f"  Test MAE: {metadata['test_metrics']['mae']:.2f}")
        print(f"  Test MAPE: {metadata['test_metrics']['mape']:.2f}%")
    
    print(f"\n✓ Deployment package ready: {save_directory}")
    print(f"✓ To use: system = load_production_system('{save_directory}')")

if __name__ == "__main__":
    main()
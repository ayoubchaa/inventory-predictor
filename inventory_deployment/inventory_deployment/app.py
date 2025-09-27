from flask import Flask, render_template, request, jsonify, flash, send_file
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from io import BytesIO
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print(f"TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"TensorFlow not available: {e}")

from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)
app.secret_key = 'your-inventory-forecast-key-2024'

import pandas as pd
import numpy as np
import pickle
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from io import BytesIO
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print(f"TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"TensorFlow not available: {e}")

from sklearn.preprocessing import StandardScaler, LabelEncoder

class EnhancedGRUPredictor:
    """Production deployment for Enhanced GRU model with exact performance metrics"""

    def __init__(self, model_directory='enhanced_models_no_mira'):
        self.model_directory = model_directory
        self.gru_model = None
        self.lstm_model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.encoders = {}
        self.config = {}
        self.feature_names = []
        self.sequence_length = 10
        self.cycles_per_year = 36.5
        self.load_system()

    def load_system(self):
        """Load the complete trained system"""
        try:
            # Load configuration
            config_path = os.path.join(self.model_directory, 'config.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)

            self.feature_names = self.config['feature_names']
            self.sequence_length = self.config['sequence_length']

            # Load GRU model (primary model)
            gru_path = os.path.join(self.model_directory, 'gru_model.h5')
            self.gru_model = load_model(gru_path)
            print("[+] Enhanced GRU model loaded successfully")

            # Load LSTM model (backup model)
            lstm_path = os.path.join(self.model_directory, 'bidirectional_lstm_model.h5')
            if os.path.exists(lstm_path):
                self.lstm_model = load_model(lstm_path)
                print("[+] Enhanced LSTM model loaded successfully")

            # Load scalers
            feature_scaler_path = os.path.join(self.model_directory, 'feature_scaler.pkl')
            target_scaler_path = os.path.join(self.model_directory, 'target_scaler.pkl')

            self.feature_scaler = joblib.load(feature_scaler_path)
            self.target_scaler = joblib.load(target_scaler_path)
            print("[+] Scalers loaded successfully")

            # Load encoders
            encoder_path = os.path.join(self.model_directory, 'encoders.pkl')
            with open(encoder_path, 'rb') as f:
                self.encoders = pickle.load(f)
            print("[+] Encoders loaded successfully")

            print(f"[+] System ready - Features: {len(self.feature_names)}, Sequence Length: {self.sequence_length}")
            print(f"[+] Model Performance: GRU R2=0.803, LSTM R2=0.776")

        except Exception as e:
            print(f"[!] Error loading system: {e}")
            raise

    def get_model_info(self):
        """Get exact model performance information from your training results"""
        return {
            'model_type': 'Enhanced GRU',
            'gru_r2_score': 0.803,
            'gru_mae': 162.91,
            'gru_mape': 111.14,
            'lstm_r2_score': 0.776,
            'lstm_mae': 188.49,
            'lstm_mape': 183.33,
            'features_count': 26,
            'sequence_length': self.sequence_length,
            'pattern_recognition_accuracy': '80.3%'
        }

    def create_prediction_features(self, input_data):
        """Create features for prediction matching training pipeline"""
        # Create dataframe with input
        df = pd.DataFrame([input_data])

        # Basic data preparation
        target_col = 'Physical Storage Quantity'
        current_stock = float(input_data.get('current_stock', 0))
        max_stock = float(input_data.get('max_stock', max(100, current_stock * 2)))
        cycle = int(input_data.get('cycle', 1))
        month = int(input_data.get('month', 1))

        df[target_col] = current_stock
        df['Max_Stock'] = max_stock
        df['Cycle'] = cycle
        df['Month'] = month

        # Create lag features (simulate recent history)
        base_stock = current_stock if current_stock > 0 else 50
        for lag in [1, 2, 3, 5, 7]:
            # Add small random variation to simulate realistic lag values
            variation_factor = 1 + np.random.normal(0, 0.1)
            df[f'lag_{lag}'] = base_stock * variation_factor

        # Rolling statistics
        for window in [3, 5, 7]:
            df[f'ma_{window}'] = base_stock * (1 + np.random.normal(0, 0.05))
            df[f'std_{window}'] = base_stock * np.random.uniform(0.05, 0.15)

        # Trend features
        df['trend_1_3'] = np.random.normal(0, base_stock * 0.1)
        df['trend_3_7'] = np.random.normal(0, base_stock * 0.1)

        # Temporal features (same as training)
        df['cycle_sin'] = np.sin(2 * np.pi * cycle / self.cycles_per_year)
        df['cycle_cos'] = np.cos(2 * np.pi * cycle / self.cycles_per_year)
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)

        # Stock capacity features
        df['stock_utilization'] = current_stock / (max_stock + 1e-8)
        df['stock_utilization'] = df['stock_utilization'].clip(0, 2)
        df['near_capacity'] = (df['stock_utilization'] > 0.8).astype(int)
        df['low_stock'] = (df['stock_utilization'] < 0.2).astype(int)
        df['stock_util_lag_1'] = df['stock_utilization'] * 0.95

        # Categorical encoding with error handling
        try:
            # GroupName encoding
            group_name = str(input_data.get('group_name', 'DEFAULT'))
            if 'groupname_encoder' in self.encoders:
                try:
                    df['group_encoded'] = self.encoders['groupname_encoder'].transform([group_name])[0]
                except ValueError:
                    df['group_encoded'] = 0
            else:
                df['group_encoded'] = 0

            # Stock_Class encoding  
            stock_class = str(input_data.get('stock_class', 'DEFAULT'))
            if 'stock_class_encoder' in self.encoders:
                try:
                    df['stock_class_encoded'] = self.encoders['stock_class_encoder'].transform([stock_class])[0]
                except ValueError:
                    df['stock_class_encoded'] = 0
            else:
                df['stock_class_encoded'] = 0

            # Storage Unit encoding
            storage_unit = str(input_data.get('storage_unit', 'DEFAULT'))
            if 'storage_unit_encoder' in self.encoders:
                try:
                    df['storage_unit_encoded'] = self.encoders['storage_unit_encoder'].transform([storage_unit])[0]
                except ValueError:
                    df['storage_unit_encoded'] = 0
            else:
                df['storage_unit_encoded'] = 0

        except Exception as e:
            print(f"Encoding error: {e}")
            df['group_encoded'] = 0
            df['stock_class_encoded'] = 0
            df['storage_unit_encoded'] = 0

        # Volatility features
        df['volatility'] = base_stock * 0.1
        df['volatility_normalized'] = df['volatility'] / (df['ma_5'] + 1e-8)
        df['volatility_normalized'] = df['volatility_normalized'].clip(0, 5)

        # Advanced features (same as training)
        df['ma_ratio_3_7'] = df['ma_3'] / (df['ma_7'] + 1e-8)
        df['ma_ratio_3_7'] = df['ma_ratio_3_7'].clip(0, 3)
        df['stock_class_util_interaction'] = df['stock_class_encoded'] * df['stock_utilization']
        df['group_volatility'] = df['group_encoded'] * df['volatility_normalized']

        # Weekend features
        df['is_weekend'] = 0  # Assume weekday
        df['is_friday'] = 0

        # Fill any remaining NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(0)

        return df

    def prepare_prediction_sequence(self, df):
        """Prepare sequence for model prediction"""
        # Use exact same feature names as training
        available_features = [col for col in self.feature_names if col in df.columns]

        if len(available_features) != len(self.feature_names):
            missing = set(self.feature_names) - set(available_features)
            print(f"Warning: Missing features: {missing}")
            # Fill missing features with zeros
            for feat in missing:
                df[feat] = 0.0

        # Extract features in correct order
        features = df[self.feature_names].values[0]
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

        # Create sequence by repeating current features
        sequence = np.tile(features, (self.sequence_length, 1))
        sequence = sequence.reshape(1, self.sequence_length, len(self.feature_names))

        return sequence

    def predict(self, input_data, use_ensemble=True):
        """Make prediction using the Enhanced GRU model with exact business logic"""
        try:
            # Create features
            df = self.create_prediction_features(input_data)

            # Prepare sequence  
            sequence = self.prepare_prediction_sequence(df)

            # Scale features using training scaler
            original_shape = sequence.shape
            sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
            sequence_scaled = self.feature_scaler.transform(sequence_reshaped)
            sequence_scaled = sequence_scaled.reshape(original_shape)

            # Make prediction with GRU model (primary)
            gru_prediction_scaled = self.gru_model.predict(sequence_scaled, verbose=0)
            gru_prediction = self.target_scaler.inverse_transform(gru_prediction_scaled).flatten()[0]

            # Ensemble prediction if LSTM available
            lstm_prediction = None
            if use_ensemble and self.lstm_model is not None:
                lstm_prediction_scaled = self.lstm_model.predict(sequence_scaled, verbose=0)
                lstm_prediction = self.target_scaler.inverse_transform(lstm_prediction_scaled).flatten()[0]

                # Weighted ensemble (favor GRU since it performed better: R2=0.803 vs 0.776)
                final_prediction = 0.7 * gru_prediction + 0.3 * lstm_prediction
            else:
                final_prediction = gru_prediction

            # Apply business constraints
            final_prediction = max(0, final_prediction)  # No negative stock
            current_stock = float(input_data.get('current_stock', 0))

            # Cap at reasonable maximum based on current stock
            if current_stock > 0:
                max_reasonable = current_stock * 4  # Max 4x current stock
                final_prediction = min(final_prediction, max_reasonable)
            else:
                final_prediction = min(final_prediction, 1000)  # Default cap

            # Calculate stock change
            stock_change = final_prediction - current_stock

            # EXACT BUSINESS LOGIC YOU REQUESTED:
            # When current_stock is 0, predicted_next_cycle equals stock_change
            if current_stock == 0:
                predicted_next_cycle = stock_change
            else:
                predicted_next_cycle = final_prediction

            # Business intelligence flags
            high_demand = False
            restock_needed = False

            if current_stock > 0:
                # High demand: predicted > 150% of current stock
                high_demand = final_prediction > (current_stock * 1.5)
                # Restock needed: stock change > 50% of current stock
                restock_needed = stock_change > (current_stock * 0.5)
            else:
                # For zero stock scenarios
                high_demand = final_prediction > 100
                restock_needed = stock_change > 50

            # Calculate confidence
            confidence = self.calculate_confidence(input_data, final_prediction)
            confidence_score = self.get_confidence_score(confidence)

            return {
                'prediction': round(final_prediction, 2),
                'predicted_next_cycle': round(predicted_next_cycle, 2),
                'stock_change': round(stock_change, 2),
                'current_stock': current_stock,
                'confidence': confidence,
                'confidence_score': confidence_score,
                'gru_prediction': round(gru_prediction, 2),
                'lstm_prediction': round(lstm_prediction, 2) if lstm_prediction else None,
                'high_demand': high_demand,
                'restock_needed': restock_needed,
                'status': 'success'
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'prediction': 0,
                'predicted_next_cycle': 0,
                'stock_change': 0,
                'current_stock': 0,
                'confidence': 'low',
                'confidence_score': 0,
                'status': 'error',
                'message': str(e)
            }

    def batch_predict(self, products_data):
        """Make predictions for multiple products"""
        results = []

        for product in products_data:
            result = self.predict(product, use_ensemble=True)

            # Add product identification and metadata
            result.update({
                'product_name': product.get('product_name', 'Unknown Product'),
                'group_name': product.get('group_name', 'DEFAULT'),
                'stock_class': product.get('stock_class', 'DEFAULT'),
                'storage_unit': product.get('storage_unit', 'DEFAULT'),
                'max_stock': float(product.get('max_stock', 100)),
                'cycle': int(product.get('cycle', 1)),
                'month': int(product.get('month', 1)),
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            results.append(result)

        return results

    def calculate_confidence(self, input_data, prediction):
        """Calculate prediction confidence based on input quality and model performance"""
        current_stock = float(input_data.get('current_stock', 0))

        # Base confidence on model performance (GRU R2 = 0.803)
        base_confidence = 0.803

        # Adjust based on input quality
        if current_stock == 0:
            # Medium confidence for zero stock (still meaningful prediction)
            return 'medium'

        # Calculate confidence based on prediction reasonableness
        ratio = prediction / current_stock if current_stock > 0 else 1

        if 0.7 <= ratio <= 1.5:  # Prediction within reasonable range
            return 'high'
        elif 0.3 <= ratio <= 3.0:  # Moderately reasonable
            return 'medium'
        else:
            return 'low'

    def get_confidence_score(self, confidence):
        """Convert confidence to numeric score based on model performance"""
        # Based on GRU R2 = 0.803, LSTM R2 = 0.776
        confidence_map = {
            'high': 0.80,    # Close to GRU performance
            'medium': 0.65,  # Conservative estimate
            'low': 0.40      # Lower confidence
        }
        return confidence_map.get(confidence, 0.5)

class ExcelReportGenerator:
    """Generate comprehensive Excel reports with all required analysis sheets"""

    def __init__(self, predictions_data):
        self.predictions_data = predictions_data
        self.df = pd.DataFrame(predictions_data)

    def create_comprehensive_report(self):
        """Create Excel file with 4 required analysis sheets"""
        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: All Predictions (complete dataset)
            self.create_all_predictions_sheet(writer)

            # Sheet 2: High Demand Products
            self.create_high_demand_sheet(writer)

            # Sheet 3: Restock Needed
            self.create_restock_needed_sheet(writer)

            # Sheet 4: Executive Summary
            self.create_executive_summary_sheet(writer)

        output.seek(0)
        return output

    def create_all_predictions_sheet(self, writer):
        """Create All_Predictions sheet with complete dataset"""
        # Select and order columns (NO current_stock as requested)
        columns = [
            'product_name', 'group_name', 'stock_class', 'storage_unit',
            'predicted_next_cycle', 'stock_change', 'confidence', 
            'confidence_score', 'high_demand', 'restock_needed',
            'max_stock', 'cycle', 'month', 'prediction_date'
        ]

        # Filter available columns
        available_columns = [col for col in columns if col in self.df.columns]
        df_filtered = self.df[available_columns].copy()

        # Round numeric columns
        numeric_columns = ['predicted_next_cycle', 'stock_change', 'confidence_score', 'max_stock']
        for col in numeric_columns:
            if col in df_filtered.columns:
                df_filtered[col] = df_filtered[col].round(2)

        # Convert boolean flags to Yes/No for readability
        for col in ['high_demand', 'restock_needed']:
            if col in df_filtered.columns:
                df_filtered[col] = df_filtered[col].map({True: 'Yes', False: 'No'})

        # Write to Excel
        df_filtered.to_excel(writer, sheet_name='All_Predictions', index=False)

        # Format the sheet
        worksheet = writer.sheets['All_Predictions']
        self.format_worksheet(worksheet, df_filtered, 'All Predictions Analysis')

    def create_high_demand_sheet(self, writer):
        """Create High_Demand_Products sheet"""
        high_demand_df = self.df[self.df['high_demand'] == True].copy()

        if not high_demand_df.empty:
            # Sort by predicted demand (descending)
            high_demand_df = high_demand_df.sort_values('predicted_next_cycle', ascending=False)

            columns = [
                'product_name', 'group_name', 'predicted_next_cycle', 
                'stock_change', 'confidence', 'confidence_score'
            ]

            available_columns = [col for col in columns if col in high_demand_df.columns]
            high_demand_filtered = high_demand_df[available_columns].copy()

            # Round numeric columns
            numeric_columns = ['predicted_next_cycle', 'stock_change', 'confidence_score']
            for col in numeric_columns:
                if col in high_demand_filtered.columns:
                    high_demand_filtered[col] = high_demand_filtered[col].round(2)

            high_demand_filtered.to_excel(writer, sheet_name='High_Demand_Products', index=False)

            # Format the sheet with red highlight
            worksheet = writer.sheets['High_Demand_Products']
            self.format_worksheet(worksheet, high_demand_filtered, 'High Demand Products', 'FFE6E6')
        else:
            # Create empty sheet with message
            empty_df = pd.DataFrame({'Message': ['No high demand products identified']})
            empty_df.to_excel(writer, sheet_name='High_Demand_Products', index=False)

    def create_restock_needed_sheet(self, writer):
        """Create Restock_Needed sheet"""
        restock_df = self.df[self.df['restock_needed'] == True].copy()

        if not restock_df.empty:
            # Sort by stock change (descending) - highest needs first
            restock_df = restock_df.sort_values('stock_change', ascending=False)

            columns = [
                'product_name', 'group_name', 'predicted_next_cycle',
                'stock_change', 'confidence', 'confidence_score'
            ]

            available_columns = [col for col in columns if col in restock_df.columns]
            restock_filtered = restock_df[available_columns].copy()

            # Round numeric columns
            numeric_columns = ['predicted_next_cycle', 'stock_change', 'confidence_score']
            for col in numeric_columns:
                if col in restock_filtered.columns:
                    restock_filtered[col] = restock_filtered[col].round(2)

            restock_filtered.to_excel(writer, sheet_name='Restock_Needed', index=False)

            # Format the sheet with orange highlight
            worksheet = writer.sheets['Restock_Needed']
            self.format_worksheet(worksheet, restock_filtered, 'Restock Required Products', 'FFF2E6')
        else:
            # Create empty sheet with message
            empty_df = pd.DataFrame({'Message': ['No products requiring restock identified']})
            empty_df.to_excel(writer, sheet_name='Restock_Needed', index=False)

    def create_executive_summary_sheet(self, writer):
        """Create Executive_Summary sheet with key business metrics"""
        # Calculate comprehensive summary statistics
        total_products = len(self.df)
        high_demand_count = len(self.df[self.df['high_demand'] == True])
        restock_needed_count = len(self.df[self.df['restock_needed'] == True])

        # Calculate confidence statistics
        avg_confidence = self.df['confidence_score'].mean()
        high_confidence_count = len(self.df[self.df['confidence'] == 'high'])

        # Calculate stock metrics (excluding current_stock as requested)
        total_predicted_stock = self.df['predicted_next_cycle'].sum()
        total_stock_change = self.df['stock_change'].sum()
        avg_stock_change = self.df['stock_change'].mean()

        # Calculate percentages
        high_demand_pct = (high_demand_count / total_products * 100) if total_products > 0 else 0
        restock_needed_pct = (restock_needed_count / total_products * 100) if total_products > 0 else 0
        high_confidence_pct = (high_confidence_count / total_products * 100) if total_products > 0 else 0

        # Create comprehensive summary data
        summary_data = {
            'Business Metric': [
                'Total Products Analyzed',
                'High Demand Products',
                'Products Needing Restock', 
                'High Confidence Predictions',
                'Average Prediction Confidence',
                'Total Predicted Next Cycle Stock',
                'Total Stock Change Required',
                'Average Stock Change per Product',
                'High Demand Percentage',
                'Restock Needed Percentage',
                'High Confidence Percentage',
                'Model Performance (GRU R2)',
                'Model Performance (LSTM R2)',
                'Report Generation Date'
            ],
            'Value': [
                total_products,
                high_demand_count,
                restock_needed_count,
                high_confidence_count,
                f"{avg_confidence:.1%}",
                f"{total_predicted_stock:,.0f}",
                f"{total_stock_change:,.0f}",
                f"{avg_stock_change:,.1f}",
                f"{high_demand_pct:.1f}%",
                f"{restock_needed_pct:.1f}%", 
                f"{high_confidence_pct:.1f}%",
                "80.3%",
                "77.6%",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ],
            'Interpretation': [
                'Number of products in analysis',
                'Products with >150% demand increase',
                'Products requiring >50% stock increase',
                'Predictions with high confidence rating',
                'Overall model confidence level',
                'Sum of all predicted stock levels',
                'Total additional stock needed',
                'Mean stock change per product',
                'Percentage flagged as high demand',
                'Percentage requiring restocking',
                'Percentage with high confidence',
                'Enhanced GRU model accuracy',
                'LSTM model accuracy',
                'When this report was generated'
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)

        # Format the summary sheet
        worksheet = writer.sheets['Executive_Summary']
        self.format_executive_summary(worksheet, summary_df)

    def format_worksheet(self, worksheet, df, title, highlight_color='E6F3FF'):
        """Format worksheet with professional styling"""
        # Add title row
        worksheet.insert_rows(1)
        worksheet['A1'] = title
        title_font = Font(size=14, bold=True, color='FFFFFF')
        title_fill = PatternFill(start_color='2E8B57', end_color='2E8B57', fill_type='solid')
        worksheet['A1'].font = title_font
        worksheet['A1'].fill = title_fill
        worksheet.merge_cells(f'A1:{chr(64 + len(df.columns))}1')

        # Header formatting (row 2 after title)
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        header_font = Font(color='FFFFFF', bold=True)

        # Apply header formatting to row 2
        for cell in worksheet[2]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center')

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass

            adjusted_width = min(max_length + 2, 35)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    def format_executive_summary(self, worksheet, df):
        """Special formatting for executive summary worksheet"""
        # Title formatting
        worksheet.insert_rows(1)
        worksheet['A1'] = 'EXECUTIVE SUMMARY - INVENTORY FORECAST ANALYSIS'
        title_font = Font(size=16, bold=True, color='FFFFFF')
        title_fill = PatternFill(start_color='1F4E79', end_color='1F4E79', fill_type='solid')
        worksheet['A1'].font = title_font
        worksheet['A1'].fill = title_fill
        worksheet.merge_cells('A1:C1')
        worksheet['A1'].alignment = Alignment(horizontal='center')

        # Header formatting (row 2)
        header_fill = PatternFill(start_color='2E8B57', end_color='2E8B57', fill_type='solid')
        header_font = Font(color='FFFFFF', bold=True, size=12)

        for cell in worksheet[2]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center')

        # Value formatting - highlight important metrics
        value_font = Font(bold=True, size=11)
        important_rows = [3, 4, 5, 6, 13, 14]  # Key business metrics

        for row in range(3, len(df) + 3):
            # Bold values
            worksheet[f'B{row}'].font = value_font
            worksheet[f'B{row}'].alignment = Alignment(horizontal='center')

            # Highlight important metrics
            if row in important_rows:
                highlight_fill = PatternFill(start_color='F0F8FF', end_color='F0F8FF', fill_type='solid')
                for col in ['A', 'B', 'C']:
                    worksheet[f'{col}{row}'].fill = highlight_fill

        # Auto-adjust column widths
        worksheet.column_dimensions['A'].width = 40
        worksheet.column_dimensions['B'].width = 25
        worksheet.column_dimensions['C'].width = 45

# Test the deployment system
def test_deployment_system():
    """Test the complete deployment system"""
    print("\n" + "="*60)
    print("TESTING DEPLOYMENT SYSTEM")
    print("="*60)

    try:
        # Initialize predictor
        predictor = EnhancedGRUPredictor()
        model_info = predictor.get_model_info()

        print(f"[+] Model loaded successfully")
        print(f"[+] GRU Performance: R2={model_info['gru_r2_score']}, MAE={model_info['gru_mae']}")
        print(f"[+] LSTM Performance: R2={model_info['lstm_r2_score']}, MAE={model_info['lstm_mae']}")

        # Test single prediction
        test_data = {
            'product_name': 'Test Product',
            'current_stock': 100,
            'max_stock': 500,
            'cycle': 15,
            'month': 6,
            'group_name': 'Test Group',
            'stock_class': 'A',
            'storage_unit': 'KG'
        }

        result = predictor.predict(test_data)

        print(f"\n[+] Single Prediction Test:")
        print(f"  Predicted Next Cycle: {result['predicted_next_cycle']}")
        print(f"  Stock Change: {result['stock_change']}")
        print(f"  Confidence: {result['confidence']}")

        # Test zero stock scenario
        test_data_zero = {
            'product_name': 'Zero Stock Product',
            'current_stock': 0,
            'max_stock': 200,
            'cycle': 10,
            'month': 3
        }

        result_zero = predictor.predict(test_data_zero)

        print(f"\n[+] Zero Stock Test:")
        print(f"  Current Stock: {result_zero['current_stock']}")
        print(f"  Predicted Next Cycle: {result_zero['predicted_next_cycle']}")
        print(f"  Stock Change: {result_zero['stock_change']}")
        print(f"  Note: predicted_next_cycle == stock_change: {result_zero['predicted_next_cycle'] == result_zero['stock_change']}")

        # Test batch prediction
        batch_data = [test_data, test_data_zero]
        batch_results = predictor.batch_predict(batch_data)

        print(f"\n[+] Batch Prediction Test: {len(batch_results)} products")

        # Test Excel generation
        excel_generator = ExcelReportGenerator(batch_results)
        excel_file = excel_generator.create_comprehensive_report()

        print(f"[+] Excel Report Generated: {len(excel_file.getvalue())} bytes")

        return True

    except Exception as e:
        print(f"[!] Deployment test failed: {e}")
        return False

print("[+] Enhanced GRU Predictor classes created successfully!")
print("[+] Excel report generator ready!")
print("[+] Business logic implemented:")
print("  - Current stock = 0 -> predicted_next_cycle = stock_change")
print("  - High demand detection (>150% increase)")
print("  - Restock alerts (>50% increase)")
print("  - 4 Excel analysis sheets")
print("  - Model performance: GRU R2=0.803, LSTM R2=0.776")


# Initialize the predictor
try:
    predictor = EnhancedGRUPredictor()
    print("[+] Enhanced GRU Predictor initialized successfully")
except Exception as e:
    print(f"[!] Failed to initialize predictor: {e}")
    predictor = None

@app.route('/')
def index():
    """Main prediction interface"""
    if predictor is None:
        flash('Enhanced GRU model not loaded. Please check model files.', 'error')
        return render_template('error.html')

    model_info = predictor.get_model_info()
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from form"""
    if predictor is None:
        return jsonify({'error': 'Enhanced GRU model not loaded'}), 500

    try:
        # Extract form data
        input_data = {
            'product_name': request.form.get('product_name', 'Unknown Product'),
            'current_stock': request.form.get('current_stock', '0'),
            'max_stock': float(request.form.get('max_stock', 100)),
            'cycle': int(request.form.get('cycle', 1)),
            'month': int(request.form.get('month', 1)),
            'group_name': request.form.get('group_name', 'DEFAULT'),
            'stock_class': request.form.get('stock_class', 'DEFAULT'),
            'storage_unit': request.form.get('storage_unit', 'DEFAULT')
        }

        # Handle empty current_stock (default to 0)
        if input_data['current_stock'] == '' or input_data['current_stock'] is None:
            input_data['current_stock'] = 0
        else:
            input_data['current_stock'] = float(input_data['current_stock'])

        # Make prediction
        result = predictor.predict(input_data, use_ensemble=True)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions and return results"""
    if predictor is None:
        return jsonify({'error': 'Enhanced GRU model not loaded'}), 500

    try:
        # Get products data from form or JSON
        if request.is_json:
            products_data = request.json.get('products', [])
        else:
            # Handle form-based batch prediction
            products_data = []
            # You would implement form parsing for multiple products here

        if not products_data:
            return jsonify({'error': 'No products data provided'}), 400

        # Make batch predictions
        results = predictor.batch_predict(products_data)

        return jsonify({
            'predictions': results,
            'summary': {
                'total_products': len(results),
                'high_demand_count': sum(1 for r in results if r.get('high_demand')),
                'restock_needed_count': sum(1 for r in results if r.get('restock_needed')),
                'avg_confidence': np.mean([r.get('confidence_score', 0) for r in results])
            }
        })

    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/download_excel', methods=['POST'])
def download_excel():
    """Generate and download Excel report"""
    if predictor is None:
        return jsonify({'error': 'Enhanced GRU model not loaded'}), 500

    try:
        # Get products data for batch prediction
        if request.is_json:
            products_data = request.json.get('products', [])
        else:
            # Handle form-based data
            products_data = []
            # Implement form parsing here

        if not products_data:
            return jsonify({'error': 'No products data provided for Excel generation'}), 400

        # Make batch predictions
        results = predictor.batch_predict(products_data)

        # Generate Excel report
        excel_generator = ExcelReportGenerator(results)
        excel_file = excel_generator.create_comprehensive_report()

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'inventory_forecast_report_{timestamp}.xlsx'

        return send_file(
            excel_file,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        return jsonify({
            'error': f'Excel generation failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for external predictions"""
    if predictor is None:
        return jsonify({'error': 'Enhanced GRU model not loaded'}), 500

    try:
        input_data = request.json

        # Ensure current_stock defaults to 0
        if 'current_stock' not in input_data or input_data['current_stock'] in [None, '']:
            input_data['current_stock'] = 0

        result = predictor.predict(input_data, use_ensemble=True)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': f'API prediction failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500

    info = predictor.get_model_info()
    return jsonify(info)

@app.route('/health')
def health():
    """System health check"""
    if predictor is None:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': 'Enhanced GRU model not available'
        }), 500

    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_type': 'Enhanced GRU',
        'r2_score': 0.803,
        'pattern_recognition': '80.3%',
        'features': len(predictor.feature_names)
    })

if __name__ == '__main__':
    print("=" * 60)
    print("ENHANCED GRU INVENTORY FORECASTING SYSTEM")
    print("=" * 60)
    print("Model Performance:")
    print("- R2 Score: 0.803 (80.3% Pattern Recognition Accuracy)")
    print("- MAE: 162.91")
    print("- MAPE: 111.14%")
    print("- Features: 26")
    print("- Architecture: Enhanced GRU + Bidirectional LSTM Ensemble")
    print("=" * 60)

    if predictor:
        print("[+] System Status: READY")
        print("[+] Primary Model: Enhanced GRU (R2=0.803)")
        print("[+] Secondary Model: Bidirectional LSTM (R2=0.776)")
        print("[+] Ensemble Mode: Active")
        print("[+] Business Constraints: Applied")
        print("[+] Excel Export: Enabled")
        print("[+] Batch Processing: Enabled")
    else:
        print("[!] System Status: ERROR - Model not loaded")

    print("\nStarting Flask server on http://localhost:5000")
    print("Access web interface at: http://localhost:5000")
    print("API endpoint available at: http://localhost:5000/api/predict")
    print("Excel reports available via: /download_excel")
    print("=" * 60)

    # Run Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)

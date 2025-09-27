# Flask Deployment for Inventory Forecasting System
# =================================================
# Complete web application with Power BI integration

from flask import Flask, request, jsonify, render_template_string, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging
from io import StringIO
import traceback

# Import your model system
from best_two_models_system import load_production_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model system (loaded once)
MODEL_SYSTEM = None

def load_models():
    """Load the trained models on app startup"""
    global MODEL_SYSTEM
    try:
        MODEL_SYSTEM = load_production_system('top_models_deployment')
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

# HTML Templates
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inventory Forecasting System</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; margin-bottom: 30px; border-radius: 5px; }
        .section { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #3498db; }
        .upload-area { border: 2px dashed #bdc3c7; padding: 30px; text-align: center; margin: 20px 0; border-radius: 5px; }
        .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #2980b9; }
        .results { background: #2ecc71; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .error { background: #e74c3c; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .model-info { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
        .metric { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .api-docs { background: #34495e; color: white; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .code { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: monospace; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔮 Inventory Forecasting System</h1>
        <p>Advanced ML-powered inventory predictions with Power BI integration</p>
    </div>

    <div class="section">
        <h2>📊 System Status</h2>
        <div class="model-info">
            <div class="metric">
                <h4>GRU Model (Primary)</h4>
                <p><strong>R² Score:</strong> 0.762</p>
                <p><strong>MAE:</strong> 29.17</p>
                <p><strong>Status:</strong> ✅ Ready</p>
            </div>
            <div class="metric">
                <h4>Bidirectional LSTM</h4>
                <p><strong>R² Score:</strong> 0.708</p>
                <p><strong>MAE:</strong> 32.22</p>
                <p><strong>Status:</strong> ✅ Ready</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📁 Upload Data for Predictions</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <input type="file" id="dataFile" name="file" accept=".csv" style="display: none;">
                <p>Click or drag CSV file here</p>
                <button type="button" class="btn" onclick="document.getElementById('dataFile').click();">
                    Choose File
                </button>
                <p id="fileName" style="margin-top: 10px; font-weight: bold;"></p>
            </div>
            
            <div style="margin: 20px 0;">
                <label for="modelSelect">Choose Model:</label>
                <select id="modelSelect" name="model" style="padding: 8px; margin-left: 10px;">
                    <option value="gru">GRU (Best Overall - R² 0.762)</option>
                    <option value="bidirectional_lstm">Bidirectional LSTM (Best MAE - 32.22)</option>
                </select>
            </div>

            <button type="submit" class="btn">Generate Forecasts</button>
        </form>

        <div id="results"></div>
    </div>

    <div class="api-docs">
        <h2>🔌 Power BI Integration</h2>
        <h3>API Endpoints for Power BI:</h3>
        
        <div class="code">
<strong>1. Get Predictions (POST):</strong>
URL: http://localhost:5000/api/predict
Headers: Content-Type: application/json
Body: {
    "model": "gru",
    "data": [your_data_records]
}

<strong>2. Get Latest Forecasts (GET):</strong>
URL: http://localhost:5000/api/latest-forecasts?model=gru

<strong>3. Model Performance (GET):</strong>
URL: http://localhost:5000/api/model-info
        </div>

        <h3>Power BI Setup Steps:</h3>
        <ol>
            <li>In Power BI Desktop, go to "Get Data" → "Web"</li>
            <li>Enter API URL: <code>http://localhost:5000/api/latest-forecasts?model=gru</code></li>
            <li>Power BI will automatically parse the JSON response</li>
            <li>Create relationships between forecast data and your existing data</li>
            <li>Build dashboards with predicted vs actual inventory</li>
        </ol>
    </div>

    <script>
        document.getElementById('dataFile').addEventListener('change', function(e) {
            document.getElementById('fileName').textContent = e.target.files[0].name;
        });

        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('dataFile');
            const modelSelect = document.getElementById('modelSelect');
            
            if (!fileInput.files[0]) {
                document.getElementById('results').innerHTML = 
                    '<div class="error">Please select a CSV file</div>';
                return;
            }
            
            formData.append('file', fileInput.files[0]);
            formData.append('model', modelSelect.value);
            
            document.getElementById('results').innerHTML = 
                '<div class="results">Processing... Please wait</div>';
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('results').innerHTML = `
                        <div class="results">
                            <h3>✅ Predictions Generated Successfully!</h3>
                            <p><strong>Model Used:</strong> ${data.model_used}</p>
                            <p><strong>Total Forecasts:</strong> ${data.total_predictions}</p>
                            <p><strong>Products Covered:</strong> ${data.unique_products}</p>
                            <p><strong>Download Links:</strong></p>
                            <a href="/download/${data.filename}" class="btn">Download CSV</a>
                            <a href="/download/json/${data.filename.replace('.csv', '.json')}" class="btn">Download JSON (Power BI)</a>
                        </div>
                        <div class="section">
                            <h4>Sample Predictions:</h4>
                            <div style="overflow-x: auto;">
                                ${data.sample_html}
                            </div>
                        </div>
                    `;
                } else {
                    document.getElementById('results').innerHTML = 
                        `<div class="error">Error: ${data.error}</div>`;
                }
            })
            .catch(error => {
                document.getElementById('results').innerHTML = 
                    `<div class="error">Error: ${error.message}</div>`;
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Main dashboard"""
    return render_template_string(MAIN_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Upload file and generate predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        model_name = request.form.get('model', 'gru')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files are supported'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        # Load data and make predictions
        data = pd.read_csv(filepath)
        predictions = MODEL_SYSTEM.predict(data, model=model_name)
        
        # Generate output filename
        output_filename = f"forecasts_{model_name}_{timestamp}.csv"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Save predictions
        predictions.to_csv(output_filepath, index=False)
        
        # Also save as JSON for Power BI
        json_filename = output_filename.replace('.csv', '.json')
        json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
        predictions.to_json(json_filepath, orient='records', date_format='iso')
        
        # Generate sample HTML table
        sample_html = predictions.head(10).to_html(classes='table', table_id='predictions-table')
        
        # Store latest predictions globally for API access
        global LATEST_PREDICTIONS
        LATEST_PREDICTIONS = {
            'gru': predictions if model_name == 'gru' else None,
            'bidirectional_lstm': predictions if model_name == 'bidirectional_lstm' else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'model_used': model_name,
            'total_predictions': len(predictions),
            'unique_products': predictions['product'].nunique(),
            'filename': output_filename,
            'sample_html': sample_html
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download/<filename>')
def download_csv(filename):
    """Download CSV predictions"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/download/json/<filename>')
def download_json(filename):
    """Download JSON predictions for Power BI"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

# API Endpoints for Power BI Integration

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic predictions"""
    try:
        data = request.json
        model_name = data.get('model', 'gru')
        
        # Convert JSON data to DataFrame
        if 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        df = pd.DataFrame(data['data'])
        predictions = MODEL_SYSTEM.predict(df, model=model_name)
        
        return jsonify({
            'success': True,
            'model_used': model_name,
            'predictions': predictions.to_dict('records'),
            'total_predictions': len(predictions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/latest-forecasts')
def api_latest_forecasts():
    """Get latest forecasts for Power BI"""
    try:
        model_name = request.args.get('model', 'gru')
        
        # Return stored predictions if available
        if 'LATEST_PREDICTIONS' in globals() and LATEST_PREDICTIONS.get(model_name) is not None:
            predictions = LATEST_PREDICTIONS[model_name]
            return jsonify({
                'success': True,
                'model_used': model_name,
                'data': predictions.to_dict('records'),
                'total_records': len(predictions),
                'timestamp': LATEST_PREDICTIONS['timestamp']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No recent predictions available. Please upload data first.'
            }), 404
            
    except Exception as e:
        logger.error(f"API latest forecasts error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info')
def api_model_info():
    """Get model information for Power BI dashboards"""
    try:
        model_info = MODEL_SYSTEM.get_model_info()
        return jsonify({
            'success': True,
            'models': model_info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"API model info error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': MODEL_SYSTEM is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """Batch prediction endpoint for scheduled jobs"""
    try:
        # This endpoint can be called from scheduled tasks
        data_source = request.json.get('data_source', 'products_with_weekend.csv')
        model_name = request.json.get('model', 'gru')
        
        # Load data from specified source
        df = pd.read_csv(data_source)
        predictions = MODEL_SYSTEM.predict(df, model=model_name)
        
        # Save results with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"batch_forecasts_{model_name}_{timestamp}.csv"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_file)
        predictions.to_csv(output_path, index=False)
        
        return jsonify({
            'success': True,
            'batch_id': timestamp,
            'model_used': model_name,
            'total_predictions': len(predictions),
            'output_file': output_file,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

# Initialize models on startup
LATEST_PREDICTIONS = {'gru': None, 'bidirectional_lstm': None, 'timestamp': None}

if __name__ == '__main__':
    print("🚀 Starting Inventory Forecasting Flask Application")
    print("=" * 60)
    
    # Load models
    if load_models():
        print("✅ Models loaded successfully")
        print("🌐 Starting Flask server...")
        print("\nAccess points:")
        print("  Main Dashboard: http://localhost:5000")
        print("  API Health: http://localhost:5000/api/health")
        print("  Power BI Endpoint: http://localhost:5000/api/latest-forecasts?model=gru")
        print("\n" + "=" * 60)
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load models. Check that 'top_models_deployment' folder exists.")
        print("Run the training script first to create the models.")
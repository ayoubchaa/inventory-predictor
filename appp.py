<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inventory Forecasting System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .model-info {
            background: #ecf0f1;
            padding: 15px;
            text-align: center;
            font-size: 0.9em;
            color: #7f8c8d;
            border-bottom: 1px solid #bdc3c7;
        }

        .form-container {
            padding: 40px;
        }

        .form-group {
            margin-bottom: 25px;
        }

        .form-row {
            display: flex;
            gap: 20px;
            margin-bottom: 25px;
        }

        .form-row .form-group {
            flex: 1;
            margin-bottom: 0;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
            font-size: 0.95em;
        }

        input[type="number"],
        input[type="text"],
        select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #ecf0f1;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
            background-color: #fafafa;
        }

        input[type="number"]:focus,
        input[type="text"]:focus,
        select:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
            background-color: white;
        }

        .current-stock-group {
            background: #e8f5e8;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #27ae60;
        }

        .current-stock-group label {
            color: #27ae60;
            font-size: 1.1em;
        }

        .predict-btn {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            width: 100%;
            margin-top: 20px;
        }

        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(39, 174, 96, 0.3);
        }

        .predict-btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .result-container {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }

        .result-success {
            border-left: 4px solid #27ae60;
            background: #e8f5e8;
        }

        .result-error {
            border-left: 4px solid #e74c3c;
            background: #fdf2f2;
        }

        .prediction-value {
            font-size: 2em;
            font-weight: bold;
            color: #27ae60;
            text-align: center;
            margin: 15px 0;
        }

        .confidence-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            text-transform: uppercase;
        }

        .confidence-high {
            background: #d5f4e6;
            color: #27ae60;
        }

        .confidence-medium {
            background: #fff3cd;
            color: #856404;
        }

        .confidence-low {
            background: #f8d7da;
            color: #721c24;
        }

        .loading {
            text-align: center;
            padding: 20px;
        }

        .loading::after {
            content: '';
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .info-box {
            background: #e8f4f8;
            border: 1px solid #bee5eb;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 25px;
        }

        .info-box h3 {
            color: #0c5460;
            margin-bottom: 10px;
        }

        .info-box p {
            color: #0c5460;
            font-size: 0.9em;
            line-height: 1.5;
        }

        @media (max-width: 768px) {
            .form-row {
                flex-direction: column;
                gap: 0;
            }
            
            .container {
                margin: 10px;
            }
            
            .form-container {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏭 Inventory Forecasting System</h1>
            <p>AI-Powered Stock Prediction for Better Inventory Management</p>
        </div>
        
        <div class="model-info">
            Enhanced GRU Model | R² = 0.802 (80.2% Pattern Recognition Accuracy) | Advanced AI for Inventory Trend Analysis
        </div>

        <div class="form-container">
            <div class="info-box">
                <h3>📊 How to Use</h3>
                <p>Enter your current stock and product details below. If you don't know the current stock, leave it empty and the system will assume 0. The AI model will predict the recommended stock level based on historical patterns and trends.</p>
            </div>

            <form id="predictionForm">
                <div class="current-stock-group">
                    <div class="form-group">
                        <label for="current_stock">
                            📦 Current Stock Quantity *
                            <small style="font-weight: normal; color: #666;">(Leave empty for 0)</small>
                        </label>
                        <input type="number" 
                               id="current_stock" 
                               name="current_stock" 
                               min="0" 
                               step="0.01"
                               placeholder="Enter current stock (default: 0)">
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label for="max_stock">📈 Maximum Stock Capacity</label>
                        <input type="number" 
                               id="max_stock" 
                               name="max_stock" 
                               min="1" 
                               step="0.01"
                               value="100"
                               required>
                    </div>
                    
                    <div class="form-group">
                        <label for="cycle">🔄 Current Cycle</label>
                        <input type="number" 
                               id="cycle" 
                               name="cycle" 
                               min="0" 
                               max="52"
                               value="1"
                               required>
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label for="month">📅 Current Month</label>
                        <select id="month" name="month" required>
                            <option value="1">January</option>
                            <option value="2">February</option>
                            <option value="3">March</option>
                            <option value="4">April</option>
                            <option value="5">May</option>
                            <option value="6">June</option>
                            <option value="7">July</option>
                            <option value="8">August</option>
                            <option value="9">September</option>
                            <option value="10">October</option>
                            <option value="11">November</option>
                            <option value="12">December</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="group_name">🏷️ Product Group</label>
                        <input type="text" 
                               id="group_name" 
                               name="group_name" 
                               placeholder="Enter product group"
                               value="DEFAULT">
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label for="stock_class">⭐ Stock Class</label>
                        <input type="text" 
                               id="stock_class" 
                               name="stock_class" 
                               placeholder="Enter stock class"
                               value="DEFAULT">
                    </div>
                    
                    <div class="form-group">
                        <label for="storage_unit">🏪 Storage Unit</label>
                        <input type="text" 
                               id="storage_unit" 
                               name="storage_unit" 
                               placeholder="Enter storage unit"
                               value="DEFAULT">
                    </div>
                </div>

                <button type="submit" class="predict-btn" id="predictBtn">
                    🔮 Predict Stock Level
                </button>
            </form>

            <div id="resultContainer" class="result-container">
                <div id="loadingDiv" class="loading" style="display: none;">
                    Making prediction...
                </div>
                
                <div id="resultContent" style="display: none;">
                    <h3 style="text-align: center; margin-bottom: 15px;">📊 Prediction Result</h3>
                    
                    <div class="prediction-value" id="predictionValue">
                        0 units
                    </div>
                    
                    <div style="text-align: center; margin-bottom: 15px;">
                        <span>Confidence Level: </span>
                        <span id="confidenceBadge" class="confidence-badge">Medium</span>
                    </div>
                    
                    <div id="recommendationText" style="text-align: center; font-size: 1.1em; color: #2c3e50; margin-top: 15px;">
                    </div>
                </div>
                
                <div id="errorContent" style="display: none;">
                    <h3 style="color: #e74c3c; text-align: center;">❌ Prediction Error</h3>
                    <p id="errorMessage" style="text-align: center; margin-top: 10px;"></p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Set current month as default
        document.getElementById('month').value = new Date().getMonth() + 1;

        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            // Handle empty current_stock - set to 0 if empty
            if (!formData.get('current_stock') || formData.get('current_stock') === '') {
                formData.set('current_stock', '0');
            }
            
            // Show loading
            const resultContainer = document.getElementById('resultContainer');
            const loadingDiv = document.getElementById('loadingDiv');
            const resultContent = document.getElementById('resultContent');
            const errorContent = document.getElementById('errorContent');
            const predictBtn = document.getElementById('predictBtn');
            
            resultContainer.style.display = 'block';
            loadingDiv.style.display = 'block';
            resultContent.style.display = 'none';
            errorContent.style.display = 'none';
            predictBtn.disabled = true;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                loadingDiv.style.display = 'none';
                
                if (result.status === 'success') {
                    // Show success result
                    resultContainer.className = 'result-container result-success';
                    resultContent.style.display = 'block';
                    
                    // Update prediction value
                    document.getElementById('predictionValue').textContent = 
                        result.prediction.toLocaleString() + ' units';
                    
                    // Update confidence badge
                    const confidenceBadge = document.getElementById('confidenceBadge');
                    confidenceBadge.textContent = result.confidence.toUpperCase();
                    confidenceBadge.className = `confidence-badge confidence-${result.confidence}`;
                    
                    // Generate recommendation
                    const currentStock = parseFloat(formData.get('current_stock'));
                    const prediction = result.prediction;
                    let recommendation = '';
                    
                    if (prediction > currentStock * 1.2) {
                        recommendation = `📈 Consider increasing stock by ${(prediction - currentStock).toFixed(0)} units`;
                    } else if (prediction < currentStock * 0.8) {
                        recommendation = `📉 Consider reducing stock by ${(currentStock - prediction).toFixed(0)} units`;
                    } else {
                        recommendation = `✅ Current stock level appears optimal`;
                    }
                    
                    document.getElementById('recommendationText').textContent = recommendation;
                    
                } else {
                    // Show error
                    resultContainer.className = 'result-container result-error';
                    errorContent.style.display = 'block';
                    document.getElementById('errorMessage').textContent = 
                        result.message || 'An error occurred during prediction';
                }
                
            } catch (error) {
                loadingDiv.style.display = 'none';
                resultContainer.className = 'result-container result-error';
                errorContent.style.display = 'block';
                document.getElementById('errorMessage').textContent = 
                    'Failed to connect to prediction service';
            }
            
            predictBtn.disabled = false;
        });

        // Add some interactivity
        document.getElementById('current_stock').addEventListener('input', function() {
            const value = this.value;
            const maxStock = document.getElementById('max_stock');
            
            if (value && !maxStock.value) {
                maxStock.value = Math.max(100, value * 2);
            }
        });
    </script>
</body>
</html>

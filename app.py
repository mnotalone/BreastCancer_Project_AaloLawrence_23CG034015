from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model artifacts
MODEL_PATH = 'model/breast_cancer_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
FEATURES_PATH = 'model/feature_names.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure model files exist.',
                'success': False
            })
        
        # Get form data
        radius_mean = float(request.form.get('radius_mean'))
        texture_mean = float(request.form.get('texture_mean'))
        area_mean = float(request.form.get('area_mean'))
        concavity_mean = float(request.form.get('concavity_mean'))
        symmetry_mean = float(request.form.get('symmetry_mean'))
        
        # Prepare input array
        input_data = np.array([[
            radius_mean,
            texture_mean,
            area_mean,
            concavity_mean,
            symmetry_mean
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get decision function score (confidence)
        confidence_score = model.decision_function(input_scaled)[0]
        
        # Prepare response
        if prediction == 1:
            result = "Benign"
            message = "The tumor is predicted to be benign (non-cancerous)."
            result_class = "benign"
        else:
            result = "Malignant"
            message = "The tumor is predicted to be malignant (cancerous)."
            result_class = "malignant"
        
        return jsonify({
            'success': True,
            'prediction': result,
            'message': message,
            'result_class': result_class,
            'confidence': abs(float(confidence_score)),
            'input_values': {
                'Mean Radius': radius_mean,
                'Mean Texture': texture_mean,
                'Mean Area': area_mean,
                'Mean Concavity': concavity_mean,
                'Mean Symmetry': symmetry_mean
            }
        })
        
    except ValueError as ve:
        return jsonify({
            'error': 'Invalid input values. Please enter numeric values only.',
            'success': False
        })
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
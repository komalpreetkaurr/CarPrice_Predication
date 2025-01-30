from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load and process the dataset
try:
    car = pd.read_csv('clean.csv')

    # Encode categorical variables
    label_encoder_company = LabelEncoder()
    label_encoder_fuel = LabelEncoder()

    car['company'] = label_encoder_company.fit_transform(car['company'].str.strip())
    car['fuel_type'] = label_encoder_fuel.fit_transform(car['fuel_type'].str.strip())

    # Define features (X) and target (y)
    X = car[['company', 'year', 'kms_driven', 'fuel_type']]
    y = car['Price']

    # Train a model
    model = LinearRegression()
    model.fit(X, y)
except FileNotFoundError as e:
    raise RuntimeError(f"Required file missing: {e}")
except Exception as e:
    raise RuntimeError(f"Error loading or processing data: {e}")


@app.route('/', methods=['GET'])
def index():
    """Render the home page with car details for prediction."""
    try:
        companies = sorted(label_encoder_company.inverse_transform(car['company'].unique()))
        years = sorted(car['year'].unique(), reverse=True)
        fuel_types = sorted(label_encoder_fuel.inverse_transform(car['fuel_type'].unique()))

        return render_template(
            'index.html',
            companies=companies,
            years=years,
            fuel_types=fuel_types
        )
    except Exception as e:
        return f"Error in loading data: {e}", 500


@app.route('/get_models', methods=['POST'])
@cross_origin()
def get_models():
    """Return car models for the selected company."""
    try:
        company = request.json.get('company')
        if not company or company == "Select Company":
            return jsonify({"error": "Invalid company selected."}), 400

        # Encode the selected company
        company_encoded = label_encoder_company.transform([company.strip()])[0]
        models = car[car['company'] == company_encoded]['name'].unique().tolist()

        return jsonify({"models": sorted(models)})
    except Exception as e:
        return jsonify({"error": f"Error fetching models: {e}"}), 500


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    """Handle the prediction logic."""
    try:
        # Get form inputs
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        driven = request.form.get('kilo_driven')

        # Validate inputs
        if not company or company == "Select Company" or not car_model or not year or not fuel_type or not driven:
            return jsonify({"error": "All fields are required and must be valid."}), 400

        # Convert inputs to proper data types
        year = int(year)
        driven = int(driven)

        # Encode inputs
        company_encoded = label_encoder_company.transform([company.strip()])[0]
        fuel_encoded = label_encoder_fuel.transform([fuel_type.strip()])[0]

        input_data = pd.DataFrame(columns=['company', 'year', 'kms_driven', 'fuel_type'],
                                  data=[[company_encoded, year, driven, fuel_encoded]])

        # Make the prediction
        prediction = model.predict(input_data)
        predicted_price = np.round(prediction[0], 2)

        # Return the predicted price as part of the response
        return jsonify({"predicted_price": predicted_price})
    except ValueError as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {e}"}), 500



if __name__ == '__main__':
    app.run(debug=True)

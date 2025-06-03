from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import joblib
import os

app = Flask(__name__)
app.secret_key = "calorie_secret_key"  # Needed for flash messages

# Load the trained model
MODEL_PATH = 'calorie_prediction_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None, input_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        flash("Model file not found. Please train and save the model first.", "danger")
        return redirect(url_for('home'))

    try:
        # Get form data
        gender = int(request.form['gender'])  # 0 for male, 1 for female
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])

        features = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])
        prediction = model.predict(features)[0]

        # Enhanced feedback based on calories burnt
        if prediction < 100:
            feedback = "Low activity. Try increasing your workout intensity!"
            emoji = "💤"
        elif prediction < 300:
            feedback = "Moderate activity. Good job, keep going!"
            emoji = "💪"
        else:
            feedback = "High activity! Excellent work!"
            emoji = "🔥"

        prediction_text = f"{emoji} Estimated Calories Burnt: <b>{prediction:.2f}</b> kcal<br><span style='font-size:1.1em;'>{feedback}</span>"

        # Pass input data back for user review
        input_data = {
            'gender': gender,
            'age': age,
            'height': height,
            'weight': weight,
            'duration': duration,
            'heart_rate': heart_rate,
            'body_temp': body_temp
        }

        return render_template(
            'index.html',
            prediction_text=prediction_text,
            input_data=input_data
        )
    except Exception as e:
        flash(f"Error: {e}", "danger")
        return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
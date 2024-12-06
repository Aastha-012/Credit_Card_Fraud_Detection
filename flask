# Flask Application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON data
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Predict using the trained model
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Return response
        return jsonify({'fraud': int(prediction), 'probability': probability})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

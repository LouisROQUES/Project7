# Dependencies
import sys

import query as query
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if xgbc_model:
        try:
            json_ = request.json
            prediction = list(xgbc_model.predict_proba(query))
            print(json_)
            query = pd.DataFrame(json_)
            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    xgbc_model = joblib.load('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/gbc_model.pkl') # Load "model.pkl"
    print ('Model loaded')

    app.run(port=port, debug=True)
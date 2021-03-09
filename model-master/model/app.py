from flask import Flask, request, jsonify,render_template
from sklearn.externals import joblib
import pickle
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():

	lr = joblib.load("reg.pkl") # Load "model.pkl"

	model_columns = joblib.load("model_reg.pkl") # Load "model_columns.pkl"

	if lr:
		try:
			#

			json_ = request.json
			# print(json_)
			query = pd.DataFrame(json_, index=[0])
			query = query.reindex(columns=model_columns, fill_value=0)

			print(query)
			prediction = lr.predict(query).round().astype(int)

			return jsonify({'prediction': str(prediction[0][0])})
		except:
			return jsonify({'trace': traceback.format_exc()})
			#K.clear_session()	
	else:
		print ('Train the model first')
		return ('No model here to use')


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    app.run(port=port, debug=False,threaded=False)




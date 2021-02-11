from flask import Flask, render_template, request
import pickle
import numpy as np

filename = 'optimizing-agriculture-production-logistic-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        n = int(request.form.get("N", False))
        p = int(request.form.get("P", False))
        k = int(request.form.get("K", False))
        temp = float(request.form.get("temperature", False))
        humidity = float(request.form.get("humidity", False))
        ph = float(request.form.get("ph", False))
        rainfall = float(request.form.get("rainfall", False))
       
        
        my_prediction = classifier.predict((np.array([[n,
                                                       p,
                                                       k,
                                                       temp,
                                                       humidity,
                                                       ph,
                                                       rainfall]])))
	    	
        return render_template('result.html', prediction=my_prediction)
    
if __name__ == '__main__':
    app.run(debug=True)
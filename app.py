from flask import Flask, render_template, request
import pickle
import numpy as np

filename = 'optimizing-agriculture-production-rf-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        n = float(request.form.get("N", False))
        p = float(request.form.get("P", False))
        k = float(request.form.get("K", False))
#        temp = float(request.form.get("temperature", False))
        humidity = float(request.form.get("humidity", False))
#        ph = float(request.form.get("ph", False))
        rainfall = float(request.form.get("rainfall", False))
       
        
        my_prediction = classifier.predict((np.array([[n,
                                                       p,
                                                       k,
                                                       humidity,
                                                       rainfall]])))
	    	
        return render_template('result.html', prediction=my_prediction)
    
if __name__ == '__main__':
    app.run(debug=True)

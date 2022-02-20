import numpy as np
from flask import Flask, request, render_template
import pickle

model= pickle.load(open('breast_flask.pkl', 'rb'))

app= Flask(__name__)

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods= ['POST'])
def index():
    data1= request.form['a']
    data2= request.form['b']
    data3= request.form['c']
    data4= request.form['d']
    data5= request.form['e']
    data6= request.form['f']
    data7= request.form['g']
    data8= request.form['h']
    arr = np.array([[data1,data2,data3,data4,data5,data6,data7,data8]])
    pred= model.predict(arr)
    return render_template('after.html', data=pred)
        

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
    
    
    
     
        
        

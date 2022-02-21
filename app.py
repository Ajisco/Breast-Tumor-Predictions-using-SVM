import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from flask import Flask, request, render_template

app= Flask(__name__)
cancer = load_breast_cancer()
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
X= df_feat[['mean radius','mean perimeter','mean area','mean concave points',
            'worst radius','worst perimeter','worst area','worst concave points']]
y= cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
model = SVC()
model.fit(X_train,y_train)
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
model= grid.fit(X_train,y_train)



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
    app.run(debug= True, use_reloader=False)
    
    
     
        
        
    
    
    
     
        
        

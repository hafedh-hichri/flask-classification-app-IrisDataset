from flask import Flask , request , render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# creating and traing the model 
df = pd.read_csv('./irisDataset.csv')

le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = SVC(probability=True)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))


# flask app 


# to run this program open your CMD.exe and cd into the folder 
# type in your cmd :   python app.exe     
# open your browser and open 127.0.0.1:5000

app = Flask(__name__)

@app.route('/', methods=["GET","POST"])
def intro():
    if request.method =="POST":
        if request.form["sepal_length"] and request.form["sepal_width"] and request.form["petal_length"] and request.form["petal_width"]:

            x1 = float(request.form["sepal_length"])
            x2 = float(request.form["sepal_width"])
            x3 = float(request.form["petal_length"])
            x4 = float(request.form["petal_width"])
            
            x = pd.DataFrame([[x1, x2,x3,x4]], columns=[
                             'sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
            print(x)
            pred_num = model.predict(x)
            proba = model.predict_proba(x)
            proba = proba[0][pred_num]
            pred = le.inverse_transform(pred_num)
            return render_template('animatedlogin.html', pred=str(pred), proba = str(proba))
    return render_template('animatedlogin.html')





if __name__ == "__main__" :
    app.run(debug=True)

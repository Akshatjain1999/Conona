from flask import Flask,render_template,request
app = Flask(__name__)
import pickle
file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()
@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method =="POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain=int(myDict['pain'])
        runnyNose = int(myDict['runny nose'])
        diffBreath =int(myDict['diffBreath'])
        inputFeatures = [fever,pain,age,runnyNose,diffBreath]
        predict=clf.predict_proba([inputFeatures])[0][1]
        return render_template('show.html',inf=round(predict*100,2))
    return render_template('index.html')
    # return str(predict)
if __name__=='__main__':
    app.run(debug=True)
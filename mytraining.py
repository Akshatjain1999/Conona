import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

if __name__ == '__main__':
    df = pd.read_excel('convid19.xlsx')
    X=df.drop('Infection Prob',axis=1)
    y=df[['Infection Prob']]
    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.25,random_state=42,shuffle=True)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    y_train = y_train.reshape(2165,)
    y_test = y_test.reshape(722,)
    lr = LogisticRegression()
    lr.fit(X_train,y_train)

    file = open('model.pkl','wb')
    pickle.dump(lr,file)
    file.close()
    inputFeatures = [100,1,23,-1,1]
    predict=lr.predict_proba([inputFeatures])[0][1]
    # print(predict)

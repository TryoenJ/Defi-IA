import numpy as np
import pandas as pd
import panel as pn
import sys  	
import os

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from joblib import dump, load

#########################
#    MAIN PROGRAM
#########################

if __name__ == '__main__':
    print("***************")
    print("*** MSE.py ***")
    
    path = './data/'
    
    data = pd.read_csv(path+'/new_result_requetes.csv')
    
    # create a new outputs directory
    if not(os.path.isdir('./results_store/')):
      os.makedirs('./results_store/')
      
    # PREPROCESSING
    print("*** preprocessing ***")
    #On affecte le bon type aux variables qualitatives
    data["avatar_id"]=pd.Categorical(data["avatar_id"],ordered=False)
    data["city"]=pd.Categorical(data["city"],ordered=False)
    data["language"]=pd.Categorical(data["language"],ordered=False)
    data["mobile"]=pd.Categorical(data["mobile"],ordered=False)
    data["hotel_id"]=pd.Categorical(data["hotel_id"],ordered=False)
    data["group"]=pd.Categorical(data["group"],ordered=False)
    data["brand"]=pd.Categorical(data["brand"],ordered=False)
    data["parking"]=pd.Categorical(data["parking"],ordered=False)
    data["pool"]=pd.Categorical(data["pool"],ordered=False)
    data["children_policy"]=pd.Categorical(data["children_policy"],ordered=False)

    data["price"]=pd.DataFrame(data["price"], dtype=float)
    
    p#rint("*** model with all the inputs except nb_requete ***")
    #X = data[["date","stock","city","language","mobile","hotel_id","group","brand","parking","pool","children_policy"]]
    
    print("*** model with the 7 most influent inputs ***")
    X = data[["date","stock","city","language","hotel_id","group","brand"]]
    Y = data[["price"]]
    
    # PIPELINE
    print("   ")
    print("*** model with TargetEncoder+XGBRegressor n_estimators=3000 and max_depth=10 ***")
    
    pip = Pipeline(steps=[("Cat_encoder", TargetEncoder()),
                      ("Standard_scaler", StandardScaler()),
                      ("Boosting", XGBRegressor(n_estimators=3000,max_depth=10)),
                      ]
               )
        
    ansB = input("* How many fold ? (type 5 or 10)")
    B = int(ansB)
    
    print("*** MSE en fonction de la date pour B folds ***")
    
    date = np.linspace(0,44,45)
    
    Vect_erreur = np.zeros((B,len(date),2))

    for b in range(0,B):
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
        pip.fit(X_train, y_train)
        pred = pip.predict(X_test)

        x = np.ravel(X_test[['date']])

        for i in range(len(x)):
            d = x[i]
            yt = y_test[i]
            p = pred[i]
            erreur = (yt-p)**2
            Vect_erreur[b,d,0] += erreur
            Vect_erreur[b,d,1] += 1

        for i in range(Vect_erreur.shape[1]):
            Vect_erreur[b,i,0] = Vect_erreur[b,i,0]/Vect_erreur[b,i,1] 
    
    arrayErreur = np.zeros((B,45))
    for b in range(B):
        arrayErreur[b,:]=Vect_erreur[b,:,0]
    
    dataframeErreur = pd.DataFrame(arrayErreur)
    plt.figure(figsize=(13,7))
    dataframeErreur.boxplot()
    #plt.show(block=False)
    plt.savefig('./results_store/MSE-BP-date.png')


    print("*** MSE en fonction de la ville pour B folds ***")

    listCity = {'amsterdam':0, 'copenhagen':1, 'madrid':2, 'paris':3, 'rome':4, 'sofia':5, 'valletta':6, 'vienna':7, 'vilnius':8}
    
    Vect_erreur = np.zeros((B,len(listCity),2))

    for b in range(0,B):
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
        pip.fit(X_train, y_train)
        pred = pip.predict(X_test)

        x = np.ravel(X_test[['city']])
    
        for i in range(len(x)):
            c = x[i]
            yt = y_test[i]
            p = pred[i]
            erreur = (yt-p)**2
            Vect_erreur[b,listCity[c],0] += erreur
            Vect_erreur[b,listCity[c],1] += 1

        for i in range(Vect_erreur.shape[1]):
            Vect_erreur[b,i,0] = Vect_erreur[b,i,0]/Vect_erreur[b,i,1] 
    
    arrayErreur = np.zeros((B,len(listCity)))
    for b in range(B):
        arrayErreur[b,:]=Vect_erreur[b,:,0]
    
    dataframeErreur = pd.DataFrame(arrayErreur,columns=['amsterdam', 'copenhagen', 'madrid', 'paris', 'rome', 'sofia', 'valletta', 'vienna', 'vilnius'])
    plt.figure(figsize=(13,7))
    dataframeErreur[['amsterdam', 'copenhagen', 'madrid', 'paris', 'rome', 'sofia', 'valletta', 'vienna', 'vilnius']].boxplot(return_type='dict')
    #plt.show(block=False)
    plt.savefig('./results_store/MSE-BP-city.png')
    
    
    print("*** MSE en fonction du langage pour B folds ***")
    
    listLang = {'austrian':0, 'belgian':1, 'bulgarian':2, 'croatian':3, 'cypriot':4, 'czech':5, 'danish':6, 'dutch':7, 'estonian':8, 'finnish':9,
            'french':10, 'german':11, 'greek':12, 'hungarian':13, 'irish':14, 'italian':15, 'latvian':16, 'lithuanian':17, 'luxembourgish':18, 
            'maltese':19, 'polish':20, 'portuguese':21, 'romanian':22, 'slovakian':23, 'slovene':24, 'spanish':25,'swedish':26}

    Vect_erreur = np.zeros((B,len(listLang),2))
    
    for b in range(0,B):
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
        pip.fit(X_train, y_train)
        pred = pip.predict(X_test)
        
        x = np.ravel(X_test[['language']])
        
        for i in range(len(x)):
            c = x[i]
            yt = y_test[i]
            p = pred[i]
            erreur = (yt-p)**2
            Vect_erreur[b,listLang[c],0] += erreur
            Vect_erreur[b,listLang[c],1] += 1

        for i in range(Vect_erreur.shape[1]):
            Vect_erreur[b,i,0] = Vect_erreur[b,i,0]/Vect_erreur[b,i,1] 
        
        dataframeErreur = pd.DataFrame(arrayErreur,columns=['austrian', 'belgian', 'bulgarian', 'croatian', 'cypriot', 'czech', 'danish', 'dutch', 'estonian', 'finnish',
            'french', 'german', 'greek', 'hungarian', 'irish', 'italian', 'latvian', 'lithuanian', 'luxembourgish', 
            'maltese', 'polish', 'portuguese', 'romanian', 'slovakian', 'slovene', 'spanish','swedish'])
        plt.figure(figsize=(13,7))
        dataframeErreur[['austrian', 'belgian', 'bulgarian', 'croatian', 'cypriot', 'czech', 'danish', 'dutch', 'estonian', 'finnish',
            'french', 'german', 'greek', 'hungarian', 'irish', 'italian', 'latvian', 'lithuanian', 'luxembourgish', 
            'maltese', 'polish', 'portuguese', 'romanian', 'slovakian', 'slovene', 'spanish','swedish']].boxplot(return_type='dict')
        #plt.show(block=False)
        plt.savefig('./results_store/MSE-BP-langage.png')
        
    print("*** end MSE.py ***")
import numpy as np
import pandas as pd
import panel as pn
import sys  	
import os
import time

import matplotlib.pyplot as plt

from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from joblib import dump, load

import pickle

class TargetEncoderSmooth(BaseEstimator, TransformerMixin):
    """Target encoder.
    
    Replaces categorical column(s) with the mean target value for
    each category.

    """
    
    def __init__(self, cols=None):
        """Target encoder
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.  Default is to target 
            encode all categorical columns in the DataFrame.
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        
        
    def fit(self, X, y):
        """Fit target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X 
                         if str(X[col].dtype)=='category']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Compute the global mean
        mean = y.mean()

        # Encode each element of each column
        self.maps = dict() #dict to store map for each column
        for col in self.cols:
            tmap = dict()
            uniques = X[col].unique()
            for unique in uniques:
                #tmap[unique] = y[X[col]==unique].mean()
                counts = pd.DataFrame(y[X[col]==unique]).count()
                means = pd.DataFrame(y[X[col]==unique]).mean()
                tmap[unique] = (counts * means + 100 * mean)/(counts + 100)
            self.maps[col] = tmap
            
        return self

        
    def transform(self, X, y=None):
        """Perform the target encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, tmap in self.maps.items():
            vals = np.full(X.shape[0], np.nan)
            for val, mean_target in tmap.items():
                vals[X[col]==val] = mean_target
            Xo[col] = vals
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)


#########################
#    MAIN PROGRAM
#########################

if __name__ == '__main__':
    print("***************")
    print("*** train.py ***")
    
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
    
    ansI = input("* Do we take the 12 inputs to build the model ? all the inputs except nb_requete ? the 7 most influent inputs ? (type 12 or 11 or 10 or 7)")
    if ansI=='12':
        X = data[["nb_requete","date","stock","city","language","mobile","hotel_id","group","brand","parking","pool","children_policy"]]
    elif ansI == '11':
        X = data[["date","stock","city","language","mobile","hotel_id","group","brand","parking","pool","children_policy"]]
    elif ansI == '10':
        X = data[["date","stock","city","language","mobile","group","brand","parking","pool","children_policy"]]
    else :
        X = data[["date","stock","city","language","hotel_id","group","brand"]]
    
    Y = data[["price"]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    
    # PIPELINE
    print("   ")
    print("*** model with " + ansI + " inputs ***")
    ansC = input("* Do we use OneHotEncoder or Classical TargetEncoder or Smooth TargetEncoder ? (type O or CT or ST)")
    if ansC=='O':
        Preprocessor = ColumnTransformer(
            transformers=[
                ("Cat_encoder", OneHotEncoder(), selector(dtype_include="category")),
                ("Standard_scaler", StandardScaler(), selector(dtype_exclude="category")),
                ]
            )
        
        print("*** XGBRegressor with OneHotEncoder without GridSearch ***")
        
        pip = Pipeline(steps=[("Preprocessor", Preprocessor),
                      ("Boosting", XGBRegressor(n_estimators=3000,max_depth=10)),
                      ]
               )
        
        #print(pip)
        
        print("*** model training ***")
        t0 = time.perf_counter()
        pip.fit(X_train, y_train)
        
    else : 
        
        if ansC=='CT':
            Categorical_transformer = TargetEncoder()
            print("*** XGBRegressor with Classical TargetEncoder ***")
            ansP = input("* Do we use no GridSearch, GridSearch for XGB, GridSearch for Classical TargetEncoder? (type NG or XG or TG)")
        else :
            ansC=='ST'
            Categorical_transformer = TargetEncoderSmooth()
            print("*** XGBRegressor with Smooth TargetEncoder with weight=100 ***")
            ansP = input("* Do we use no GridSearch or GridSearch for XGB? (type NG or XG)")

    
        if ansP == 'NG':
            
            print("*** no GridSearch ***")
            print("*** XGBRegressor with n_estimators=3000 and max_depth=10 ***")
            print("*** model training ***")
        
            pip = Pipeline(steps=[("Cat_encoder", Categorical_transformer),
                                  ("Standard_scaler", StandardScaler()),
                                  ("Boosting", XGBRegressor(n_estimators=3000,max_depth=10)),
                                  ]
                           )
        
            #print(pip)
            t0 = time.perf_counter()
            pip.fit(X_train, y_train)
    
        else :
        
            if ansP == 'XG':
        
                print("*** GridSearch for XGBoost ***")
                print("*** model training ***")
            
                pip = Pipeline(steps=[("Cat_encoder", Categorical_transformer),
                                      ("Standard_scaler", StandardScaler()),
                                      ("Boosting", XGBRegressor()),
                                      ]
                               )
        
                param_grid = {
                    "Boosting__n_estimators":[1000, 3000, 5000],
                    "Boosting__max_depth": [7, 10, 13],
                    }
             
            else :
                ansP == 'TG'
                print("*** GridSearch for Classical Target ***")
                print("*** model training ***")
                    
                pip = Pipeline(steps=[("Cat_encoder", Categorical_transformer),
                                      ("Standard_scaler", StandardScaler()),
                                      ("Boosting", XGBRegressor(n_estimators=3000,max_depth=10)),
                                      ]
                               )
            
                param_grid = {
                    "Cat_encoder__min_samples_leaf":[1,10,20],
                    "Cat_encoder__smoothing": [0.2,1.,10.],
                    }
            
            grid_search = GridSearchCV(pip, param_grid, cv=5, n_jobs=-1)
            
            #print(grid_search)
            t0 = time.clock()
            grid_search.fit(X_train, y_train)
            
            print("Best params:")
            print(grid_search.best_params_)
        
            print(f"Internal CV score: {grid_search.best_score_:.3f}")
            print(f"GridSearch score: {grid_search.score(X_test, y_test):.3f}")
    
            pip = grid_search
    
    t1 = time.perf_counter()
    time = t1-t0
    
    print("computational time: "+str(time)+" seconds")
    
    print("   ")
    print("*** plots ***")
    y_hat = pip.predict(X_test)

    plt.figure(1)
    plt.plot(y_hat,y_test,"o")
    plt.xlabel(u"predict prices")
    plt.ylabel(u"observed prices")
    #plt.show(block=False)
    plt.savefig('./results_store/XGB'+ ansI + ansC +'_ytest-vs-yhat.png')
    
    plt.figure(2)
    plt.plot(y_hat,y_test-y_hat,"o")
    plt.xlabel(u"predict prices")
    plt.ylabel(u"residuals")
    plt.hlines(0,0,500)
    #plt.show(block=False)
    plt.savefig('./results_store/XGB'+ ansI + ansC +'_residuals-vs-yhat.png')
        
    print("*** RMSE + R2 ***")
    print("RMSE=",mean_squared_error(y_test, y_hat, squared=False))  
    print("R2=",r2_score(y_test,y_hat))
    
    print("   ")
    print("*** save ***")
    ansM = input("* Do we save the model ? (type y or n)")
    if ansM=='y':
        #dump(pip,'XGB'+ ansI + ansC +'_model_saved_Final.joblib')
        #pickle.dump(pip, open('XGB'+ ansI + ansC +'_model_saved_Final.sav', 'wb'))
        pickle.dump(pip, open('XGB'+ ansI + ansC +'_model_saved_Final.pkl', 'wb'))
    
    print("   ")
    print("*** predict test Defi ***")
    ansD = input("* Do we predict prices for the Defi data ? (type y or n)")
    if ansD=='y':
        test = pd.read_csv('./data/test_set3.csv')
        #On affecte le bon type aux variables qualitatives
        test["avatar_id"]=pd.Categorical(test["avatar_id"],ordered=False)
        test["city"]=pd.Categorical(test["city"],ordered=False)
        test["language"]=pd.Categorical(test["language"],ordered=False)
        test["mobile"]=pd.Categorical(test["mobile"],ordered=False)
        test["hotel_id"]=pd.Categorical(test["hotel_id"],ordered=False)
        test["group"]=pd.Categorical(test["group"],ordered=False)
        test["brand"]=pd.Categorical(test["brand"],ordered=False)
        test["parking"]=pd.Categorical(test["parking"],ordered=False)
        test["pool"]=pd.Categorical(test["pool"],ordered=False)
        test["children_policy"]=pd.Categorical(test["children_policy"],ordered=False)
        
        if ansI=='12':
            T = test[["nb_requete","date","stock","city","language","mobile","hotel_id","group","brand","parking","pool","children_policy"]]
        elif ansI == '11':
            T = test[["date","stock","city","language","mobile","hotel_id","group","brand","parking","pool","children_policy"]]
        else :
            T = test[["date","stock","city","language","hotel_id","group","brand"]]
       
        Yprev=pip.predict(T)
        Yprev0=np.around(Yprev, decimals=1)
        Yindice=pd.DataFrame(np.arange(0,len(Yprev)),columns = ['index'])
        Yprice=pd.DataFrame(Yprev0,columns = ['price'])
        Ysub=Yindice.join(Yprice, on=None, how='right', lsuffix='', rsuffix='', sort=False)
        Ysub.to_csv('./results_store/submissionXGB.csv', index= False)

    print("*** end train.py ***")
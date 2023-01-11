import gradio as gr
import pandas as pd
import joblib
import datasets
import numpy as np
import warnings

warnings.filterwarnings("ignore")

model = joblib.load('./XGB11_Target_model_saved_Final.pkl')

in1 = gr.inputs.Slider(minimum=0, maximum=44, step=1)#date
in2 = gr.inputs.Slider(minimum=1, maximum=100, step=1) #stock
#in3 = gr.inputs.Dropdown(choices=['paris', 'copenhagen', 'madrid', 'rome', 'sofia', 'vilnius', 'vienna', 'amsterdam', 'valletta'])
in4 = gr.inputs.Dropdown(choices=['romanian', 'swedish', 'maltese', 'belgian', 'luxembourgish',
       'dutch', 'french', 'finnish', 'austrian', 'slovakian', 'hungarian',
       'bulgarian', 'danish', 'greek', 'croatian', 'polish', 'german',
       'spanish', 'estonian', 'lithuanian', 'cypriot', 'latvian', 'irish',
       'italian', 'slovene', 'czech', 'portuguese'])
in5 = gr.inputs.Radio(["0","1"])#mobile
in6 = gr.inputs.Slider(minimum=0, maximum=998, step=1) #hotel_id

inputs = [in1, in2, in4, in5, in6]
#inputs = [in1, in2, in3, in4, in5, in6]
outputs = "text"


def infer(date,stock,language,mobile,hotel_id):#city

  input_dataframe = pd.DataFrame([date,stock,language,mobile,hotel_id]).T #city
  input_dataframe.columns = ["date","stock","language","mobile","hotel_id"] #,"city"

  #ajout des features_hotels
  #hotels = pd.read_csv('/content/gdrive/My Drive/Defi-IA/features_hotels.csv', index_col=['hotel_id', 'city'])
  #input_complete = input_dataframe.join(hotels, on=['hotel_id', 'city'])
  hotels = pd.read_csv('./data/features_hotels.csv', index_col=['hotel_id'])
  #hotels = hotels.drop(['city'],axis=1)
  input_complete = input_dataframe.join(hotels, on=['hotel_id'])

  #On affecte le bon type aux variables quantitatives
  input_complete["date"]=input_complete["date"].astype('int64')
  input_complete["stock"]=input_complete["stock"].astype('int64')

  #On affecte le bon type aux variables qualitatives
  input_complete["city"]=pd.Categorical(input_complete["city"],ordered=False)
  input_complete["language"]=pd.Categorical(input_complete["language"],ordered=False)
  input_complete["mobile"]=pd.Categorical(input_complete["mobile"],ordered=False)
  input_complete["hotel_id"]=pd.Categorical(input_complete["hotel_id"],ordered=False)
  input_complete["group"]=pd.Categorical(input_complete["group"],ordered=False)
  input_complete["brand"]=pd.Categorical(input_complete["brand"],ordered=False)
  input_complete["parking"]=pd.Categorical(input_complete["parking"],ordered=False)
  input_complete["pool"]=pd.Categorical(input_complete["pool"],ordered=False)
  input_complete["children_policy"]=pd.Categorical(input_complete["children_policy"],ordered=False)
  #print(input_complete)
  return "You, as a/an "+str(language)+" speaker, chose to spend a night in hotel "+ str(hotel_id)+ " situated in "+str(input_complete["city"][0])+" in "+str(date)+" day(s). You supposed that "+str(input_complete["stock"][0])+ " rooms are still available. We estimate the price of your night at : " + str(model.predict(input_complete)[0])
#infer(1,10,'finnish',3,999)

I = gr.Interface(fn = infer, inputs = inputs, outputs = "text")
I.launch(share=True)

#debug=True, share=True
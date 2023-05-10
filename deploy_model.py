from numbers import Real
import tabpy
from tabpy.tabpy_tools.client import Client
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

np.set_printoptions(precision=2)

sq_model = tf.keras.saving.load_model('C:\\models\\model_10_05.h5')
sq_model.compile(loss="sparse_categorical_crossentropy", optimizer= "Adam", metrics=['accuracy'])


def arytmia_classification(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8, _arg9, _arg10, arg11, _arg12, _arg13):
  input_data = np.column_stack([_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8, _arg9, _arg10, arg11, _arg12, _arg13])
  x = pd.DataFrame(input_data, columns=['PatientAge',	
                                        'Gender',	
                                        'VentricularRate', 
                                        'AtrialRate',	
                                        'QRSDuration',	
                                        'QTInterval',	
                                        'QTCorrected',	
                                        'RAxis',	
                                        'TAxis',	
                                        'QRSCount',	
                                        'QOnset',	
                                        'QOffset',	
                                        'TOffset'])
  y = sq_model.predict(x)
  prob = np.amax(y)
  prob = float(prob)
  result = np.argmax(y)
  rhythm = ''
  if result == 0: 
     rhythm = 'AFIB'
  elif result == 1: 
     rhythm ='SB'
  elif result == 2:
     rhythm ='SR'
  else: rhythm ='ST'

  return prob
  

client = Client('http://localhost:9004/')
# Connect to TabPy server using the client library
connection = Client('http://localhost:9004/') 
connection.deploy('arytmia_classification_v4_prob', arytmia_classification, 'Returns probability of arytmia', override=True)

import six
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow import keras

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

#from qkeras import *
#from qkeras.utils import model_save_quantized_weights


import tempfile, os
from quantize import * 

import csv
 
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def show(img):
  plt.imshow(np.reshape(img, (16,-1)))
  plt.show()

def show_dist(x):
  plt.hist(x)
  plt.show()


f_out = '/input_val_q'
#f_out_fmt = 'csv'
f_out_fmt = 'hex'

def get_csv(f_in):
    list=[]
    with open(f_in, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            for r in row:
                if len(r) > 0:
                    list.append(float(r))

    arr = np.asarray(list)
#    arr = np.array(list(csv.reader(f_in, delimiter=',')))
    print("length of arr:", len(arr))
    return arr

def main():

    id_01 = np.array([])
    id_02 = np.array([])
    id_03 = np.array([])
    id_04 = np.array([])
    
    for id_str in range(1,5): 
      idx=0
      if id_str == 1 :
        for image in range(614): # 614 files : id_01
        
             data=get_csv('./input_val/sample_id_0{id_str}_{num}.csv'.format(id_str=id_str,num=idx))     
             print(data.shape)
             
             id_01 = np.append(id_01, data)
             
             output=[]
             arr = np.asarray(data)
             train_range=120
             output = quantize(arr.flatten(), 8, train_range) #note) to fix one scalefactor for all imaages 
             save_file(output, './input_val_q/sample_id_0{id_str}_{num}'.format(id_str=id_str,num=idx), f_out_fmt)
#            np.savetxt(path_for_save+filename+str(idx)+'.csv', np.asarray(list_val), delimiter=",", fmt='%1.3d')
             idx +=1       
             
             
      else :
             
        for image in range(615): # 615 files : id_02,id_03,id_04 

          data=get_csv('./input_val/sample_id_0{id_str}_{num}.csv'.format(id_str=id_str,num=idx))         
          
          if id_str==2 :  
            id_02 = np.append(id_02, data) #for describe
            
          if id_str==3 :  
            id_03 = np.append(id_03, data) #for describe  
          
          if id_str==4 : 
            id_04 = np.append(id_04, data) #for describe  
          

          
          output=[]
          arr = np.asarray(data)
          print(arr.shape)
          train_range=160
          output = quantize(arr.flatten(), 8, train_range) #note) to fix one scalefactor for all imaages 
          save_file(output, './input_val_q/sample_id_0{id_str}_{num}'.format(id_str=id_str,num=idx), f_out_fmt)
#          np.savetxt(path_for_save+filename+str(idx)+'.csv', np.asarray(list_val), delimiter=",", fmt='%1.3d')
          idx +=1  
          
          
          
    print("------id_01-----")
    series = pd.Series(id_01)
    summary = series.describe()
    show_dist(id_01)
    print(summary) 
    
    print("------id_02-----")
    series = pd.Series(id_02)
    summary = series.describe()
    show_dist(id_02)
    print(summary)        
           
    print("------id_03-----")
    series = pd.Series(id_03)
    summary = series.describe()
    show_dist(id_03)
    print(summary)        

    print("------id_04-----")
    series = pd.Series(id_04)
    summary = series.describe()
    show_dist(id_04)
    print(summary)        
                          
    
    
    
    return


if __name__=='__main__':
    main()

import csv 
import argparse
import numpy as np


def get_csv(f_in):
    list=[]
    with open(f_in, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True, delimiter=',')
        for row in reader:
            for r in row:
                if len(r) > 0:
                    list.append(float(r))
    
    arr = np.asarray(list)
    return arr


def save_file(list, f_out, format):
    f = open(f_out+'.'+format, 'w') 
    
    if format == 'csv':
        for val in list:
            f.write(str(val)+',\n')

    elif format == 'hex':
        size = len(list)//4
        res = len(list)%4
        print("size of list:", size, "res of list:", res)
        for n in range(size):
#            val = ''.join('%02x' %a for a in [list[4*n+3], list[4*n+2], list[4*n+1], list[4*n]]) # a is integer case, not 2's com hex
            val = ''.join('{:02X}'.format(a & 0xff) for a in [list[4*n+3], list[4*n+2], list[4*n+1], list[4*n]]) # a is integer case
#            print(val, type(val))
            f.write(val+'\n')
        if res > 0: #! truncation bug fix
            rev_list=[]
            for r in reversed(range(res)):
                rev_list.append(list[4*size+r])
            val = ''.join('{:02X}'.format(a & 0xff) for a in rev_list)
            f.write(val+'\n')

    f.close()

    return 0


def get_dense_weights(weights_arr,row,col):
    weights_arr=weights_arr.reshape(row,-1)
    return weights_arr


def get_batchnorm_weights(gamma=np.array([]), beta=np.array([]),moving_mean=np.array([]),moving_var=np.array([])):

    if  (beta.size == 0 ):
      
       beta=np.full((moving_mean.size),0)
        
    if (gamma.size == 0) :
      
       gamma=np.full((moving_mean.size),1)

    
    
    weights_arr = np.stack((gamma, beta, moving_mean, moving_var), axis=0)   
      
    return weights_arr



def get_dense_folded_weights(dense_weights,batch_norm_weights,epsilon=1e-3,dense_bias=np.array([])):
  
  gamma = batch_norm_weights[0]
  beta = batch_norm_weights[1]
  mean = batch_norm_weights[2]
  variance = batch_norm_weights[3]

  if dense_bias.size == 0 :
    dense_bias=np.full((batch_norm_weights[0].size),0)
  

  weights=[]
  bias=[]

  for i in range(len(dense_weights)):
    new_weights = ( dense_weights[i]*gamma ) /np.sqrt(variance+epsilon)
    weights.append(new_weights.tolist())
    
    
  weights=np.asarray(weights)
  weights=weights.reshape(-1)
  weights=weights.tolist()


  bias = beta + (dense_bias-mean)*gamma / np.sqrt(variance+epsilon)

  
  return weights, bias



# dense, BatchNormalized fold weights 
# batch_mean = batch_moving_mean
# batch_var  = batch_moving_variance


  
def main():
    
    
    '''BN dense 1'''
    #get batchnormalized1 weights
    #BN1_gamma=get_csv('Batchnorm1_gamma.csv')
    BN1_beta=get_csv('Batchnorm1_beta.csv')
    BN1_mean=get_csv('Batchnorm1_moving_mean.csv')
    BN1_var=get_csv('Batchnorm1_moving_variance.csv')
    
    batch1_weights=get_batchnorm_weights( beta=BN1_beta, moving_mean = BN1_mean, moving_var=BN1_var) #gamma=BN1_gamma, beta=BN1_beta
    
    #get dense1 weights
    
    dense1_weights=get_csv('dense1_params.csv')
    dense1_weights=get_dense_weights(dense1_weights,256,128)
    
    
    #get folding weight, bias 
    weights, bias = get_dense_folded_weights(dense1_weights,batch1_weights)

    #save folding weight, bias 
    save_file(weights,'BN1_weights','csv')
    save_file(bias,'BN1_bias','csv')
    
    
    
    '''BN dense 2    '''
    #get batchnormalized1 weights
    BN2_gamma=get_csv('Batchnorm2_gamma.csv')
    BN2_beta=get_csv('Batchnorm2_beta.csv')
    BN2_mean=get_csv('Batchnorm2_moving_mean.csv')
    BN2_var=get_csv('Batchnorm2_moving_variance.csv')
    
    batch2_weights=get_batchnorm_weights(gamma=BN2_gamma,beta=BN2_beta, moving_mean=BN2_mean, moving_var=BN2_var) #beta=BN2_beta,
    
    
    #get dense2 weights
    dense2_weights=get_csv('dense2_params.csv')
    dense2_weights=get_dense_weights(dense2_weights,128,10)
    
    
    
    #get folding weight, bias 
    weights, bias = get_dense_folded_weights(dense2_weights, batch2_weights)

    
    #save folding weight, bias 
    save_file(weights,'BN2_weights','csv')
    save_file(bias,'BN2_bias','csv')
    

    
    
if __name__=='__main__':
    main()
    
    
    
    
    

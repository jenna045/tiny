
import numpy as np
import pandas as pd 
import csv

import argparse

#path_for_save = './val_data/params'
#if not os.path.exists(path_for_save): os.mkdir(path_for_save)

#f_in = './dense1_params.csv'
#f_out = './dense1_parmas_q'
#f_out2 = './dense1_parmas_q.bin'

#TODO: check bin file contents
def save_bin(f_in, f_out):
    df = pd.read_csv(f_in, sep=',')
    df.describe()
    np.asarray(df.values).tofile(f_out)

    return 0

def get_range(arr):
    arr_max = arr.max()
    arr_min = arr.min()
    arr_range = abs(arr_max - arr_min)
    print("max:", arr_max, "min:", arr_min, "range:", arr_range, "mean:", arr.mean(), "std:", arr.std())
#    print(arr)
#TODO: histogram using matplot

    return arr_range 

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

def twos_comp(val, bits):
    for i in range(bits):
        mask_bits |= 0x1<<i 
    val = hex(((abs(val) ^ mask_bits) + 1) & mask_bits)
    return val
'''
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
'''
def save_file(list, f_out, format, is_bias=False):
    f = open(f_out + '.' + format, 'w')

    if format == 'csv':
        for val in list:
            f.write(str(val) + ',\n')
            
    
    elif format == 'hex':

        size = len(list) // 4
        res = len(list) % 4
        
        if is_bias: 
          for n in range(len(list)) : 
                  val = ''.join('{:08X}'.format(a & 0xFFFFFFFF) for a in [list[n]])
                  f.write(val + '\n')     

        
        else:    
          for n in range(size):
                val = ''.join('{:02X}'.format(a & 0xff) for a in [list[4 * n + 3], list[4 * n + 2], list[4 * n + 1], list[4 * n]])
                f.write(val + '\n')

          if res > 0:         
            rev_list = []
            for r in reversed(range(res)):
              rev_list.append(list[4 * size + r])
            val = ''.join('{:02X}'.format(a & 0xff) for a in rev_list)
            f.write(val + '\n')
  
    f.close()

    return 0


'''
 float32 to int8
  clip[ round((2**(bits-1)-1)*x), -2**(bits-1), +2**(bits-1)-1 ] 
--> clip[ round((2**(bits-1)-1)*x), -2**(bits-1)+1, +2**(bits-1)-1 ] 
  @note (20220330 by nina)
    -128 should be removed in the sign&magnitude scheme 
    ex) [-127, +127]
'''

def quantize(arr, bits, arr_range):

    print("arr type:", type(arr), "arr dim:", arr.shape, "arr dtype:", arr.dtype)

    lower_bnd = -1 * pow(2, bits-1) +1
    upper_bnd = pow(2, bits-1) -1 
    #levels = (pow(2, bits-1)-1)*2 #symmetric, uniform - assumed sign&magnitude scheme
    levels = upper_bnd - lower_bnd
    scale_factor = levels/arr_range
    print("scale_factor:", scale_factor)

    list=[]
    for i in arr:
#        print("f32_val:", i)
#        i = int(np.round(i*scale_factor))

        i = int(round(i*scale_factor))
#TODO: stochastic rounding 
        " clipping " 
        if i > upper_bnd:
            print(" apply upper bound:", i, "->", upper_bnd)
            i = upper_bnd
           
        if i < lower_bnd:
            print(" apply lower bound:", i, "->", lower_bnd)
            i = lower_bnd
   
#        print("--> i8_val:", i)
        list.append(i)

    return list


def bias_quantize(list, val):
    
    new_bias=[]
    for i in range(len(list)):
      bias=list[i]*val
      new_bias.append(bias)
      
    return new_bias
      
      


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('i', type=str, help="input file name in csv ex)aaa.csv")
    parser.add_argument('o', type=str, help="output file name without format")
    parser.add_argument('f', type=str, help="output file format to save")
    parser.add_argument('b', type=int, help="number of bits to quantize")
    parser.add_argument('w', type=int, help="select bias or weight to quantize")
    args = parser.parse_args()

    arr = get_csv(args.i)

#20230208.nina - clipping error debug 
#    arr_range = get_range(arr)
    arr_range = 160
    
    #quantized for weight
    save_file(quantize(arr, args.b, arr_range), args.o, args.f, args.w) 
    #quantized for bias 
    #save_file(bias_quantize(quantize(arr, args.b, arr_range),59), args.o, args.f, args.w) 


if __name__=='__main__':
    main()



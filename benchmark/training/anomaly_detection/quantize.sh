#!/bin/bash

path=$1
fmt=$2


python quantize.py $path/BN1_bias.csv $path/dense1_params_q_bias $fmt 8 1 > $path/dense1_params_q.info 
python quantize.py $path/BN2_bias.csv $path/dense2_params_q_bias $fmt 8 1 > $path/dense2_params_q.info 


python quantize.py $path/BN1_weights.csv $path/dense1_params_q $fmt 8 0 > $path/dense1_params_q.info 
python quantize.py $path/BN2_weights.csv $path/dense2_params_q $fmt 8 0 > $path/dense2_params_q.info 

#python quantize.py $path/dense2_params.csv $path/dense2_params_q $fmt 8 > $path/dense2_params_q.info

##! csv int8 format or hex dump 
#python utils/quantize.py ${in_path}/dense1_params.csv ${out_path}/dense1_params_q csv 8 > ${out_path}/dense1_params_q.info
#python utils/quantize.py ${in_path}/dense2_params.csv ${out_path}/dense2_params_q csv 8 > ${out_path}/dense2_params_q.info
#
##! hex dump 
##python utils/quantize.py ${in_path}/dense1_params.csv ${out_path}/dense1_params_q hex 8 > ${out_path}/dense1_params_q.info 
##python utils/quantize.py ${in_path}/dense2_params.csv ${out_path}/dense2_params_q hex 8 > ${out_path}/dense2_params_q.info 

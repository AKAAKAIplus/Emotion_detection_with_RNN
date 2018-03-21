# -*- coding: utf-8 -*-
import functools
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import sets
import scipy.spatial
from gensim.models import Word2Vec
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity
import pprint
pp = pprint.PrettyPrinter(indent=4)

import json
from scipy import stats
import pandas as pd
import os
from gensim.models import word2vec
import logging
import jieba
import json

# *************************************************************************************************************************
# 
# Setting Environment & Functions
# 
# *************************************************************************************************************************
itr_notChange = 10
threshold_loss = 0.000001

learning_rate = 0.01
training_iters = 5000

KEEP_PROB = 0.68


max_length = 25
n_input = 400
n_classes = 5

num_hidden = 250 
num_layers = 2

savePath_o = "model_original.ckpt"
# *****************************************************************************************************************
# 
# load Word Vector & Method
# 
# *****************************************************************************************************************
model = word2vec.Word2Vec.load("../word2vec/word2vector_400.model.bin")

with open("../word2vec/filterwords.txt") as f:
	filter_word_content = f.readlines()
filter_word_content = [x.split('\n')[0] for x in filter_word_content] 

def get_vector(word):
	try:
		v = model.wv[word].tolist()
	except:
		v = None
	return np.array(v, dtype = np.float).astype(np.float32)

def fix_input(input_):
	if len(input_) < max_length:
		for i in range(max_length-len(input_)):
			input_.append(  np.zeros((400,), dtype=np.float32)	)
	return input_

# return the length of data
def _length(data):
	used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
	length = tf.reduce_sum(used, reduction_indices=1)
	length = tf.cast(length, tf.int32)
	return length

def weight_variable(shape):
	initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
	return tf.get_variable("weights", shape,initializer=initializer, dtype=tf.float32)

def bias_variable(shape):
	initializer = tf.constant_initializer(0.0)
	return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)

def saveModel_o(sess,path):
	save_path = saver.save(sess,path)



# *****************************************************************************************************************
# 
# Setting Tensorflow Structure For RNN
# 
# *****************************************************************************************************************
original_graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
sess_original = tf.Session(graph = original_graph,config=tf.ConfigProto(gpu_options=gpu_options))
sess_original = tf.Session(graph = original_graph)

with original_graph.as_default():
	# ***************
	# Set placeholder
	# ***************
		data_o = tf.placeholder(tf.float32, [None, max_length, n_input])
		target_o = tf.placeholder(tf.float32, [None, n_classes])
	# ******************
	# Recurrent network.
	# ******************
		output_o, last_o = rnn.dynamic_rnn(
			rnn_cell.GRUCell(num_hidden),
			data_o,
			dtype=tf.float32,
			sequence_length=_length(data_o),
			time_major=False
		)

		weight_o = weight_variable([num_hidden,n_classes])
		bias_o = bias_variable([n_classes])
		prediction_o = tf.matmul(last_o, weight_o) + bias_o

	# *****************************************
	# Setting Loss and Optimizer & Initializing
	# *****************************************
		loss_o = tf.reduce_mean(  tf.nn.softmax_cross_entropy_with_logits(labels=target_o, logits=prediction_o)   )
		optimizer_o = tf.train.AdamOptimizer(learning_rate).minimize(loss_o)

		correct_prediction = tf.equal(tf.argmax(prediction_o,1), tf.argmax(target_o,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		init_o = tf.global_variables_initializer()
		saver_o = tf.train.Saver() 	

tf.reset_default_graph()

# ******************************************************************************************************************************
# 
# Loading & Training
# 
# ******************************************************************************************************************************
sess_original.run(init_o)

print("\n\n******************************************************")
print('Loading Data and Vector...')

# ****************************
# Loading DataSet and Training
# ****************************
batch_size = 5000

count = 0
train_data = []
train_label = []

test_acc = []

for file in os.listdir("../data_pre_processing"):
	if str(file) != '.DS_Store':
		df = pd.read_csv(open( "../data_pre_processing/"+str(file),'rU' ) )
		df=df.values

		print('--------------------------------------change file')
		
		for i in df:
			if isinstance(i[0],str) and isinstance(i[1],str) and len(i[0])>0 and len(i[0]) < max_length:
				if i[4] == 0 and i[5] == 0 and i[6] == 0 and i[7] == 0 and i[8] == 0:
					continue

				i[0] = jieba.cut( i[0], cut_all=False)
				i[0] = "".join([word for word in list(i[0]) if word.encode('utf-8') not in filter_word_content])

				a = []
				for word in list(i[0]):
					v = get_vector(word)
					if np.isnan(v).any():
						# print('.')
						continue
					a.append(v)

				a = fix_input(a)
				train_data.append(a)

				train_label.append(np.array([i[4],i[5],i[6],i[7],i[8]], dtype = np.float).astype(np.float32))

				count += 1

				# training
				if count <= 3640000:
					if count % batch_size == 0:
						try:
							for _ in range(10):
								sess_original.run(optimizer_o,{data_o: train_data, target_o: train_label})
							acc = sess_original.run(accuracy, {data_o: train_data, target_o: train_label})
							print ("Iter " + str(count) + ", Training Accuracy= " + "{:.5f}".format(acc))
							train_data = []
							train_label = []
							
						except Exception as e:
							print(e)
							break
				# testing
				elif count <= 5195000:
					if count % batch_size == 0:
						try:
							acc = sess_original.run(accuracy, {data_o: train_data, target_o: train_label})
							print ("\n\nIter " + str(count) + ", Testing Accuracy= " + "{:.5f}".format(acc))
							train_data = []
							train_label = []
							test_acc.append(acc)
						except Exception as e:
								print(e)
								break
				else:
					break
print(count)
print(sum(test_acc)/count)

saveModel_o(sess_original,savePath_o)
sess_original.close()



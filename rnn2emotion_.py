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
import os, fnmatch
import random
from gensim.models import word2vec
import logging
import jieba
import json
import random
# *************************************************************************************************************************
# 
# Setting Environment & Functions
# 
# *************************************************************************************************************************

max_length = 20
word2vec_size = sys.argv[1]

# *****************************************************************************************************************
# 
# load Word Vector & Method
# 
# *****************************************************************************************************************
model = word2vec.Word2Vec.load("../word2vec/word2vector_"+str(word2vec_size)+".model.bin")

with open("../word2vec/filtersingleword.txt") as f:
	filter_word_content = f.readlines()
filter_single_word_content = [x.split('\n')[0] for x in filter_word_content] 

def get_vector(word):
	try:
		v = model.wv[word].tolist()
	except:
		v = None
	return np.array(v, dtype = np.float).astype(np.float32)

def fix_input(input_):
	if len(input_) < max_length:
		for i in range(max_length-len(input_)):
			input_.append(  np.zeros((int(word2vec_size),), dtype=np.float32)	)
	return input_

# return the length of data
def _length(data):
	used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
	length = tf.reduce_sum(used, reduction_indices=1)
	length = tf.cast(length, tf.int32)
	return length

def _last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant

def weight_variable(shape,n):
	initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
	return tf.get_variable(n, shape,initializer=initializer, dtype=tf.float32)

def bias_variable(shape,n):
	initializer = tf.constant_initializer(0.0)
	return tf.get_variable(n, shape, initializer=initializer, dtype=tf.float32)


# *****************************************************************************************************************
# 
# Setting Tensorflow Structure For RNN
# 
# *****************************************************************************************************************
learning_rate = 0.001
KEEP_PROB = 0.83

num_hidden = 512

n_input = int(word2vec_size)

n_classes = 5

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

		last_relevant_o = _last_relevant(output_o,_length(data_o))

		drop_output = tf.nn.dropout( last_relevant_o, KEEP_PROB )

		weight_o = weight_variable([num_hidden,n_classes],'w2')
		bias_o = bias_variable([n_classes],'b2')
		prediction_o = tf.matmul(drop_output, weight_o) + bias_o

	# *****************************************
	# Setting Loss and Optimizer & Initializing
	# *****************************************
		loss_o = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_o, logits=prediction_o))
		optimizer_o = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_o)

		max_index = tf.argmax(prediction_o,1)

		correct_prediction = tf.equal(tf.argmax(prediction_o,1), tf.argmax(target_o,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		init_o = tf.global_variables_initializer()
		saver_o = tf.train.Saver(tf.global_variables()) 	

tf.reset_default_graph()


# ******************************************************************************************************************************
# 
# Loading
# 
# ******************************************************************************************************************************
savePath_o = "./rnn_saved_model/rn2emotion_model_original.ckpt"

print("\n\n******************************************************")
print('Checking Model')
print("******************************************************")

train_or_not = True
if len(os.listdir("./rnn_saved_model/")) >= 4:    
    saver_o.restore(sess_original, tf.train.latest_checkpoint('./rnn_saved_model/'))
    train_or_not = False
else:
    sess_original.run(init_o)

# ****************************
# Loading DataSet and Training
# ****************************
print("\n\n******************************************************")
print('Loading Data and Vector to Train')


load_file = "../data_pre_processing_politics"
file_list = random.sample(os.listdir(load_file),len(os.listdir(load_file)))

all_raw_training_data = []
all_raw_testing_data = []
c = 0
for index , file in enumerate(file_list):
	if str(file) != '.DS_Store' and c <= 6:
		df = pd.read_csv(open( load_file+"/"+str(file),'rU' ) )
		all_raw_training_data.extend(random.sample(df.values, len(df.values)))

		c += 1


load_file = "../data_pre_processing_politics_emotion"
file_list = random.sample(os.listdir(load_file),len(os.listdir(load_file)))
c = 0
for index , file in enumerate(file_list):
	if str(file) != '.DS_Store' and str(file) == str(sys.argv[2])+'.csv':
		df = pd.read_csv(open( load_file+"/"+str(file),'rU' ) )
		all_raw_testing_data.extend(random.sample(df.values, len(df.values)))

print(len(all_raw_training_data))
print(len(all_raw_testing_data))
# ******************************************************************************************************************************
# 
# Training
# 
# ******************************************************************************************************************************
count = 0
train_data = []
train_label = []

pre_loss = 100
break_time = 0
limit_break_time = 40

epoch = 9
batch_size = 15000

# training
try:
	if not train_or_not:
		print('------no training------')
		raise StopIteration

	for __ in range(epoch):
		for i in random.sample(all_raw_training_data,len(all_raw_training_data)):
			comment = str(i[0]).decode('utf-8')

			label = [i[1],i[2],i[3],i[4],i[5]]

			if len(comment) < max_length and (1 in label):
				comment = "".join([word for word in list(comment) if word.encode('utf-8') not in filter_single_word_content])

				temp_array = []
				for word in list(comment):
					v = get_vector(word)
					if np.isnan(v).any():
						list(comment).remove(word)
						continue
					temp_array.append(v)

				if len(comment) == 0:
					continue
				# print(comment)
				train_data.append( fix_input(temp_array) )
				train_label.append(np.array(label, dtype = np.float).astype(np.float32))
					
				count += 1

				if (count % batch_size == 0):
					try:
						for _ in range(11):
							sess_original.run(optimizer_o,{data_o: train_data, target_o: train_label})
						acc = sess_original.run(accuracy, {data_o: train_data, target_o: train_label})
						loss = sess_original.run(loss_o, {data_o: train_data, target_o: train_label})

						print ("Iter " + str(count) + ", Training Accuracy= " + str(acc)+ ", Loss= " + str(loss))
							
						if loss > pre_loss:
							break_time += 1
							print('---'+str(break_time))

						train_data = []
						train_label = []
						pre_loss = loss		
					except Exception as e:
						print(e)
						break

			if break_time >= limit_break_time:
				raise StopIteration

except StopIteration:
	if not train_or_not:
		pass
	else:
		save_path = saver_o.save(sess_original, savePath_o)




count = 0
train_data = []
train_label = []

max_0 = 0
max_1 = 0
max_2 = 0
max_3 = 0
max_4 = 0
max_all = 0

test_acc = []
# testing
print('-----------------------testing---------------------------')
for i in random.sample(all_raw_testing_data,len(all_raw_testing_data)):
	comment = str(i[0]).decode('utf-8')
	label = [i[1],i[2],i[3],i[4],i[5]]

	if len(comment) < max_length and (1 in label):
		comment = "".join([word for word in list(comment) if word.encode('utf-8') not in filter_single_word_content])

		temp_array = []
		for word in list(comment):
			v = get_vector(word)
			if np.isnan(v).any():
				list(comment).remove(word)
				# print('.')
				continue
			temp_array.append(v)

		if len(comment) == 0:
			continue
		# print(comment)
			
		train_data.append( fix_input(temp_array) )
		train_label.append(np.array(label, dtype = np.float).astype(np.float32))
			
		count += 1
		if count % batch_size == 0:
			try:
				acc = sess_original.run(accuracy, {data_o: train_data, target_o: train_label})
				max_index_ = sess_original.run(max_index, {data_o: train_data, target_o: train_label})
				print ("\nIter " + str(count) + ", Testing Accuracy= " + "{:.5f}".format(acc))

				train_data = []
				train_label = []
				test_acc.append(acc)

				
				for m_i in max_index_:
					if m_i.item() == 0:
						max_0 += 1
					elif m_i.item() == 1:
						max_1 += 1
					elif m_i.item() == 2:
						max_2 += 1
					elif m_i.item() == 3:
						max_3 += 1
					elif m_i.item() == 4:
						max_4 += 1
					max_all += 1

			except Exception as e:
				print(e)


print('Accuracy:')
print(sum(test_acc)/(len(test_acc)))
print('------')
print(float(max_0)/float(max_all))
print(float(max_1)/float(max_all))
print(float(max_2)/float(max_all))
print(float(max_3)/float(max_all))
print(float(max_4)/float(max_all))


sess_original.close()

# 2020.11.16
import math
import numpy as np

def sigmoid(x):
	return 1.0 / (1.0 + math.e ** -x)

def softplus(x):
	return math.log(1.0 + math.exp(x))

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

def y_list_add(lst, key, add_value):
	if key in lst:
		lst[key] += add_value
	else:
		lst[key] = add_value
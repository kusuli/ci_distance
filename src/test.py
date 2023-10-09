# 2023.10.08

from tps import TPS
from gtps import GTPS
from gtps2 import GTPS2
import setting
import train
import util
import os
import math
import re
import statistics as st
import torch
import copy
from multiprocessing import Pool


if __name__ == '__main__':

	def exp_i(title, index, gtps2, tensor_param=None):
		data_dir = './rntxt/20230311/mod21_phrase'
		output_file = 'test_01.txt'
		if tensor_param is None:
			tensor_param = torch.tensor([0.0 for _ in range(gtps2.get_param_count_all() + gtps2.get_pi_param_count_all())], requires_grad=True)
		op = torch.optim.SGD([tensor_param], lr=0.001, momentum=0.0)
		torch.set_printoptions(edgeitems=tensor_param.size()[0])
		ret_str = train.train(data_dir, 200, 100, gtps2, tensor_param, op, max_length=50, wait_epoch=2, parallel_count=12)
		with open(output_file, mode='a') as f:
			f.write('\n\n' + title + ': '+ str(index) + ' de_tpl_list: ' + str(gtps2.de_tpl_list) + ' pi_index: ' + str(gtps2.pi_index) + ' params: ' + str(gtps2.get_param_count_all()) + ' redundancy: ' + str(len(gtps2.de_tpl_list)) + ' ' + ret_str + '\n')

	if True:
		b_fixed_list = [0, 0, 0]
		#b_fixed_list = [0, 0, 1]
		g_tpl_list_x2 = [] 
		for i1 in range(2): # s_scale (0 if not used)
			for i2 in range(i1 + 2): # d_degree
				for i3 in range(max(i1, i2) + 2): # s_degree
					for i4 in range(max(i1, i2, i3) + 2): # d_degree
						for i5 in range(max(i1, i2, i3, i4) + 2): # tonic (region)
							g_tpl_list_x2.append([i1, i2, i3, i4, i5])
		#print(g_tpl_list_x2)
		#print(len(g_tpl_list_x2))
		de_tpl_list_x2 = []
		for g_tpl_list in g_tpl_list_x2:
			de_tpl_list = []
			for i in range(1, 6): # at most 5 terms
				if i <= max(g_tpl_list):
					de_tpl = [0, 0, 0]
					# scale
					if g_tpl_list[0] == i and g_tpl_list[1] == i:
						de_tpl[0] = 3
					elif g_tpl_list[0] == i:
						de_tpl[0] = 1
					elif g_tpl_list[1] == i:
						de_tpl[0] = 2
					# degree
					if g_tpl_list[2] == i and g_tpl_list[3] == i:
						de_tpl[1] = 3
					elif g_tpl_list[2] == i:
						de_tpl[1] = 1
					elif g_tpl_list[3] == i:
						de_tpl[1] = 2
					# tonic(region)
					if g_tpl_list[4] == i:
						de_tpl[2] = 1
					de_tpl_list.append(de_tpl)
			de_tpl_list_x2.append(de_tpl_list)
		#print(de_tpl_list_x2)
		#print(len(de_tpl_list_x2))
		de_tpl_list_x2_ = de_tpl_list_x2
		de_tpl_list_x2 = []
		for de_tpl_list in de_tpl_list_x2_: # expand about scale
			de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
			for i in range(len(de_tpl_list)):
				if de_tpl_list[i][0] == 3:
					de_tpl_list[i][0] = 4
					de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
					de_tpl_list[i][0] = 5
					de_tpl_list_x2.append(de_tpl_list)
					break
		de_tpl_list_x2_ = de_tpl_list_x2
		de_tpl_list_x2 = []
		for de_tpl_list in de_tpl_list_x2_: # expand about degree
			de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
			for i in range(len(de_tpl_list)):
				if de_tpl_list[i][1] == 3:
					de_tpl_list[i][1] = 4
					de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
					de_tpl_list[i][1] = 5
					de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
					de_tpl_list[i][1] = 6
					de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
					de_tpl_list[i][1] = 7
					de_tpl_list_x2.append(de_tpl_list)
					break
		de_tpl_list_x2_ = de_tpl_list_x2
		de_tpl_list_x2 = []
		for de_tpl_list in de_tpl_list_x2_: # expand about tonic (region)
			de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
			for i in range(len(de_tpl_list)):
				if de_tpl_list[i][2] == 1:
					de_tpl_list[i][2] = 2
					de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
					de_tpl_list[i][2] = 3
					de_tpl_list_x2.append(copy.deepcopy(de_tpl_list))
					de_tpl_list[i][2] = 4
					de_tpl_list_x2.append(de_tpl_list)
					break
		#print([str(e) + "<br>" for e in de_tpl_list_x2])
		#print(len(de_tpl_list_x2))
		for i, de_tpl_list in enumerate(de_tpl_list_x2):
			if i > 0:
				gtps2 = GTPS2(de_tpl_list, _b_fixed_list=b_fixed_list)
				exp_i('○test', i, gtps2)

# coding: utf-8
# 2023.01.19

import heapq
import math
import util
import setting
from tps import TPS
import numpy
import torch

class GTPS2:
	SCALE_TYPE_COUNT = 5
	DEGREE_TYPE_COUNT = 7
	REGION_TYPE_COUNT = 4
	FIXED_COST_FUNCITON_COUNT = 3
	PC_INT_TYPE_COUNT = 5
	
	def __init__(self, _de_tpl_list, _b_fixed_list = [], _pi_index = 0):
		self.tps = TPS()
		self.de_tpl_list = _de_tpl_list # de_tpl は (scale要素インデックス, degree要素インデックス, region要素インデックス) なので (0～SCALE_TYPE_COUNT, 0～DEGREE_TYPE_COUNT, 0～REGION_TYPE_COUNT) という形
		self.b_fixed_list = [1 if i < len(_b_fixed_list) and _b_fixed_list[i] == 1 else 0 for i in range(self.FIXED_COST_FUNCITON_COUNT)] # fixed cost functionの有効フラグリスト
		self.matrix_list = []
		self.matrix_param_count = 0
		for de_tpl in self.de_tpl_list:
			param_count = self.get_param_count(de_tpl)
			self.matrix_list.append(numpy.array([1.0 for i in range(param_count)]))
			self.matrix_param_count += param_count
		self.edge_store = {}
		self.edge_tensor_store = {}
		
		self.pi_index = _pi_index
		self.pi_de_list = []
		self.pi_de_list.append(numpy.array([1.0 for i in range(2)]))           # chord notes / others
		self.pi_de_list.append(numpy.array([1.0 for i in range(3)]))           # chord notes / scale notes / others
		self.pi_de_list.append(numpy.array([1.0 for i in range(5)]))           # root / third / fifth / scale notes / others
		self.pi_de_list.append(numpy.array([1.0 for i in range(2 * 5)]))       # scale * (root / third / fifth / scale notes / others)
		self.pi_store = {} 
		self.pi_tensor_store = {} 
	
	def get_distance2(self, keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2):
		dist_list = []
		if self.b_fixed_list[0] or self.b_fixed_list[1] or self.b_fixed_list[2]:
			temp_sum = self.tps.get_distance2(keypos_1, scale_1, degree_1, 0, keypos_2, scale_2, degree_2, 0)
		if self.b_fixed_list[0]: # TPS.region
			dist_list.append(temp_sum[0])
		if self.b_fixed_list[1]: # TPS.chord
			dist_list.append(temp_sum[1])
		if self.b_fixed_list[2]: # TPS.basicspace
			dist_list.append(temp_sum[2])
		for de_tpl in self.de_tpl_list:
			dist_list.extend(self.get_matrix_coefs(de_tpl, keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2))
		return dist_list
	
	def get_scalar_distance(self, dist_list):
		if not tuple(dist_list) in self.edge_store.keys():
			current_index = 0
			dist = 0.0
			if dist_list != TPS.ZERO_TPL:
				if self.b_fixed_list[0]: # TPS.region
					#dist += dist_list[current_index] * self.tps.coef_i
					dist += dist_list[current_index]
					current_index += 1
				if self.b_fixed_list[1]: # TPS.chord
					#dist += dist_list[current_index] * self.tps.coef_j
					dist += dist_list[current_index]
					current_index += 1
				if self.b_fixed_list[2]: # TPS.basicspace
					#dist += dist_list[current_index] * (self.tps.coef_sum - self.tps.coef_i - self.tps.coef_j)
					dist += dist_list[current_index]
					current_index += 1
				for i in range(len(self.de_tpl_list)):
					# 全部one-hot系
					dist += self.matrix_list[i][dist_list[current_index]]
					current_index += 1
			self.edge_store[tuple(dist_list)] = dist
		return self.edge_store[tuple(dist_list)]
	
	def get_scalar_distance_tensor(self, dist_list, tensor_param):
		if not tuple(dist_list) in self.edge_tensor_store.keys():
			current_dist_index = 0
			current_param_index = 0
			dist_tensor = torch.tensor(0.0)
			if dist_list != TPS.ZERO_TPL:
				if self.b_fixed_list[0]: # TPS.region
					dist_tensor += dist_list[current_dist_index]
					current_dist_index += 1
				if self.b_fixed_list[1]: # TPS.chord
					dist_tensor += dist_list[current_dist_index]
					current_dist_index += 1
				if self.b_fixed_list[2]: # TPS.basicspace
					dist_tensor += dist_list[current_dist_index]
					current_dist_index += 1
				for i in range(len(self.de_tpl_list)):
					# 全部one-hot系
					dist_tensor += tensor_param[current_param_index + dist_list[current_dist_index]]
					current_dist_index += 1
					current_param_index += len(self.matrix_list[i])
			self.edge_tensor_store[tuple(dist_list)] = dist_tensor
		return self.edge_tensor_store[tuple(dist_list)]
	
	def get_param_count_all(self):
		param_count = 0
		for de_tpl in self.de_tpl_list:
			param_count = param_count + self.get_param_count(de_tpl)
		return param_count
	
	def get_param_count(self, de_tpl):
		param_count = 0
		# scale要素
		temp = self.get_param_count_scale(de_tpl[0])
		param_count = temp
		# degree要素
		temp = self.get_param_count_degree(de_tpl[1])
		if param_count == 0:
			param_count = temp
		elif temp > 0:
			param_count = param_count * temp
		# region要素
		temp = self.get_param_count_region(de_tpl[2])
		if param_count == 0:
			param_count = temp
		elif temp > 0:
			param_count = param_count * temp
		return param_count
	
	def get_param_count_scale(self, index):
		param_count = 0
		if index == 0: #なし
			param_count = 0
		elif index == 1: # srcだけ
			param_count = 2
		elif index == 2: # destだけ
			param_count = 2
		elif index == 3: # src×dest（非対称）
			param_count = 2 * 2
		elif index == 4: # src×dest（対称、対角成分を統一しない）
			param_count = 3
		elif index == 5: # src×dest（対称、対角成分を統一）
			param_count = 2
		else:
			print('get_param_count_scale : パラメータが不正です', index)
			exit()
		return param_count
	
	def get_param_count_degree(self, index):
		param_count = 0
		if index == 0: #なし
			param_count = 0
		elif index == 1: # srcだけ
			param_count = 7
		elif index == 2: # destだけ
			param_count = 7
		#elif index == 3: # dest root on src tonic（dr12）だけ
		#	param_count = 12
		#elif index == 4: # dest root on src tonic（全部無理やりdegree化A）（dr7a）だけ
		#	param_count = 7
		#elif index == 5: # dest root on src tonic（全部無理やりdegree化B）（dr7b）だけ
		#	param_count = 7
		elif index == 3: # src×dest 組み合わせ（非対称）
			param_count = 7 * 7
		elif index == 4: # src×dest 組み合わせ（対称、対角成分を統一しない）
			param_count = 28
		elif index == 5: # src×dest 組み合わせ（対称、対角成分を統一）
			param_count = 22
		#elif index == 9: # src×dr12 組み合わせ（非対称）
		#	param_count = 7 * 12
		#elif index == 10: # src×dr7a 組み合わせ（非対称）
		#	param_count = 7 * 7
		#elif index == 11: # src×dr7b 組み合わせ（非対称）
		#	param_count = 7 * 7
		elif index == 6: # 移動量（非対称）
			param_count = 7
		elif index == 7: # 移動量（対称）
			param_count = 4
		else:
			print('get_param_count_degree : パラメータが不正です', index)
			exit()
		return param_count
	
	def get_param_count_region(self, index):
		param_count = 0
		if index == 0: #なし
			param_count = 0
		elif index == 1: # 移動量（tonic、非対称）
			param_count = 12
		elif index == 2: # 移動量（tonic、対称）
			param_count = 7
		elif index == 3: # 移動量（major tonic、非対称）
			param_count = 12
		elif index == 4: # 移動量（major tonic、対称）
			param_count = 7
		else:
			print('get_param_count_region : パラメータが不正です', index)
			exit()
		return param_count
	
	def get_matrix_coefs(self, de_tpl, keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2):
		ret = 0
		temp = self.get_matrix_coefs_scale(de_tpl[0], keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2)
		param_count = self.get_param_count_scale(de_tpl[0])
		ret = temp
		temp2 = self.get_matrix_coefs_degree(de_tpl[1], keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2)
		param_count2 = self.get_param_count_degree(de_tpl[1])
		if temp > 0 and param_count2 > 0:
			temp = temp * param_count2
		temp = temp + temp2
		temp3 = self.get_matrix_coefs_region(de_tpl[2], keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2)
		param_count3 = self.get_param_count_region(de_tpl[2])
		if temp > 0 and param_count3 > 0:
			temp = temp * param_count3
		temp = temp + temp3
		return [temp]
	
	def get_matrix_coefs_scale(self, index, keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2):
		b_major_1 = self.tps.scale_to_b_major(scale_1)
		b_major_2 = self.tps.scale_to_b_major(scale_2)
		if index == 1: # srcだけ
			return b_major_1
		elif index == 2: # destだけ
			return b_major_2
		elif index == 3: # src×dest（非対称）
			return b_major_1 * 2 + b_major_2
		elif index == 4: # src×dest（対称、対角成分を統一しない）
			return b_major_1 + b_major_2
		elif index == 5: # src×dest（対称、対角成分を統一）
			return (b_major_1 - b_major_2) % 2
		else:
			return 0 # 0番目ということではない
	
	#get_matrix_coefs_degree_list1 = [0, 7, 1, 2, 2, 3, 8, 4, 5, 5, 6, 6] # 単二度と増四度を別枠にして無理やりdegree化
	get_matrix_coefs_degree_list2 = [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6] # 全部無理やりdegree化
	def get_matrix_coefs_degree(self, index, keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2):
		val_1 = degree_1 - 1
		val_2 = degree_2 - 1
		val_min = min(val_1, val_2)
		val_max = max(val_1, val_2)
		dr12 = (keypos_2 - keypos_1 + setting.SCALE_DISTANCE_DIC[scale_2][val_2]) % 12 # 遷移元keyの1度の音に対して
		dr7a = self.get_matrix_coefs_degree_list2[dr12]
		dr7b = min(dr12, 12 - dr12)
		if index == 1: # srcだけ
			return val_1
		elif index == 2: # destだけ
			return val_2
		#elif index == 3: # dest root on src tonic（dr12）
		#	return dr12
		#elif index == 4: # dest root on src tonic（全部無理やりdegree化）（dr7a）だけ
		#	return dr7a
		#elif index == 5: # dest root on src tonic（全部無理やりdegree化）（dr7b）だけ
		#	return dr7b
		elif index == 3: # src×dest 組み合わせ（非対称）
			return val_1 * 7 + val_2
		elif index == 4: # src×dest 組み合わせ（対称、対角成分を統一しない）
			ret = 0
			for i1 in range(7):
				for i2 in range(7 - i1):
					if val_min == i1 and val_max == i1 + i2:
						return ret
					ret = ret + 1
			print('get_matrix_coefs_degree : エラー2', degree_1, degree_2)
			exit()
		elif index == 5: # src×dest 組み合わせ（対称、対角成分を統一）
			if val_1 == val_2:
				return 0
			else:
				ret = 1 # イコールの場合を除くため 1 スタート
				for i1 in range(7): # i1 == 6 までは来ないが
					for i2 in range(6 - i1):
						if val_min == i1 and val_max == i1 + i2 + 1:
							return ret
						ret = ret + 1
				print('get_matrix_coefs_degree : エラー1', degree_1, degree_2)
				exit()
		#elif index == 9: # src×dr12 組み合わせ（非対称）
		#	return val_1 * 12 + dr12
		#elif index == 10: # src×dr7a 組み合わせ（非対称）
		#	return val_1 * 7 + dr7a
		#elif index == 11: # src×dr7b 組み合わせ（非対称）
		#	return val_1 * 7 + dr7b
		elif index == 6: # 移動量（非対称）
			return (val_2 - val_1) % 7
		elif index == 7: # 移動量（対称）
			return min((val_2 - val_1) % 7, (val_1 - val_2) % 7)
		else:
			return 0 # 0番目ということではない
	
	def get_matrix_coefs_region(self, index, keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2):
		major_keypos_1 = keypos_1
		if scale_1 != setting.SCALE_MAJOR:
			major_keypos_1 += 3
		major_keypos_2 = keypos_2
		if scale_2 != setting.SCALE_MAJOR:
			major_keypos_2 += 3
		if index == 1: # 移動量（tonic、非対称）
			return (keypos_2 - keypos_1) % 12
		elif index == 2: # 移動量（tonic、対称）
			return min((keypos_1 - keypos_2) % 12, (keypos_2 - keypos_1) % 12)
		elif index == 3: # 移動量（major  tonic、非対称）
			return (major_keypos_2 - major_keypos_1) % 12
		elif index == 4: # 移動量（major tonic、対称）
			return min((major_keypos_1 - major_keypos_2) % 12, (major_keypos_2 - major_keypos_1) % 12)
		else:
			return 0 # 0番目ということではない
	
	def get_pi_param_count_all(self):
		param_count = 0
		for i in range(self.PC_INT_TYPE_COUNT - 1): # 0は無効、1はbasicspace（fixed）
			if self.pi_index == i + 2:
				param_count += len(self.pi_de_list[i])
		return param_count
	
	def get_pi_distance(self, keypos, scale, degree):
		pc_dist_list_x2 = [[] for i in range(12)]
		pc_attr_list = self.convert_tpl_to_pc_attr_list((keypos, scale, degree))
		if self.pi_index == 1: # basicspace（-13ずれる）
			for pc in range(12):
				if pc_attr_list[pc] == 4: # root
					pc_dist_list_x2[pc].append(-4)
				elif pc_attr_list[pc] == 3: # third
					pc_dist_list_x2[pc].append(-2)
				elif pc_attr_list[pc] == 2: # fifth
					pc_dist_list_x2[pc].append(-3)
				elif pc_attr_list[pc] == 1: # scale
					pc_dist_list_x2[pc].append(-1)
				else:
					pc_dist_list_x2[pc].append(0)
		if self.pi_index == 2: # chord notes / others
			for pc in range(12):
				if pc_attr_list[pc] in (4, 3, 2): # root, third, fifth
					pc_dist_list_x2[pc].append(1)
				else:
					pc_dist_list_x2[pc].append(0)
		if self.pi_index == 3: # chord notes / scale notes / others
			for pc in range(12):
				if pc_attr_list[pc] in (4, 3, 2): # root, third, fifth
					pc_dist_list_x2[pc].append(2)
				elif pc_attr_list[pc] == 1: # scale
					pc_dist_list_x2[pc].append(1)
				else:
					pc_dist_list_x2[pc].append(0)
		if self.pi_index == 4: # root / third / fifth / scale notes / others
			for pc in range(12):
				pc_dist_list_x2[pc].append(pc_attr_list[pc])
		if self.pi_index == 5: # scale * (root / third / fifth / scale notes / others)
			for pc in range(12):
				pc_dist_list_x2[pc].append(5 * scale + pc_attr_list[pc])
		ret = []
		for pc in range(12):
			ret.append(tuple(pc_dist_list_x2[pc]))
		#print(keypos, scale, degree, ret)
		return ret
	
	def get_pi_scalar_distance(self, chroma_list, pc_dist_list_x2):
		if not tuple(pc_dist_list_x2) in self.pi_store.keys():
			chroma_pc_dist_list = [] 
			for pc in range(12):
				current_index = 0
				chroma_pc_dist_list.append(0.0)
				if self.pi_index == 1: # basicspace（-13ずれる）
					chroma_pc_dist_list[pc] += pc_dist_list_x2[pc][current_index]
					current_index += 1
				for i in range(self.PC_INT_TYPE_COUNT - 1):
					if self.pi_index == i + 2:
						chroma_pc_dist_list[pc]  += self.pi_de_list[i][pc_dist_list_x2[pc][current_index]] # one-hot系
						current_index += 1
			self.pi_store[tuple(pc_dist_list_x2)] = chroma_pc_dist_list
		else:
			chroma_pc_dist_list = self.pi_store[tuple(pc_dist_list_x2)]
		ret = sum([chroma_list[i] * chroma_pc_dist_list[i] for i in range(12)])
		return ret
	
	def get_pi_scalar_distance_tensor(self, chroma_list, pc_dist_list_x2, tensor_param):
		if not tuple(pc_dist_list_x2) in self.pi_tensor_store.keys():
			chroma_dist_tensor_list = torch.tensor([0.0 for i in range(12)])
			for pc in range(12):
				current_dist_index = 0
				current_param_index = self.matrix_param_count
				if self.pi_index == 1: # PI_DE1, basicspace（-13ずれる）
					chroma_dist_tensor_list[pc] = chroma_dist_tensor_list[pc] + pc_dist_list_x2[pc][current_dist_index]
					current_dist_index += 1
				for i in range(self.PC_INT_TYPE_COUNT - 1):
					if self.pi_index == i + 2:
						chroma_dist_tensor_list[pc] = chroma_dist_tensor_list[pc] + tensor_param[current_param_index + pc_dist_list_x2[pc][current_dist_index]] # one-hot系
						current_dist_index += 1
						current_param_index += len(self.pi_de_list[i])
			self.pi_tensor_store[tuple(pc_dist_list_x2)] = chroma_dist_tensor_list
		else:
			chroma_dist_tensor_list = self.pi_tensor_store[tuple(pc_dist_list_x2)]
		ret = torch.sum(chroma_dist_tensor_list * torch.tensor(chroma_list))
		return ret

	def update_params(self, tensor_param):
		current_index = 0
		for i in range(len(self.de_tpl_list)):
			for i2 in range(len(self.matrix_list[i])):
				self.matrix_list[i][i2] = tensor_param[current_index].item()
				current_index += 1
		for i in range(self.PC_INT_TYPE_COUNT - 1):
			if self.pi_index == i + 2:
				for i2 in range(len(self.pi_de_list[i])):
					self.pi_de_list[i][i2] = tensor_param[current_index].item()
					current_index += 1
		#print(self.pi_de_list)
		self.edge_store = {}
		self.edge_tensor_store = {}
		self.pi_store = {}
		self.pi_tensor_store = {}

	@classmethod
	def convert_tpl_to_chroma_list(cls, tpl):
		(key, scale, degree) = tpl
		temp = setting.SCALE_DISTANCE_DIC[scale]
		root = (key + temp[degree - 1]) % 12
		third = (key + temp[(degree - 1 + 2) % len(temp)]) % 12
		fifth = (key + temp[(degree - 1 + 4) % len(temp)]) % 12
		#return [root, third, fifth]
		ret = [0.0 for i in range(12)]
		ret[root] = 1.0
		ret[third] = 1.0
		ret[fifth] = 1.0
		return ret
	
	@classmethod
	def convert_tpl_to_pc_attr_list(cls, tpl):
		(key, scale, degree) = tpl
		pc_attr_list = [0] * 12
		scale_list = setting.SCALE_DISTANCE_DIC[scale]
		for i in scale_list:
			pc_attr_list[(key + i) % 12] = 1
		pc_attr_list[(key + scale_list[degree - 1]) % 12] = 4
		pc_attr_list[(key + scale_list[(degree - 1 + 2) % len(scale_list)]) % 12] = 3
		pc_attr_list[(key + scale_list[(degree - 1 + 4) % len(scale_list)]) % 12] = 2
		return pc_attr_list

	@classmethod
	def convert_pc_list_to_vector(cls, pc_list):
		return [1 if i in pc_list else 0 for i in range(12)]

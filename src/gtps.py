# coding: utf-8
# 2022.01.03

import heapq
import math
import util
import setting
from tps import TPS
import numpy
import torch

class GTPS:
	DISTANCE_ELEMENT_COUNT = 35 # 最大DE番号 + 1
	PC_INT_DISTANCE_ELEMENT_COUNT = 7 # 最大PI_DE番号 + 1
	
	def __init__(self, _b_list, _b_pi_list = []):
		self.tps = TPS()
		self.b_list = [1 if i < len(_b_list) and _b_list[i] == 1 else 0 for i in range(self.DISTANCE_ELEMENT_COUNT)] # 距離要素の有効フラグリスト
		self.matrix_list = []
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 2 * 12)]))         # DE3, key matrix with scale and direction, tps2.py
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 7 * 2 * 12 * 7)])) # DE4, region degree matrix with direction, tps3.py
		#self.matrix_list.append(numpy.array([1.0 for i in range(12 * 12)]))           # tps4.py これは de9_root_degree_pc_matrix があるので不要か
		self.matrix_list.append(numpy.array([1.0 for i in range(12)]))                 # DE5, root_pc_difference_matrix, tps5.py（無意味！）
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 12)]))             # DE6, root_scale_pc_difference_matrix, tps6.py
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 12 * 12)]))        # DE7, root_scale_pc_matrix, tps7.py
		self.matrix_list.append(numpy.array([1.0 for i in range(7 * 12)]))             # DE8, degree matrix with direction, tps8.py
		self.matrix_list.append(numpy.array([1.0 for i in range(4 * 4)]))              # DE9, root_elm_matrix, tps9.py（無意味！）
		self.matrix_list.append(numpy.array([1.0 for i in range(7)]))                  # DE10, parallel key matrix
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 7 * 2)]))          # DE11, key matrix with scale
		self.matrix_list.append(numpy.array([1.0 for i in range(12)]))                 # DE12, parallel key matrix with direction
		self.matrix_list.append(numpy.array([1.0 for i in range(7)]))                  # DE13, relative key matrix
		self.matrix_list.append(numpy.array([1.0 for i in range(12)]))                 # DE14, relative key matrix with direction
		self.matrix_list.append(numpy.array([1.0 for i in range(7 * 7)]))              # DE15, root degree matrix
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 7 * 2 * 7 * 7)]))  # DE16, region degree matrix
		self.matrix_list.append(numpy.array([1.0 for i in range(3 * 7)]))              # DE17, key matrix with symmetrical scale
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 7)]))              # DE18, key matrix with scale diff
		self.matrix_list.append(numpy.array([1.0 for i in range(3)]))                  # DE19, scale matrix
		self.matrix_list.append(numpy.array([1.0 for i in range(4)]))                  # DE20, scale matrix with direction
		self.matrix_list.append(numpy.array([1.0 for i in range(4)]))                  # DE21, sym degree diff
		self.matrix_list.append(numpy.array([1.0 for i in range(2)]))                  # DE22, degree group matrix
		self.matrix_list.append(numpy.array([1.0 for i in range(3)]))                  # DE23, degree group matrix with direction
		self.matrix_list.append(numpy.array([1.0 for i in range(3 * 7 * 7 * 7)]))      # DE24, key degree matrix
		self.matrix_list.append(numpy.array([1.0 for i in range(2)]))                  # DE25, symmetric scale matrix
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 7 * 7 * 7)]))      # DE26, symmetric key-degree matrix
		self.matrix_list.append(numpy.array([1.0 for i in range(7)]))                  # DE27, asym degree diff
		self.matrix_list.append(numpy.array([1.0 for i in range(7 * 7)]))              # DE28, sym degree tonic_diff
		self.matrix_list.append(numpy.array([1.0 for i in range(7 * 12)]))             # DE29, asym degree tonic_diff
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 7 * 7 * 7 * 2)]))  # DE30, DE26 + dest_beat
		self.matrix_list.append(numpy.array([1.0 for i in range(7 * 7 * 2 * 2)]))      # DE31, asym degree beat
		self.matrix_list.append(numpy.array([1.0 for i in range(7 * 7)]))              # DE32, asym degree
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 2)]))              # DE33, asym beat
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 7 * 7 * 7 * 2 * 2)])) # DE34, DE30 + src_beat
		self.matrix_param_count = 0 # 解釈距離の必要パラメータ数
		for i in range(self.DISTANCE_ELEMENT_COUNT - 3):
			if self.b_list[i + 3]:
				self.matrix_param_count += len(self.matrix_list[i])
		self.edge_store = {} # 解釈遷移に対するパラメータの掛かり方を保存する
		self.edge_tensor_store = {} # 解釈遷移行列のtensorを保存する
		self.b_pi_list = [1 if i < len(_b_pi_list) and _b_pi_list[i] == 1 else 0 for i in range(self.PC_INT_DISTANCE_ELEMENT_COUNT)] # PC距離要素の有効フラグリスト
		self.pi_de_list = []
		self.pi_de_list.append(numpy.array([1.0 for i in range(2)]))           # PI_DE2, chord notes / others
		self.pi_de_list.append(numpy.array([1.0 for i in range(3)]))           # PI_DE3, chord notes / scale notes / others
		self.pi_de_list.append(numpy.array([1.0 for i in range(5)]))           # PI_DE4, root / third / fifth / scale notes / others
		self.pi_de_list.append(numpy.array([1.0 for i in range(2 * 5)]))       # PI_DE5, scale * (root / third / fifth / scale notes / others)
		self.pi_de_list.append(numpy.array([1.0 for i in range(2 * 7 * 12)]))  # PI_DE6, scale * degree * (pc * 12)
		self.pi_store = {} # 各PCに対するパラメータの掛かり方を保存する
		self.pi_tensor_store = {} # PI matrix tensorを保存する
	
	# 指定された解釈遷移に対応するdist_listを返す（つまり、パラメータインデックスと重さを返す）。ただしTPSの距離要素の場合はパラメータではなく距離そのものを返す（パラメータは1固定と考えることもできる）
	def get_distance2(self, keypos_1, scale_1, degree_1, arr135_1, beat_1, keypos_2, scale_2, degree_2, arr135_2, beat_2):
		dist_list = []
		if self.b_list[0] or self.b_list[1] or self.b_list[2]:
			temp_sum = self.tps.get_distance2(keypos_1, scale_1, degree_1, arr135_1, keypos_2, scale_2, degree_2, arr135_2) # 常にi,j,kすべてを使った遠隔調計算となる
		if self.b_list[0]:
			dist_list.append(temp_sum[0])
		if self.b_list[1]:
			dist_list.append(temp_sum[1])
		if self.b_list[2]:
			dist_list.append(temp_sum[2])
		for i in range(self.DISTANCE_ELEMENT_COUNT - 3):
			if self.b_list[i + 3]:
				#dist_list.extend([1.0 if idx == self.get_matrix_coefs(i + 3, keypos_1, scale_1, degree_1, arr135_1, keypos_2, scale_2, degree_2, arr135_2) else 0.0 for idx in range(len(self.matrix_list[i]))])
				dist_list.extend(self.get_matrix_coefs(i + 3, keypos_1, scale_1, degree_1, arr135_1, beat_1, keypos_2, scale_2, degree_2, arr135_2, beat_2))
		return dist_list
	
	def get_scalar_distance(self, dist_list):
		if not tuple(dist_list) in self.edge_store.keys():
			current_index = 0
			dist = 0.0
			if dist_list != TPS.ZERO_TPL:
				if self.b_list[0]:
					#dist += dist_list[current_index] * self.tps.coef_i
					dist += dist_list[current_index]
					current_index += 1
				if self.b_list[1]:
					#dist += dist_list[current_index] * self.tps.coef_j
					dist += dist_list[current_index]
					current_index += 1
				if self.b_list[2]:
					#dist += dist_list[current_index] * (self.tps.coef_sum - self.tps.coef_i - self.tps.coef_j) # 2変数版
					dist += dist_list[current_index]
					current_index += 1
				for i in range(self.DISTANCE_ELEMENT_COUNT - 3):
					if self.b_list[i + 3]:
						if i + 3 == 9: # root_elm_matrix
							dist += sum(dist_list[current_index : current_index + len(self.matrix_list[i])] * self.matrix_list[i])
							current_index += len(self.matrix_list[i])
						else: # one-hot系
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
				if self.b_list[0]:
					dist_tensor += dist_list[current_dist_index]
					current_dist_index += 1
				if self.b_list[1]:
					dist_tensor += dist_list[current_dist_index]
					current_dist_index += 1
				if self.b_list[2]:
					dist_tensor += dist_list[current_dist_index]
					current_dist_index += 1
				for i in range(self.DISTANCE_ELEMENT_COUNT - 3):
					if self.b_list[i + 3]:
						if i + 3 == 9: # root_elm_matrix
							for i2 in range(len(self.matrix_list[i])):
								dist_tensor += tensor_param[current_param_index + i2] * dist_list[current_dist_index + i2]
							current_dist_index += len(self.matrix_list[i])
							current_param_index += len(self.matrix_list[i])
						else: # one-hot系
							dist_tensor += tensor_param[current_param_index + dist_list[current_dist_index]]
							current_dist_index += 1
							current_param_index += len(self.matrix_list[i])
			self.edge_tensor_store[tuple(dist_list)] = dist_tensor
			#return torch.exp(-dist_tensor)
			#return -dist_tensor
		return self.edge_tensor_store[tuple(dist_list)]
	
	def get_matrix_coefs(self, distance_element_index, keypos_1, scale_1, degree_1, arr135_1, beat_1, keypos_2, scale_2, degree_2, arr135_2, beat_2):
		if distance_element_index == 3: # DE3, key matrix with scale and direction
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			key_distance = (keypos_2 - keypos_1) % 12
			temp = b_major_1 * 12 * 2 + b_major_2 * 12 + key_distance
			#return [1.0 if idx == temp else 0.0 for idx in range(len(self.matrix_list[distance_element_index - 3]))]
			return [temp]
		elif distance_element_index == 4: # DE4, region degree matrix with direction
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			key_distance = (keypos_2 - keypos_1) % 12
			temp = b_major_1 * 7 * 2 * 12 * 7 + (degree_1 - 1) * 2 * 12 * 7 + b_major_2 * 12 * 7 + key_distance * 7 + (degree_2 - 1)
			#return [1.0 if idx == temp else 0.0 for idx in range(len(self.matrix_list[distance_element_index - 3]))]
			return [temp]
		elif distance_element_index == 5: # DE5, root_pc_difference_matrix
			pc_1 = setting.SCALE_DISTANCE_DIC[scale_1][degree_1 - 1]
			pc_2 = (keypos_2 - keypos_1 + setting.SCALE_DISTANCE_DIC[scale_2][degree_2 - 1]) % 12
			temp = pc_1 * 12 + pc_2
			#return [1.0 if idx == temp else 0.0 for idx in range(len(self.matrix_list[distance_element_index - 3]))]
			return [temp]
		elif distance_element_index == 6: # DE6, root_scale_pc_difference_matrix
			pc_1 = (keypos_1 + setting.SCALE_DISTANCE_DIC[scale_1][degree_1 - 1])
			pc_2 = (keypos_2 + setting.SCALE_DISTANCE_DIC[scale_2][degree_2 - 1])
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			temp = b_major_1 * 12 + ((pc_2 - pc_1) % 12)
			#return [1.0 if idx == temp else 0.0 for idx in range(len(self.matrix_list[distance_element_index - 3]))]
			return [temp]
		elif distance_element_index == 7: # DE7, root_scale_pc_matrix
			pc_1 = setting.SCALE_DISTANCE_DIC[scale_1][degree_1 - 1]
			pc_2 = (keypos_2 - keypos_1 + setting.SCALE_DISTANCE_DIC[scale_2][degree_2 - 1]) % 12
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			temp = b_major_1 * 12 * 12 + pc_1 * 12 + pc_2
			#return [1.0 if idx == temp else 0.0 for idx in range(len(self.matrix_list[distance_element_index - 3]))]
			return [temp]
		elif distance_element_index == 8: # DE8, root_degree_pc_matrix
			pc_2 = (keypos_2 - keypos_1 + setting.SCALE_DISTANCE_DIC[scale_2][degree_2 - 1]) % 12 # 遷移元keyの1度の音に対して
			temp = (degree_1 - 1) * 12 + pc_2
			#return [1.0 if idx == temp else 0.0 for idx in range(len(self.matrix_list[distance_element_index - 3]))]
			return [temp]
		elif distance_element_index == 9: # DE9, root_elm_matrix
			bs1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			bs2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			scale1 = setting.SCALE_DISTANCE_DIC[scale_1]
			scale2 = setting.SCALE_DISTANCE_DIC[scale_2]
			bs1[(scale1[(0 + degree_1 - 1) % 7] + keypos_1) % 12] = 1 # root
			bs1[(scale1[(2 + degree_1 - 1) % 7] + keypos_1) % 12] = 2 # 3rd
			bs1[(scale1[(4 + degree_1 - 1) % 7] + keypos_1) % 12] = 3 # 5th
			bs2[(scale2[(0 + degree_2 - 1) % 7] + keypos_2) % 12] = 1 # root
			bs2[(scale2[(2 + degree_2 - 1) % 7] + keypos_2) % 12] = 2 # 3rd
			bs2[(scale2[(4 + degree_2 - 1) % 7] + keypos_2) % 12] = 3 # 5th
			ret = [0.0 for _ in range(4 * 4)]
			for i in range(12):
				ret[bs1[i] * 4 + bs2[i]] += 1.0
			return ret
		elif distance_element_index == 10: # DE10, parallel key matrix
			temp = min((keypos_1 - keypos_2) % 12, (keypos_2 - keypos_1) % 12)
			return [temp]
		elif distance_element_index == 11: # DE11, key matrix with scale
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			temp = b_major_1 * 7 * 2 + min((keypos_1 - keypos_2) % 12, (keypos_2 - keypos_1) % 12) * 2 + b_major_2
			return [temp]
		elif distance_element_index == 12: # DE12, parallel key matrix with direction
			temp = (keypos_2 - keypos_1) % 12
			return [temp]
		elif distance_element_index == 13: # DE13, relative key matrix
			major_keypos_1 = keypos_1
			if scale_1 != setting.SCALE_MAJOR:
				major_keypos_1 += 3
			major_keypos_2 = keypos_2
			if scale_2 != setting.SCALE_MAJOR:
				major_keypos_2 += 3
			temp = min((major_keypos_1 - major_keypos_2) % 12, (major_keypos_2 - major_keypos_1) % 12)
			return [temp]
		elif distance_element_index == 14: # DE14, relative key matrix with direction
			major_keypos_1 = keypos_1
			if scale_1 != setting.SCALE_MAJOR:
				major_keypos_1 += 3
			major_keypos_2 = keypos_2
			if scale_2 != setting.SCALE_MAJOR:
				major_keypos_2 += 3
			temp = (major_keypos_2 - major_keypos_1) % 12
			return [temp]
		elif distance_element_index == 15: # DE15, root degree matrix
			pc_2 = (keypos_2 - keypos_1 + setting.SCALE_DISTANCE_DIC[scale_2][degree_2 - 1]) % 12 # 遷移元keyの1度の音に対して
			temp = (degree_1 - 1) * 7 + min(pc_2, 12 - pc_2)
			return [temp]
		elif distance_element_index == 16: # DE16, region degree matrix
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			key_distance = min((keypos_2 - keypos_1) % 12, (keypos_1 - keypos_2) % 12)
			temp = b_major_1 * 7 * 2 * 7 * 7 + (degree_1 - 1) * 2 * 7 * 7 + b_major_2 * 7 * 7 + key_distance * 7 + (degree_2 - 1)
			return [temp]
		elif distance_element_index == 17: # DE17, key matrix with symmetrical scale
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			temp = (b_major_1 + b_major_2) * 7 + min((keypos_1 - keypos_2) % 12, (keypos_2 - keypos_1) % 12)
			return [temp]
		elif distance_element_index == 18: # DE18, key matrix with scale diff
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			temp1 = 1
			if b_major_1 == b_major_2:
				temp1 = 0
			temp = ((b_major_1 - b_major_2) % 2) * 7 + min((keypos_1 - keypos_2) % 12, (keypos_2 - keypos_1) % 12)
			return [temp]
		elif distance_element_index == 19: # DE19, scale matrix
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			temp = b_major_1 + b_major_2
			return [temp]
		elif distance_element_index == 20: # DE20, scale matrix with direction
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			temp = b_major_1 * 2 + b_major_2
			return [temp]
		elif distance_element_index == 21: # DE21, sym degree diff
			degree_distance = min((degree_2 - degree_1) % 7, (degree_1 - degree_2) % 7)
			return [degree_distance]
		elif distance_element_index == 22: # DE22, degree group matrix
			temp1 = 0 # tonic
			if degree_1 == 5 or degree_1 == 7:
				temp1 = 1 # dominant
			elif degree_1 == 2 or degree_1 == 4:
				temp1 = 2 # sub dominant
			temp2 = 0 # tonic
			if degree_2 == 5 or degree_2 == 7:
				temp2 = 1 # dominant
			elif degree_2 == 2 or degree_2 == 4:
				temp2 = 2 # sub dominant
			temp = min((temp2 - temp1) % 3, (temp1 - temp2) % 3)
			return [temp]
		elif distance_element_index == 23: # DE23, degree group matrix with direction
			temp1 = 0 # tonic
			if degree_1 == 5 or degree_1 == 7:
				temp1 = 1 # dominant
			elif degree_1 == 2 or degree_1 == 4:
				temp1 = 2 # sub dominant
			temp2 = 0 # tonic
			if degree_2 == 5 or degree_2 == 7:
				temp2 = 1 # dominant
			elif degree_2 == 2 or degree_2 == 4:
				temp2 = 2 # sub dominant
			temp = (temp2 - temp1) % 3
			return [temp]
		elif distance_element_index == 24: # DE24, key degree matrix
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			key_distance = min((keypos_2 - keypos_1) % 12, (keypos_1 - keypos_2) % 12)
			temp = (b_major_1 + b_major_2) * 7 * 7 * 7 + (degree_1 - 1) * 7 * 7 + (degree_2 - 1) * 7 + key_distance
			return [temp]
		elif distance_element_index == 25: # DE25, symmetric scale matrix
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			temp = (b_major_1 - b_major_2) % 2
			return [temp]
		elif distance_element_index == 26: # DE26, symmetric key degree matrix
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			scale_diff = (b_major_1 - b_major_2) % 2
			key_distance = min((keypos_2 - keypos_1) % 12, (keypos_1 - keypos_2) % 12)
			temp = scale_diff * 7 * 7 * 7 + (degree_1 - 1) * 7 * 7 + (degree_2 - 1) * 7 + key_distance
			return [temp]
		elif distance_element_index == 27: # DE27, asym degree diff
			degree_distance = (degree_2 - degree_1) % 7
			return [degree_distance]
		elif distance_element_index == 28: # DE28, sym degree tonic_diff
			tonic_diff = min((keypos_1 - keypos_2) % 12, (keypos_2 - keypos_1) % 12)
			temp = (degree_1 - 1) * 7 + tonic_diff
			return [temp]
		elif distance_element_index == 29: # DE29, asym degree tonic_diff
			tonic_diff = (keypos_2 - keypos_1) % 12
			temp = (degree_1 - 1) * 12 + tonic_diff
			return [temp]
		elif distance_element_index == 30: # DE30, DE26 + dest_beat
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			scale_diff = (b_major_1 - b_major_2) % 2
			key_distance = min((keypos_2 - keypos_1) % 12, (keypos_1 - keypos_2) % 12)
			temp = scale_diff * 7 * 7 * 7 * 2 + (degree_1 - 1) * 7 * 7 * 2 + (degree_2 - 1) * 7 * 2 + key_distance * 2 + beat_2
			return [temp]
		elif distance_element_index == 31: # DE31, asym degree beat
			temp = (degree_1 - 1) * 7 * 2 * 2 + (degree_2 - 1) * 2 * 2 + beat_1 * 2 + beat_2
			return [temp]
		elif distance_element_index == 32: # DE32, asym degree
			temp = (degree_1 - 1) * 7 + (degree_2 - 1)
			return [temp]
		elif distance_element_index == 33: # DE33, asym beat
			temp = beat_1 * 2 + beat_2
			return [temp]
		elif distance_element_index == 34: # DE34, DE30 + src_beat
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			scale_diff = (b_major_1 - b_major_2) % 2
			key_distance = min((keypos_2 - keypos_1) % 12, (keypos_1 - keypos_2) % 12)
			temp = scale_diff * 7 * 7 * 7 * 2 * 2 + (degree_1 - 1) * 7 * 7 * 2 * 2 + (degree_2 - 1) * 7 * 2 * 2 + key_distance * 2 * 2 + beat_1 * 2 + beat_2
			return [temp]
		else:
			print('get_matrix_indices: 不正なDE_indexです', distance_element_index)
			return []
	
	# chromae_listを適用する前の、各PCに対するパラメータの掛かり方を返す
	#   戻り値：[pc: [param_index: weight]]
	def get_pi_distance(self, keypos, scale, degree, arr135):
		pc_dist_list_x2 = [[] for i in range(12)]
		pc_attr_list = self.convert_tpl_to_pc_attr_list((keypos, scale, degree))
		if self.b_pi_list[0]: # PI_DE1, basicspace（-13ずれる）
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
		if self.b_pi_list[1]: # PI_DE2, chord notes / others
			for pc in range(12):
				if pc_attr_list[pc] in (4, 3, 2): # root, third, fifth
					pc_dist_list_x2[pc].append(1)
				else:
					pc_dist_list_x2[pc].append(0)
		if self.b_pi_list[2]: # PI_DE3, chord notes / scale notes / others
			for pc in range(12):
				if pc_attr_list[pc] in (4, 3, 2): # root, third, fifth
					pc_dist_list_x2[pc].append(2)
				elif pc_attr_list[pc] == 1: # scale
					pc_dist_list_x2[pc].append(1)
				else:
					pc_dist_list_x2[pc].append(0)
		if self.b_pi_list[3]: # PIPI_DE4, root / third / fifth / scale notes / others
			for pc in range(12):
				pc_dist_list_x2[pc].append(pc_attr_list[pc])
		if self.b_pi_list[4]: # PI_DE5, scale * (root / third / fifth / scale notes / others)
			for pc in range(12):
				pc_dist_list_x2[pc].append(5 * scale + pc_attr_list[pc])
		if self.b_pi_list[5]: # PI_DE6, scale * degree * (pc * 12)
			for pc in range(12):
				pc_dist_list_x2[pc].append(scale * 7 * 12 + (degree - 1) * 12 + pc_attr_list[pc])
		# インデックスにするために内部をtuple化
		ret = []
		for pc in range(12):
			ret.append(tuple(pc_dist_list_x2[pc]))
		#print(keypos, scale, degree, ret)
		return ret
	
	# chroma_list = [pc: energy(weight)], pc_dist_list_x2 = [pc: [param_index: weight]]
	def get_pi_scalar_distance(self, chroma_list, pc_dist_list_x2):
		# まず、各PCに対する距離の係数を得る（内部変数にはその形で保存する）
		if not tuple(pc_dist_list_x2) in self.pi_store.keys():
			chroma_pc_dist_list = [] # [pc: paramを掛けた後の合計weight]
			for pc in range(12):
				current_index = 0
				chroma_pc_dist_list.append(0.0)
				if self.b_pi_list[0]: # PI_DE1, basicspace（-13ずれる）
					chroma_pc_dist_list[pc] += pc_dist_list_x2[pc][current_index]
					current_index += 1
				for i in range(self.PC_INT_DISTANCE_ELEMENT_COUNT - 1):
					if self.b_pi_list[i + 1]:
						chroma_pc_dist_list[pc]  += self.pi_de_list[i][pc_dist_list_x2[pc][current_index]] # one-hot系
						current_index += 1
			self.pi_store[tuple(pc_dist_list_x2)] = chroma_pc_dist_list
		else:
			chroma_pc_dist_list = self.pi_store[tuple(pc_dist_list_x2)]
		# そして、PCベクトルとそれの内積を取って距離を計算する
		ret = sum([chroma_list[i] * chroma_pc_dist_list[i] for i in range(12)])
		#ret = sum([1 * chroma_pc_dist_list[pc] for pc in pc_list]) # pc_list版
		#print(ret, chroma_list, chroma_pc_dist_list, pc_dist_list_x2)
		return ret
	
	def get_pi_scalar_distance_tensor(self, chroma_list, pc_dist_list_x2, tensor_param):
		# まず、各PCに対する距離の係数のtensorを得る（内部変数にはその形で保存する）
		if not tuple(pc_dist_list_x2) in self.pi_tensor_store.keys():
			chroma_dist_tensor_list = torch.tensor([0.0 for i in range(12)])
			for pc in range(12):
				current_dist_index = 0
				current_param_index = self.matrix_param_count
				if self.b_pi_list[0]: # PI_DE1, basicspace（-13ずれる）
					chroma_dist_tensor_list[pc] = chroma_dist_tensor_list[pc] + pc_dist_list_x2[pc][current_dist_index]
					current_dist_index += 1
				for i in range(self.PC_INT_DISTANCE_ELEMENT_COUNT - 1):
					if self.b_pi_list[i + 1]:
						chroma_dist_tensor_list[pc] = chroma_dist_tensor_list[pc] + tensor_param[current_param_index + pc_dist_list_x2[pc][current_dist_index]] # one-hot系
						current_dist_index += 1
						current_param_index += len(self.pi_de_list[i])
			self.pi_tensor_store[tuple(pc_dist_list_x2)] = chroma_dist_tensor_list
		else:
			chroma_dist_tensor_list = self.pi_tensor_store[tuple(pc_dist_list_x2)]
		#dist_tensor_list = self.pi_tensor_store[tuple(pc_dist_list_x2)]
		ret = torch.sum(chroma_dist_tensor_list * torch.tensor(chroma_list))
		#ret = torch.einsum('i, i -> ', chroma_dist_tensor_list, torch.tensor(chroma_list))
		#print(ret, chroma_list, chroma_dist_tensor_list)
		return ret
		#return torch.sum(torch.tensor([1 * chroma_dist_tensor_list[pc] for pc in pc_list])) # pc_list版

	def update_params(self, tensor_param):
		current_index = 0
		for i in range(self.DISTANCE_ELEMENT_COUNT - 3):
			if self.b_list[i + 3]:
				for i2 in range(len(self.matrix_list[i])):
					self.matrix_list[i][i2] = tensor_param[current_index].item()
					current_index += 1
		for i in range(self.PC_INT_DISTANCE_ELEMENT_COUNT - 1):
			if self.b_pi_list[i + 1]:
				for i2 in range(len(self.pi_de_list[i])):
					self.pi_de_list[i][i2] = tensor_param[current_index].item()
					current_index += 1
		#print(self.pi_de_list)
		self.edge_store = {}
		self.edge_tensor_store = {}
		self.pi_store = {}
		self.pi_tensor_store = {}

	#! (key, scale, degree)を構成音のpitch class列（[root, third, fifth]）に変換
	#    ではなく、chromaリストに変換
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
	
	#! (key, scale, degree)を各pitch classの属性値（4: root, 3: third, 2: fifth, 1: その他scale音, 0: scale外）列に変換
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

	#! pitch class列を12次元のバイナリベクトルに変換
	@classmethod
	def convert_pc_list_to_vector(cls, pc_list):
		return [1 if i in pc_list else 0 for i in range(12)]

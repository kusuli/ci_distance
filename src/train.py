# 2023.10.08

import os
import math
import sys
import re
import torch
import numpy
import statistics as st
import util
import random
from multiprocessing import Pool

from tps import TPS
from gtps import GTPS
import setting
#from base import Base
#from ext1 import Ext1


def rn_to_int(rn):
	rn.replace('#', '').replace('b', '')
	rn = rn.upper()
	if rn == 'I':
		return 1
	elif rn == 'II':
		return 2
	elif rn == 'III':
		return 3
	elif rn == 'IV':
		return 4
	elif rn == 'V':
		return 5
	elif rn == 'VI':
		return 6
	elif rn == 'VII':
		return 7
	else:
		print('rn_to_int error: ' + rn)
		raise Exception

def convert_csv_str(csv_line):
	arr = csv_line.split(',') # measure number, C, C#, D, D#, E, F, F#, G, G#, A, A#, B, count, (key tonic, key mode, figure, degree, quality, inversion, root) × count
	int_num = int(arr[13]) # count
	ret_list = []
	for i in range(int_num):
		key = setting.NOTE_POS_DIC[arr[14 + i * 7].strip()]
		scale = int(arr[15 + i * 7])
		degree = int(arr[17 + i * 7])
		beat = 0 # dummy
		ret_list.append((key, scale, degree, beat))
	return ret_list

def load_rntxt(data_path):
	ret_tpl_list = []
	with open(data_path, "r") as f:
		#print(data_path)
		prev_tpl_list = [] # "epsilon"で使用
		for line in f:
			#print(line)
			tpl_list = convert_csv_str(line.replace("\n", '')) 
			if len(tpl_list) == 0:
				pass
			else:
				if False: # applied chordを全部含める
					for tpl in tpl_list:
						if True: # "applied unique"。重複するコード・解釈を排除
							if len(ret_tpl_list) == 0 or ret_tpl_list[-1] != tpl:
								ret_tpl_list.append(tpl)
						else: # "applied"
							ret_tpl_list.append(tpl)
				else:
					mode = "original"
					if mode == "local": # "local"。applied chordは最終的に表れているdegree/keyを使用。SMC2021はこれ
						ret_tpl_list.append(tpl_list[0])
					elif mode == "bottom": # "bottom"。applied chordは無視して元のkeyの部分を使用
						ret_tpl_list.append(tpl_list[-1])
					elif mode == "original": # "original"。applied chordで元keyで無理やりdegreeを当てはめる
						degree = 0
						for tpl in tpl_list:
							degree += tpl[2] - 1
						degree = (degree % 7) + 1
						tpl = tpl_list[-1]
						ret_tpl_list.append((tpl[0], tpl[1], degree, tpl[3]))
						#print(line, ret_tpl_list[-1])
					elif mode == "epsilon": # "epsilon"。"top"と同様だが、applied区間の最後にそのキーの一度を加える
						if len(prev_tpl_list) > len(tpl_list): # applied chord数が減っている場合
							tpl = (prev_tpl_list[0][0], prev_tpl_list[0][1], 1, prev_tpl_list[0][3])
							ret_tpl_list.append(tpl)
						ret_tpl_list.append(tpl_list[0]) # ここは"top"と同じ
					else:
						print("load_rntxt エラー：", mode)
						exit()
			prev_tpl_list = tpl_list
		temp_tpl_list = []
		for i, tpl in enumerate(ret_tpl_list):
			if i == 0 or ret_tpl_list[i - 1] != tpl:
				temp_tpl_list.append(tpl)
		ret_tpl_list_x2 = [temp_tpl_list]
	return ret_tpl_list_x2

#! (key, scale, degree, beat) → (chord name, root_pos, chord_type, beat)
def convert_tpl_to_tpl2(tpl):
	root_pos = tpl[0] + setting.SCALE_DISTANCE_DIC[tpl[1]][tpl[2] - 1]
	root_pos %= 12
	chord_str = [k for k, v in setting.NOTE_POS_DIC.items() if v == root_pos][0]
	if tpl[1] == setting.SCALE_MAJOR:
		if tpl[2] == 1 or tpl[2] == 4 or tpl[2] == 5:
			chord_str += ':maj'
			chord_type = setting.CHORD_TYPE_MAJ
		elif tpl[2] == 2 or tpl[2] == 3 or tpl[2] == 6:
			chord_str += ':min'
			chord_type = setting.CHORD_TYPE_MIN
		elif tpl[2] == 7:
			chord_str += ':dim'
			chord_type = setting.CHORD_TYPE_DIM7
		else:
			print('convert_tpl_to_chord_name error: ' + tpl)
			raise Exception
	else:
		if tpl[2] == 3 or tpl[2] == 6 or tpl[2] == 7:
			chord_str += ':maj'
			chord_type = setting.CHORD_TYPE_MAJ
		elif tpl[2] == 1 or tpl[2] == 4 or tpl[2] == 5:
			chord_str += ':min'
			chord_type = setting.CHORD_TYPE_MIN
		elif tpl[2] == 2:
			chord_str += ':dim'
			chord_type = setting.CHORD_TYPE_DIM7 
		else:
			print('convert_tpl_to_chord_name error: ' + tpl)
			raise Exception
	return (chord_str, root_pos, chord_type, tpl[3])

#! (key, scale, degree, beat) → ((key, scale, degree, beat), cost)
def get_chord_interpretation_list(tpl):
	(chord_str, root_pos, chord_type, beat) = convert_tpl_to_tpl2(tpl)
	#print(chord_str, root_pos, chord_type)
	if chord_type == setting.CHORD_TYPE_MAJ:
		return [
				((root_pos, setting.SCALE_MAJOR, 1, beat), 0),
				(((root_pos + 9) % 12, setting.SCALE_NATURAL_MINOR, 3, beat), 0),
				(((root_pos + 7) % 12, setting.SCALE_MAJOR, 4, beat), 0),
				(((root_pos + 4) % 12, setting.SCALE_NATURAL_MINOR, 6, beat), 0),
				(((root_pos + 5) % 12, setting.SCALE_MAJOR, 5, beat), 0),
				(((root_pos + 2) % 12, setting.SCALE_NATURAL_MINOR, 7, beat), 0)
		]
	elif chord_type == setting.CHORD_TYPE_MIN:
		return [
				((root_pos, setting.SCALE_NATURAL_MINOR, 1, beat), 0),
				(((root_pos + 3) % 12, setting.SCALE_MAJOR, 6, beat), 0),
				(((root_pos + 7) % 12, setting.SCALE_NATURAL_MINOR, 4, beat), 0),
				(((root_pos + 10) % 12, setting.SCALE_MAJOR, 2, beat), 0),
				(((root_pos + 5) % 12, setting.SCALE_NATURAL_MINOR, 5, beat), 0),
				(((root_pos + 8) % 12, setting.SCALE_MAJOR, 3, beat), 0)
		]
	elif chord_type == setting.CHORD_TYPE_DIM7:
		return [
				(((root_pos + 1) % 12, setting.SCALE_MAJOR, 7, beat), 0),
				(((root_pos + 10) % 12, setting.SCALE_NATURAL_MINOR, 2, beat), 0)
		]
	else:
		print('get_chord_interpretation エラー: ' + tpl)
		raise Exception

def get_chord_interpretation_list_x2(tpl_list):
	ret_list = []
	for tpl in tpl_list:
		ret_list.append(get_chord_interpretation_list(tpl))
		#print(tpl, ret_list[-1])
	return ret_list

def make_interpretation_graph(arg_list):
	(gtps, interpretation_list_x2) = arg_list
	node_list_x2 = [] # [layer index: [node index in the layer: [key, scale, degree, 以前の情報だけを考慮した場合にこのノードに至る確率]]]
	edge_distance_list_x3 = [] # [元layer index: [元node index: [先node index: (GTPS距離係数LIST)]]]（GTPS距離自体はパラメータ依存だが、(region距離要素)は非依存
	# node_list_x2側
	for i, interpretation_list in enumerate(interpretation_list_x2):
		node_list = []
		for i2, interpretation in enumerate(interpretation_list):
			node_list.append([interpretation[0][0], interpretation[0][1], interpretation[0][2], 0]) # 確率に関してはとりあえず0
		node_list_x2.append(node_list)
	# edge_distance_list_x3側
	for i, node_list in enumerate(node_list_x2):
		if i < len(node_list_x2) - 1:
			edge_distance_list_x2 = []
			for i2, node in enumerate(node_list):
				edge_distance_list = []
				for i3, next_node in enumerate(node_list_x2[i + 1]):
					sum_tpl = gtps.get_distance2(node[0], node[1], node[2], next_node[0], next_node[1], next_node[2])
					#edge_distance_list.append([tps.get_scalar_distance(sum_tpl), sum_tpl])
					edge_distance_list.append(sum_tpl)
				edge_distance_list_x2.append(edge_distance_list)
			edge_distance_list_x3.append(edge_distance_list_x2)
	# 最後の層としてendノードを追加
	edge_distance_list_x2 = []
	for i, node in enumerate(node_list_x2[-1]):
		edge_distance_list_x2.append([TPS.ZERO_TPL]) # 全距離要素が0ということで
	edge_distance_list_x3.append(edge_distance_list_x2)
	node_list_x2.append([[0, 0, 0, 0]]) # endノード。内容は適当
	return (node_list_x2, edge_distance_list_x3)

#! 解釈グラフにedge確率情報を追加
def add_probability_to_interpretation_graph(gtps, graph):
	node_list_x2 = graph[0]
	edge_distance_list_x3 = graph[1]
	edge_probability_list_x3 = [] # [layer index: [元node index: [先node index: exp(-GTPS距離) / Z(n)]]
	# 最初の層のノード到達確率をセット
	for node_index in range(len(node_list_x2[0])):
		node_list_x2[0][node_index][3] = 1 / len(node_list_x2[0])
	# 各node、edgeの確率を計算
	for n in range(len(node_list_x2) - 1):
		Z = 0
		src_node_list = node_list_x2[n]
		dest_node_list = node_list_x2[n + 1]
		edge_probability_list_x2 = []
		# GTPS距離から正規化されないedge確率を計算
		for src_node_index, src_node in enumerate(src_node_list):
			edge_probability_list = []
			for dest_node_index, dest_node in enumerate(dest_node_list):
				#temp = math.exp(-edge_distance_list_x3[n][src_node_index][dest_node_index][0])
				temp = math.exp(-gtps.get_scalar_distance(edge_distance_list_x3[n][src_node_index][dest_node_index]))
				edge_probability_list.append(temp)
				Z += src_node[3] * temp
				#print(n, src_node_index, dest_node_index, src_node[4], temp, edge_distance_list_x3[n][src_node_index][dest_node_index][0], edge_distance_list_x3[n][src_node_index][dest_node_index][1])
			edge_probability_list_x2.append(edge_probability_list)
		edge_probability_list_x3.append(edge_probability_list_x2)
		# edge確率を正規化
		for src_node_index, src_node in enumerate(src_node_list):
			if False:
				Z = sum(edge_probability_list_x2[src_node_index]) # src単位でZを計算する
			for dest_node_index, dest_node in enumerate(dest_node_list):
				edge_probability_list_x2[src_node_index][dest_node_index] /= Z
		# dest nodeの到達確率をセット
		for dest_node_index, dest_node in enumerate(dest_node_list):
			temp = 0
			for src_node_index, src_node in enumerate(src_node_list):
				temp += src_node[3] * edge_probability_list_x2[src_node_index][dest_node_index]
			dest_node[3] = temp
	# 拡張したgraphを返す
	return (node_list_x2, edge_distance_list_x3, edge_probability_list_x3)

#! 解釈グラフ（到達確率は設定不要）から各ノードに至る最短経路での直前ノードへのリンクのリストを求める
#    return: [layer index: [node index: [back node index]]]
def get_back_link_list_x3(gtps, graph):
	small_value = 0.00000001
	node_list_x2 = graph[0]
	edge_distance_list_x3 = graph[1]
	back_link_list_x3 = [] # これを返す
	node_cost_list_x2 = [] # 各ノードの、最短到達コストを保持する（計算用）
	# 最初の層の計算
	back_link_list_x3.append([[] for i in node_list_x2[0]]) # ここはブランク
	node_cost_list_x2.append([0 for i in node_list_x2[0]])
	# そして各層の計算
	for n in range(len(node_list_x2) - 1):
		src_node_list = node_list_x2[n]
		dest_node_list = node_list_x2[n + 1]
		back_link_list_x2 = []
		node_cost_list = []
		# 各destノードに対して、最短経路を与えるsrcノードのリストを計算する
		for dest_node_index, dest_node in enumerate(dest_node_list):
			min_distance = math.inf
			cand_list = []
			for src_node_index, src_node in enumerate(src_node_list):
				#distance = node_cost_list_x2[n][src_node_index] + edge_distance_list_x3[n][src_node_index][dest_node_index][0]
				distance = node_cost_list_x2[n][src_node_index] + gtps.get_scalar_distance(edge_distance_list_x3[n][src_node_index][dest_node_index])
				if  (min_distance - small_value) <= distance <= (min_distance + small_value):
					cand_list.append(src_node_index)
				elif distance < min_distance + small_value:
					cand_list = [src_node_index]
					min_distance = distance
			back_link_list_x2.append(cand_list)
			node_cost_list.append(min_distance)
		back_link_list_x3.append(back_link_list_x2)
		node_cost_list_x2.append(node_cost_list)
	return back_link_list_x3

#! 直前ノードへのリンクのリストから、各層で正規化された各ノードの到達確率を求める
#    return: [layer index: [node index: 確率]]
#    経路の確率を出すための途中計算で出てくるノード到達確率とは別（そっちはその時点までのグラフしか考慮していない確率）
def get_node_probability_list_x2(back_link_list_x3):
	node_probability_list_x2 = [[0 for node_index in back_link_list_x2] for back_link_list_x2 in back_link_list_x3] # これを返す
	node_path_count_list_x2 = [[0 for node_index in back_link_list_x2] for back_link_list_x2 in back_link_list_x3] # [layer index: [node index: このノードから始まる経路の本数]]
	# 末尾から逆にたどって、各ノードについてそのノードから始まる経路数を edge_weight_list_x3 にセットしていく
	if True:
		# endノードから始まる経路数は1
		if True:
			node_path_count_list_x2[-1][0] = 1
		# 残りの層を計算
		for n0 in range(len(back_link_list_x3) - 1):
			n = len(back_link_list_x3) - 2 - n0
			back_link_list_x2 = back_link_list_x3[n + 1] # dest → src のリンクだよ
			for dest_node_index, back_link_list in enumerate(back_link_list_x2):
				path_count = node_path_count_list_x2[n + 1][dest_node_index]
				for src_node_index in back_link_list:
					node_path_count_list_x2[n][src_node_index] += path_count
	#print(node_path_count_list_x2)
	# 今度は先頭から順にたどって、各ノードについてそのノードを通る経路数を node_probability_list_x2 にセットしていく
	if True:
		# 先頭は node_path_count_list_x2 の結果を正規化したもの
		if True:
			path_sum = sum([node_probability for node_probability in node_path_count_list_x2[0]])
			for node_index, path_count in enumerate(node_path_count_list_x2[0]):
				node_probability_list_x2[0][node_index] = path_count / path_sum
		# 残りの層を計算
		for n in range(len(back_link_list_x3) - 1):
			back_link_list_x2 = back_link_list_x3[n + 1] # dest → src のリンクだからね
			for src_node_index in range(len(node_path_count_list_x2[n])):
				# まずリンク先（src → dest）のpath_countの合計をとる
				path_count_sum = sum([node_path_count_list_x2[n + 1][dest_node_index] for dest_node_index, src_node_index_list in enumerate(back_link_list_x2) if src_node_index in src_node_index_list])
				if path_count_sum > 0:
					#print(n, src_node_index, path_count_sum)
					# そしてsrc側のノードの経路数を分配していく
					for dest_node_index, back_link_list in enumerate(back_link_list_x2):
						if src_node_index in back_link_list:
							node_probability_list_x2[n + 1][dest_node_index] += node_probability_list_x2[n][src_node_index] * (node_path_count_list_x2[n + 1][dest_node_index] / path_count_sum)
	#print(node_probability_list_x2)
	return node_probability_list_x2

#! 解釈グラフ（到達確率は設定不要）から確率の計算グラフを生成
def make_computation_graph(graph, gtps, tensor_param):
	node_list_x2 = graph[0]
	edge_distance_list_x3 = graph[1]
	tensor_edge_list = []
	# 最初の層のノード到達確率をセット
	tensor_node_list = torch.tensor([1.0 / len(node_list_x2[0]) for elm in node_list_x2[0]])
	# 各node、edgeの確率を計算
	for n in range(len(node_list_x2) - 2): # node_list_x2の最後の層はendノードなので無視
		src_node_list = node_list_x2[n]
		dest_node_list = node_list_x2[n + 1]
		# GTPS距離から正規化されないedge確率を計算
		tensor_edge = torch.tensor(numpy.zeros((len(src_node_list), len(dest_node_list))))
		for src_node_index, src_node in enumerate(src_node_list):
			for dest_node_index, dest_node in enumerate(dest_node_list):
				tensor_edge[src_node_index][dest_node_index] = -gtps.get_scalar_distance_tensor(edge_distance_list_x3[n][src_node_index][dest_node_index], tensor_param)
		tensor_edge = torch.exp(tensor_edge)
		if False:
			tensor_Z = tensor_edge.sum(1)
			tensor_edge = (tensor_edge.t() / tensor_Z).t()
		else:
			tensor_Z = torch.sum(tensor_node_list * tensor_edge.sum(1))
			tensor_edge = tensor_edge / tensor_Z # edge確率を正規化
		tensor_edge_list.append(tensor_edge)
		#print(n, tensor_Z, tensor_node_list, tensor_edge.sum(1), tensor_edge)
		# dest nodeの到達確率をセット
		prev_tensor_node_list = tensor_node_list
		tensor_node_list = torch.tensor([0.0 for elm in node_list_x2[n + 1]])
		for dest_node_index, dest_node in enumerate(dest_node_list):
			tensor_node_list[dest_node_index] = sum([prev_tensor_node_list[src_node_index] * tensor_edge[src_node_index][dest_node_index] for src_node_index in range(len(src_node_list))])
			#print(tensor_node_list[dest_node_index], [prev_tensor_node_list[src_node_index] * tensor_edge[src_node_index][dest_node_index] for src_node_index in range(len(src_node_list))])
		#exit()
	return tensor_edge_list

#! 経路探索してaccuracyを計算する
def get_accuracy(arg_list):
	(gtps, answer_tpl_list, interpretation_list_x2, graph) = arg_list
	#graph = make_interpretation_graph(gtps, interpretation_list_x2)
	node_list_x2 = graph[0]
	back_link_list_x3 = get_back_link_list_x3(gtps, graph)
	#print(back_link_list_x3)
	node_probability_list_x2 = get_node_probability_list_x2(back_link_list_x3)
	#print(node_probability_list_x2)
	answer_index_list = []
	for n, tpl in enumerate(answer_tpl_list):
		node_list = node_list_x2[n]
		answer_index = -1
		for i, node in enumerate(node_list):
			if tpl[0] == node[0] and tpl[1] == node[1] and tpl[2] == node[2]:
				answer_index = i
				break
		if answer_index < 0:
			print('エラー')
		answer_index_list.append(answer_index)
	correct_count = 0.0
	for n, tpl in enumerate(answer_tpl_list):
		correct_count += node_probability_list_x2[n][answer_index_list[n]]
	return (correct_count, len(answer_tpl_list))

def get_nl_prob(arg_list):
	(gtps, answer_tpl_list, graph) = arg_list
	graph2 = add_probability_to_interpretation_graph(gtps, graph)
	nl_prob = calc_gradient(graph, answer_tpl_list, graph2[2], False)
	return nl_prob

def get_average_accuracy(gtps, tpl_list_x2, int_list_x3, graph_list, parallel_count=8, b_nl_prob=False):
	acc_list = []
	#shortest_path_count_list = []
	file_count = len(tpl_list_x2)
	with Pool(parallel_count) as p:
		acc_list = p.map(func=get_accuracy, iterable=zip([gtps] * len(tpl_list_x2), tpl_list_x2, int_list_x3, graph_list))
	if True: 
		acc_list2 = [v[0] / v[1] for v in acc_list]
		mean = st.mean(acc_list2)
	else: 
		mean = sum([v[0] for v in acc_list]) / sum([v[1] for v in acc_list])
	nl_prob_sum = 0.0
	if b_nl_prob:
		with Pool(parallel_count) as p:
			nl_prob_list = p.map(func=get_nl_prob, iterable=zip([gtps] * len(tpl_list_x2), tpl_list_x2, graph_list))
		nl_prob_sum = sum(nl_prob_list)
	return (round(mean, 4), 0, round(nl_prob_sum, 4))

def calc_gradient(graph, answer_tpl_list, edge_list, b_tensor):
	node_list_x2 = graph[0]
	edge_distance_list_x3 = graph[1]
	answer_index_list = []
	for n, tpl in enumerate(answer_tpl_list):
		node_list = node_list_x2[n]
		answer_index = -1
		for i, node in enumerate(node_list):
			if tpl[0] == node[0] and tpl[1] == node[1] and tpl[2] == node[2]:
				answer_index = i
				break
		if answer_index < 0:
			print('エラー: answer_indexがありません')
		answer_index_list.append(answer_index)
	if b_tensor:
		nl_prob = torch.tensor(0.0)
	else:
		nl_prob = 0.0
	for n in range(len(answer_index_list) - 1):
		edge = edge_list[n]
		if b_tensor:
			nl_prob -= torch.log(edge[answer_index_list[n]][answer_index_list[n + 1]])
		else:
			nl_prob -= math.log(edge[answer_index_list[n]][answer_index_list[n + 1]])
	return nl_prob

def divide_dataset(data_dir, max_length, test_set_interval):
	filename_list = []
	tpl_list_x2 = []
	for i, filename in enumerate(sorted(os.listdir(data_dir))):
		print('\rload_rntxt データ:', (i + 1), end='')
		temp_tpl_list_x2 = load_rntxt(os.path.join(data_dir, filename))
		for i1, tpl_list in enumerate(temp_tpl_list_x2):
			for i2 in range((len(tpl_list) // max_length) + 1):
				if len(tpl_list[i2 * max_length : (i2 + 1) * max_length]) > 1:
					filename_list.append(filename + '_' + str(i1) + '_' + str(i2 * max_length) + ':' + str((i2 + 1) * max_length))
					tpl_list_x2.append(tpl_list[i2 * max_length : (i2 + 1) * max_length])
				else:
					pass
	print('\rload_rntxt 完了 ', len(tpl_list_x2), '件                          ')
	int_list_x3 = []
	for i, tpl_list in enumerate(tpl_list_x2):
		print('\rget_chord_interpretation_list_x2 データ:', (i + 1), '/', len(tpl_list_x2), end='')
		int_list_x3.append(get_chord_interpretation_list_x2(tpl_list))
	print('\rget_chord_interpretation_list_x2 完了                       ')
	if test_set_interval > 0:
		validation_filename_list = [v for i, v in enumerate(filename_list) if i % test_set_interval == 0]
		test_filename_list = [v for i, v in enumerate(filename_list) if i % test_set_interval == 1]
		training_filename_list = [v for i, v in enumerate(filename_list) if i % test_set_interval >= 2]
		validation_tpl_list_x2 = [v for i, v in enumerate(tpl_list_x2) if i % test_set_interval == 0]
		test_tpl_list_x2 = [v for i, v in enumerate(tpl_list_x2) if i % test_set_interval == 1]
		training_tpl_list_x2 = [v for i, v in enumerate(tpl_list_x2) if i % test_set_interval >= 2]
		validation_int_list_x3 = [v for i, v in enumerate(int_list_x3) if i % test_set_interval == 0]
		test_int_list_x3 = [v for i, v in enumerate(int_list_x3) if i % test_set_interval == 1]
		training_int_list_x3 = [v for i, v in enumerate(int_list_x3) if i % test_set_interval >= 2]
	else:
		validation_filename_list = filename_list
		test_filename_list = filename_list
		training_filename_list = filename_list
		validation_tpl_list_x2 = tpl_list_x2
		test_tpl_list_x2 = tpl_list_x2
		training_tpl_list_x2 = tpl_list_x2
		validation_int_list_x3 = int_list_x3
		test_int_list_x3 = int_list_x3
		training_int_list_x3 = int_list_x3
	if False:
		print('training phrases', len(training_tpl_list_x2), 'tpls', sum([len(v) for v in training_tpl_list_x2]))
		print('validation phrases', len(validation_int_list_x3), 'tpls', sum([len(v) for v in validation_int_list_x3]))
		print('test phrases', len(test_tpl_list_x2), 'tpls', sum([len(v) for v in test_tpl_list_x2]))
	return (training_filename_list, validation_filename_list, test_filename_list, training_tpl_list_x2, validation_tpl_list_x2, test_tpl_list_x2, training_int_list_x3, validation_int_list_x3, test_int_list_x3)

#! フォルダpathを与えてwait_epochだけtrain accの更新がなくまるまで学習を続ける
def train(data_dir, max_epoch, batch_size, gtps, tensor_param, optimizer, max_length=100, b_refresh_graph=False, test_set_interval=10, wait_epoch=5, parallel_count=8, b_min_acc=True):
	ret_str = ''
	(training_filename_list, validation_filename_list, test_filename_list, training_tpl_list_x2, validation_tpl_list_x2, test_tpl_list_x2, training_int_list_x3, validation_int_list_x3, test_int_list_x3) = divide_dataset(data_dir, max_length, test_set_interval)
	# 解釈グラフを準備する
	training_graph_list = []
	validation_graph_list = []
	test_graph_list = []
	zipped = list(zip([gtps] * len(training_int_list_x3), training_int_list_x3))
	with Pool(parallel_count) as p:
		(training_graph_list) = p.map(func=make_interpretation_graph, iterable=zipped)
	zipped = list(zip([gtps] * len(validation_int_list_x3), validation_int_list_x3))
	with Pool(parallel_count) as p:
		(validation_graph_list) = p.map(func=make_interpretation_graph, iterable=zipped)
	zipped = list(zip([gtps] * len(test_int_list_x3), test_int_list_x3))
	with Pool(parallel_count) as p:
		(test_graph_list) = p.map(func=make_interpretation_graph, iterable=zipped)
	print('\rmake_interpretation_graph 完了                              ')
	# テスト集合内個別のaccuracy調査
	if False:
		acc_sum = 0.0
		len_sum = 0
		acc_len_sum = 0.0
		for i, tpl_list in enumerate(test_tpl_list_x2):
			back_link_list_x3 = get_back_link_list_x3(gtps, test_graph_list[i])
			acc = debug_show_shortest_paths(tpl_list, test_graph_list[i], back_link_list_x3)
			print('○', test_filename_list[i], 'test acc', acc, 'length', len(tpl_list))
			acc_sum += acc
			len_sum += len(tpl_list)
			acc_len_sum += acc * len(tpl_list)
		print('曲平均acc', acc_sum / len(test_tpl_list_x2), '長さ平均acc', acc_len_sum / len_sum)
		exit()
	# 学習前のaccuracyを表示
	print('param: ', tensor_param)
	gtps.update_params(tensor_param)
	acc0 = get_average_accuracy(gtps, training_tpl_list_x2, training_int_list_x3, training_graph_list, parallel_count=parallel_count, b_nl_prob=False)
	acc1 = get_average_accuracy(gtps, validation_tpl_list_x2, validation_int_list_x3, validation_graph_list, parallel_count=parallel_count, b_nl_prob=True)
	acc2 = get_average_accuracy(gtps, test_tpl_list_x2, test_int_list_x3, test_graph_list, parallel_count=parallel_count, b_nl_prob=True)
	print('\rget_average_accuracy training mean:', acc0[0], ', nl_prob:', acc0[2], ', validation mean:', acc1[0], ', nl_prob:', acc1[2], ', test mean:', acc2[0], ', nl_prob:', acc2[2], '          ')
	max_acc_mean = -sys.float_info.max # 学習前accuracyとは別にする（下がる場合もあるし）
	max_test_acc_mean = 0.0 # これのmaxというわけではないが（以下同様）
	max_test_nl_prob = 0.0
	max_acc_epoch = 0
	max_acc_tensor = 0
	ret_str = ''
	# 学習
	for epoch in range(max_epoch + 1):
		# 学習期間終了
		if (epoch > max_acc_epoch + wait_epoch) or (epoch >= max_epoch):
			ret_str = "max_test_acc: " + str(max_test_acc_mean) + " nl_prob: " + str(max_test_nl_prob) + " epoch: " + str(max_acc_epoch + 1) + "\n" + str(max_acc_tensor) + "\n" + ret_str
			return ret_str
		# とりあえずバッチ学習
		optimizer.zero_grad()
		nl_prob = 0
		#for i, int_list_x2 in enumerate(int_list_x3):
		zipped = list(zip([gtps] * len(training_tpl_list_x2), training_tpl_list_x2, training_int_list_x3, training_graph_list, [tensor_param] * len(training_tpl_list_x2)))
		random.shuffle(zipped)
		for i0 in range(math.ceil(len(training_tpl_list_x2) / batch_size)):
			with Pool(parallel_count) as p:
				(result_list) = p.map(func=train2, iterable=zipped[i0 * batch_size:i0 * batch_size + batch_size])
			(nl_prob_list, grad_list) = zip(*result_list)
			tensor_param.grad = sum(grad_list) # こうしないと呼び出し元プロセスに反映されないような感じ
			# 更新
			nl_prob = sum(nl_prob_list) / len(training_int_list_x3)
			optimizer.step()
			print('\repoch:', (epoch + 1), '-', ((i0 + 1) * batch_size), 'param:', tensor_param, ', grad:', tensor_param.grad, '                                               ')
			optimizer.zero_grad()
			# accuracy途中経過を表示（更新の度に）
			gtps.update_params(tensor_param)
			# 途中経過accuracyの計算
			if True:
				# 解釈グラフを更新する（プログラム更新していない）
				if b_refresh_graph:
					#test_graph_list = []
					#for i, int_list_x2 in enumerate(test_int_list_x3):
					#	print('\rmake_interpretation_graph データ:', (i + 1), '/', len(test_tpl_list_x2), end='')
					#	test_graph_list.append(make_interpretation_graph(gtps, int_list_x2))
					#print('\rmake_interpretation_graph 完了                              ')
					print('エラー：解釈グラフの更新は対応していません')
				if test_set_interval > 0:
					#acc0 = get_average_accuracy(gtps, training_tpl_list_x2, training_int_list_x3, training_graph_list, parallel_count=parallel_count, b_nl_prob=False)
					acc0 = [0.0, 0.0, 0.0] # 時間節約のため
					acc1 = get_average_accuracy(gtps, validation_tpl_list_x2, validation_int_list_x3, validation_graph_list, parallel_count=parallel_count, b_nl_prob=True)
					acc2 = get_average_accuracy(gtps, test_tpl_list_x2, test_int_list_x3, test_graph_list, parallel_count=parallel_count, b_nl_prob=True)
				else:
					acc0 = get_average_accuracy(gtps, training_tpl_list_x2, training_int_list_x3, training_graph_list, parallel_count=parallel_count, b_nl_prob=False)
					acc1 = acc0
					acc2 = acc0
				print('\rget_average_accuracy training mean:', acc0[0], ', nl_prob:', acc0[2], ', validation mean:', acc1[0], ', nl_prob:', acc1[2], ', test mean:', acc2[0], ', nl_prob:', acc2[2], '          ')
				ret_str += '\nepoch:' + str(epoch + 1) + '-' + str((i0 + 1) * batch_size) + '  training mean:' + str(acc0[0]) + ', nl_prob:' + str(acc0[2]) + ', validation mean:' + str(acc1[0]) + ', nl_prob:' + str(acc1[2]) + ', test mean:' + str(acc2[0]) + ', nl_prob:' + str(acc2[2])
				if b_min_acc: # 基準：validation mean
					criterion_value = acc1[0]
				else: # 基準：validation log likelihood
					criterion_value = -acc1[2]
				if max_acc_mean < criterion_value: # criterion_valueが更新されていたら
					max_acc_mean = criterion_value
					max_test_acc_mean = acc2[0]
					max_test_nl_prob = acc2[2]
					max_acc_epoch = epoch
					max_acc_tensor = tensor_param.clone()

# train() 内部の処理
def train2(arg_list):
	(gtps, tpl_list, int_list_x2, graph, tensor_param) = arg_list
	tensor_edge_list = make_computation_graph(graph, gtps, tensor_param)
	tensor_nl_prob = calc_gradient(graph, tpl_list, tensor_edge_list, True)
	nl_prob = tensor_nl_prob.item()
	tensor_nl_prob.backward()
	return nl_prob, tensor_param.grad

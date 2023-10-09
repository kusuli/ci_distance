# 2023.03.13
import os
import music21 as m21
import pathlib
import re
import datetime
import setting
from collections.abc import Iterable

org_directory = "./rntxt/20230311/org" # rntxt
dest_directory = "./rntxt/20230311/mod21_phrase"

def getChromaVector(pitches):
	ret = [0 for _ in range(12)]
	for p in pitches:
		p2 = str(p)[:-1].replace('-', 'b')
		ret[setting.NOTE_POS_DIC[p2]] = ret[setting.NOTE_POS_DIC[p2]] or 1
	#return [str(p) for p in pitches]
	return str(ret).replace('[', '').replace(']', '')

def scaleCode(mode):
	if mode.strip() == 'major':
		return str(setting.SCALE_MAJOR)
	elif mode.strip() == 'minor':
		return str(setting.SCALE_NATURAL_MINOR)
	else:
		print('error: scaleCode(', mode, ')')
		exit()

def getNoteStr(note, level):
	ret = note.figure.replace('-', 'b') + ', ' + str(note.scaleDegree) + ', ' + note.quality + ', ' + str(note.inversion()) + ', ' + str(note.root()).replace('-', 'b')
	if note.secondaryRomanNumeralKey:
		ret = str(note.secondaryRomanNumeralKey.tonic).replace('-', 'b') + ', ' + scaleCode(note.secondaryRomanNumeralKey.mode) + ', ' + ret
	else:
		ret = str(note.key.tonic).replace('-', 'b') + ', ' + scaleCode(note.key.mode) + ', ' + ret
	if note.secondaryRomanNumeral:
		ret2 = getNoteStr(note.secondaryRomanNumeral, level + 1)
		ret += ', ' + ret2[0]
		level = ret2[1]
	return (ret, level)

rn_count = 0
for file_index, filename in enumerate(os.listdir(org_directory)):
	in_file_count = 1
	arr0 = filename.split('_')
	file_index2 = arr0[0]
	print(file_index, file_index2, datetime.datetime.now(), filename)
	output_str = ''
	#if file_index < 500:
	#	continue
	p = m21.converter.parse(os.path.join(org_directory, filename))
	#p.show('text')
	for measure in p.flat.makeMeasures():
		if isinstance(measure, Iterable):
			for elm in measure:
				if isinstance(elm, m21.roman.RomanNumeral):
					#print(elm)
					temp = getNoteStr(elm, 1)
					output_str += str(measure.number) + ', ' + getChromaVector(elm.pitches) + ', ' + str(temp[1]) + ', ' + temp[0] + "\n" # measure number, beat, C, C#, D, D#, E, F, F#, G, G#, A, A#, B, count, (key tonic, key mode, figure, degree, quality, inversion, root) × count
					rn_count += 1
				elif isinstance(elm, m21.romanText.translate.RomanTextUnprocessedToken):
					new_filename = pathlib.PurePath(os.path.join(org_directory, str(file_index2))).stem + '_' + '{:02d}'.format(in_file_count) + '.txt'
					#with open(os.path.join(dest_directory, new_filename), "w", encoding='UTF-8') as f:
					#	f.write(output_str)
					in_file_count += 1
					output_str = ''
	if len(output_str) > 0:
		new_filename = pathlib.PurePath(os.path.join(org_directory, str(file_index2))).stem + '_' + '{:02d}'.format(in_file_count) + '.txt'
		#with open(os.path.join(dest_directory, new_filename), "w", encoding='UTF-8') as f:
		#	f.write(output_str)
print('Note count:', rn_count)
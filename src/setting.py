# 2022.12.11

CHORD_X = 'X'
CHORD_N = 'N'
	
NOISE_SD = 0.000001 # ノイズの標準偏差

#NOTE_POS_DIC = {'A': 0, 'A#': 1, 'Bb': 1, 'B': 2, 'Cb': 2, 'C': 3, 'B#': 3, 'C#': 4, 'Db': 4, 'D': 5, 'D#': 6, 'Eb': 6, 'E': 7, 'Fb': 7, 'F': 8, 'E#': 8, 'F#': 9, 'Gb': 9, 'G': 10, 'G#': 11, 'Ab': 11}
NOTE_POS_DIC = {
	'C###' : 3, 'D###' : 5, 'E###' : 7, 'F###' : 8, 'G###' : 10, 'A###' : 0, 'B###' : 2,
	'Cbb' : 10, 'Dbb' : 0, 'Ebb' : 2, 'Fbb' : 3, 'Gbb' : 5, 'Abb' : 7, 'Bbb' : 9,
	'C##' : 2, 'D##' : 4, 'E##' : 6, 'F##' : 7, 'G##' : 9, 'A##' : 11, 'B##' : 1,
	'Cb' : 11, 'Db' : 1, 'Eb' : 3, 'Fb' : 4, 'Gb' : 6, 'Ab' : 8, 'Bb' : 10,
	'C#' : 1, 'D#' : 3, 'E#' : 5, 'F#' : 6, 'G#' : 8, 'A#' : 10, 'B#' : 0,
	'C' : 0, 'D' : 2, 'E' : 4, 'F' : 5, 'G' : 7, 'A' : 9, 'B' : 11}

SCALE_MAJOR = 1
SCALE_NATURAL_MINOR = 0
SCALE_HARMONIC_MINOR = 2
SCALE_MELODIC_MINOR = 3

SCALE_DISTANCE_DIC = {
	SCALE_MAJOR:          [0, 2, 4, 5, 7, 9, 11],
	SCALE_NATURAL_MINOR:  [0, 2, 3, 5, 7, 8, 10],
	SCALE_HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
	SCALE_MELODIC_MINOR:  [0, 2, 3, 5, 7, 9, 11]
} # degreeは1始まりだが、リストは0始まりなので注意

CHORD_FUNCTION_T = 1
CHORD_FUNCTION_SD = 2
CHORD_FUNCTION_SDM = 3
CHORD_FUNCTION_D = 4
CHORD_FUNCTION_NONE = 0

CHORD_TYPE_MAJ = 1
CHORD_TYPE_7 = 2
CHORD_TYPE_MAJ7 = 3
CHORD_TYPE_MAJ6 = 4
CHORD_TYPE_MIN = 5
CHORD_TYPE_MIN7 = 6
CHORD_TYPE_MINMAJ7 = 7
CHORD_TYPE_MIN6 = 8
CHORD_TYPE_MIN7_FLAT5 = 9
CHORD_TYPE_DIM7 = 10
CHORD_TYPE_COUNT = 10

CHORD_QUALITY_OTHER = 0
CHORD_QUALITY_MAJ = 1
CHORD_QUALITY_MIN = 2
CHORD_QUALITY_DIM = 3
CHORD_QUALITY_AUG = 4

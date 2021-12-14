
import numpy as np
import re


def is_uchar(uchar):
	if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
		return True
	if uchar >= u'\u0030' and uchar <= u'\u0039':
		return True
	if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
		return True
	if uchar in ('，', '。', '：', '？', '“', '”', '！', '；', '、', '《', '》', '——'):
		return True
	return False
class Preprocessing:
	@staticmethod
	def read_dataset(file):
		# Open raw file
		with open(file, 'r',encoding='gbk') as f:
			 data = f.readlines()
		pattern = re.compile(r'\(.*\)')
		data = [pattern.sub('', lines) for lines in data]
		data = [line.replace('……', '。') for line in data if len(line) > 1]

		data = ''.join(data)
		data = [char for char in data if is_uchar(char)]
		data = ''.join(data)
		return data
		
	@staticmethod
	def create_dictionary(data):
		text = set(data)
		char_to_idx = dict()
		idx_to_char = dict()
		
		idx = 0
		for char in text:
			if char not in char_to_idx.keys():
				char_to_idx[char] = idx
				idx_to_char[idx] = char
				idx += 1
				
		print("Vocab: ", len(char_to_idx))
		
		return char_to_idx, idx_to_char
		
	@staticmethod
	def build_sequences_target(text, char_to_idx, window):
		
		x = list()
		y = list()
	
		for i in range(len(text)):
			try:
				# Get window of chars from text
				# Then, transform it into its idx representation
				sequence = text[i:i+window]
				sequence = [char_to_idx[char] for char in sequence]
				
				# Get char target
				# Then, transfrom it into its idx representation
				target = text[i+window]
				target = char_to_idx[target]
				
				# Save sequences and targets
				x.append(sequence)
				y.append(target)
			except:
				pass
		
		x = np.array(x)
		y = np.array(y)
		
		return x, y
		

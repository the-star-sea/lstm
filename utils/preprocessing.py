import numpy as np
import re

import torch
from torch.utils.data import DataLoader,Dataset



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



class Preprocessing(Dataset):
    @staticmethod
    def read_dataset(file):
        # Open raw file
        with open(file, 'r', encoding='utf-8') as f:
            data = f.readlines()

        pattern = re.compile(r'\(.*\)')
        data = [pattern.sub('', lines) for lines in data]
        data = [line.replace('……', '。') for line in data if len(line) > 1]

        data = ''.join(data)
        data = [char for char in data if is_uchar(char)]
        data = ''.join(data)
        # data = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", data)
        # data = jieba.cut(data)
        # data = [i for i in data if i[0] != '第' and i[-1] != '章']
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
                sequence = text[i:i + window]
                sequence = [char_to_idx[char] for char in sequence]

                target = text[i + window]
                target = char_to_idx[target]
                if (len(sequence) == window ):
                    # Save sequences and targets
                    x.append(sequence)
                    y.append(target)
            except:
                pass

        x = np.array(x)
        y = np.array(y)
        # print('x:',x.shape)
        # print('y:',y.shape)
        return x, y

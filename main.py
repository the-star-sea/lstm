import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from src import TextGenerator
from utils import Preprocessing
from utils import parameter_parser
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
class Execution:

	def __init__(self, args):
		self.file = 'data/jing.txt'
		self.window = args.window
		self.batch_size = args.batch_size
		self.learning_rate = args.learning_rate
		self.num_epochs = args.num_epochs
		
		self.targets = None
		self.sequences = None
		self.vocab_size = None
		self.char_to_idx = None
		self.idx_to_char = None

	def prepare_data(self):
	
		# Initialize preprocessor object
		preprocessing = Preprocessing()
		
		# The 'file' is loaded and split by char
		text = preprocessing.read_dataset(self.file)


		self.char_to_idx, self.idx_to_char = preprocessing.create_dictionary(text)
		
		# Given the 'window', it is created the set of training sentences as well as
		# the set of target chars
		self.sequences, self.targets = preprocessing.build_sequences_target(text, self.char_to_idx, window=self.window)
			
		# Gets the vocabuly size
		self.vocab_size = len(self.char_to_idx)


	def train(self, args,clip=5):
	
		# Model initialization
		model = TextGenerator(args, self.vocab_size).to('cuda')
		# Optimizer initialization
		optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
		# Defining number of batches
		num_batches = int(len(self.sequences) / self.batch_size)
		# Set model in training mode
		model.train()

		for epoch in range(self.num_epochs):
			hidden=None
			for i in range(num_batches):
				optimizer.zero_grad()
				# try:
				# 	x_bs = self.sequences[i * self.batch_size : (i + 1) * self.batch_size]
				# 	y_bs = self.targets[i * self.batch_size : (i + 1) * self.batch_size]
				# except:
				# 	x_bs = self.sequences[i * self.batch_size :]
				# 	y_bs = self.targets[i * self.batch_size :]
				x_bs = []
				y_bs = []
				for j in range(self.batch_size):
					try:
						x_bs.append(self.sequences[j* num_batches+i])
						y_bs.append(self.targets[j * num_batches+i])
					except:
						pass
				x_bs=np.array(x_bs)
				y_bs=np.array(y_bs)
				# Convert numpy array into torch tensors
				x = torch.from_numpy(x_bs).type(torch.LongTensor).to('cuda')
				y = torch.from_numpy(y_bs).type(torch.LongTensor).to('cuda')

				y_pred,hidden = model(x,hidden=hidden)
				hidden = tuple([Variable(each.data) for each in hidden])
				# y_pred=y_pred.permute(1,0,2)
				# y_pred=y_pred.view(self.batch_size*self.window,-1)
				# y=y.squeeze().view(-1)

				loss = F.cross_entropy(y_pred,y )
				# Clean gradients

				# Calculate gradientes
				loss.backward()
				nn.utils.clip_grad_norm(model.parameters(), clip)
				optimizer.step()

			print("Epoch: %d,  loss: %.5f " % (epoch, loss.item()))
			scheduler.step()
			torch.save(model.state_dict(), f'weights/jing-{epoch}.pt')
	
	@staticmethod
	def generator(model, sequences, idx_to_char, n_chars):
		
		# Set the model in evalulation mode
		model.eval()

		# Define the softmax function
		softmax = nn.Softmax(dim=1)
		
		# Randomly is selected the index from the set of sequences
		# start = np.random.randint(0, len(sequences)-1)
		start=0
		patterns = sequences[start:start+10]
		hidden=None
		# By making use of the dictionaries, it is printed the pattern
		print("\nPattern: \n")
		print(''.join([idx_to_char[value] for value in patterns[9]]), "\"")
		pattern=None
		# In full_prediction we will save the complete prediction
		for patter in patterns:

			pattern = torch.from_numpy(patter).type(torch.LongTensor)

			pattern = pattern.view(1, -1).to('cuda')
			# Make a prediction given the pattern
			_, hidden = model(pattern, hidden=hidden, predict=True)
		full_prediction = patterns[9].copy()
		
		# The prediction starts, it is going to be predicted a given
		pattern = patterns[9]
		for i in range(n_chars):
			pattern = torch.from_numpy(pattern).type(torch.LongTensor)
			pattern = pattern.view(1, -1).to('cuda')
			prediction, hidden = model(pattern, hidden=hidden, predict=True)
			# It is applied the softmax function to the predicted tensor
			prediction = softmax(prediction)
			
			# The prediction tensor is transformed into a numpy array
			prediction = prediction.squeeze().detach().cpu().numpy()
			# _,index=torch.topk(prediction, 5,  largest=True, sorted=False, out=None)
			# index=index.cpu().numpy()
			arg_max=np.random.choice(len(prediction),p=prediction)
			# It is taken the idx with the highest probability
			# arg_max = np.argmax(prediction)
			
			# The current pattern tensor is transformed into numpy array
			pattern = pattern.squeeze().detach().cpu().numpy()
			# The window is sliced 1 character to the right
			pattern = pattern[1:]
			# The new pattern is composed by the "old" pattern + the predicted character
			pattern = np.append(pattern, arg_max)
			
			# The full prediction is saved
			full_prediction = np.append(full_prediction, arg_max)
			
		print("Prediction: \n")
		print(''.join([idx_to_char[value] for value in full_prediction]), "\"")

if __name__ == '__main__':
	
	args = parameter_parser()

	# If you already have the trained weights
	if args.load_model == True:
		if os.path.exists(args.model):

			# Load and prepare sequences
			execution = Execution(args)
			execution.prepare_data()

			sequences = execution.sequences
			idx_to_char = execution.idx_to_char
			vocab_size = execution.vocab_size

			# Initialize the model
			model = TextGenerator(args, vocab_size,predict=True).to('cuda')
			# Load weights
			model.load_state_dict(torch.load(args.model, map_location='cuda'))

			# Text generator
			execution.generator(model, sequences, idx_to_char, 1000)

	# If you will train the model
	else:
		# Load and preprare the sequences
		execution = Execution(args)
		execution.prepare_data()
		execution.train(args)

		sequences = execution.sequences
		idx_to_char = execution.idx_to_char
		vocab_size = execution.vocab_size

		# Initialize the model
		model = TextGenerator(args, vocab_size,predict=True).to('cuda')

		model.load_state_dict(torch.load('weights/jing-500.pt', map_location='cuda'))

		# Text generator
		execution.generator(model, sequences, idx_to_char, 1000)

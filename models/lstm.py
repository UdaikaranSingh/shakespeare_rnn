import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class LSTM(nn.Module):

	def __init__(self, input_size, hidden_size, output_size, device):
		super(LSTM,self).__init__()
		
		#variable storage
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.device = device

		#architecture
		self.hidden_layer = nn.LSTMCell(input_size = self.input_size,
			hidden_size = self.hidden_size)
		self.fully_connected = nn.Linear(in_features = self.hidden_size,
			out_features = self.output_size)

	def forward(self, input, hidden):
		
		hidden, _ = self.hidden_layer(input)
		out = self.fully_connected(hidden)

		return out, hidden

	def init_hidden(self):

		return torch.zeros(1, self.hidden_size).to(self.device)
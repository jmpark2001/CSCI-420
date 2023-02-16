"""
define moduals of model
"""
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch


class CNNModel(nn.Module):
	"""docstring for ClassName"""
	
	def __init__(self, args):
		super(CNNModel, self).__init__()
		##--------------------------------------------------------
		## define the model architecture here
		## image input size batch * 28 * 28
		##--------------------------------------------------------
		
		## define CNN layers below
		#self.conv = nn.sequential( 	# nn.Conv2d(in_channels,...),
									# activation fun,
									# dropout,
									# nn.Conv2d(in_channels,...),
									# activation fun,
									# dropout,
									## continue like above,
									## **define pooling (bonus)**,
		self.conv = nn.Sequential(nn.Conv2d(1,args.channel_out1, kernel_size=args.k_size, stride=args.stride),
									nn.ReLU(),
									nn.Dropout(args.dropout),
									# second layer
									nn.Conv2d(args.channel_out1, args.channel_out2, kernel_size=args.k_size,stride=args.stride),
									nn.ReLU(),
									nn.Dropout(args.dropout),
									# third layer
									nn.Conv2d(args.channel_out2, 10, kernel_size= args.k_size, stride= args.stride),
									nn.ReLU(),
									nn.Dropout(args.dropout)
									## continue like above,
									## **define pooling (bonus)**,
								)

		##------------------------------------
		## define fully connected layer below
		##------------------------------------
		
		# 10 * 19 * 19, number of classes

		self.fc = nn.Linear(3610, 10)
		

	'''feed features to the model'''
	def forward(self, x):
		##---------------------------------------------------
		## feed input features to the CNN models defined above
		##---------------------------------------------------
		x_out = self.conv(x)

		## write flatten tensor code below (it is done)
		x_out = torch.flatten(x_out,1) # x_in is output of last layer
		
		## ---------------------------------------------------
		## write fully connected layer (Linear layer) below
		## ---------------------------------------------------
		result = self.fc(x_out)
		
		
		return result
        
		
		
	
		
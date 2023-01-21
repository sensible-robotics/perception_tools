from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

from lenet5.lenet5_class import LeNet5
from lenet5.helpers import training_loop

###############################
## Parameters
N_CLASSES = 5

# Check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


model = LeNet5(N_CLASSES).to(DEVICE)
example = torch.rand([1, 1,32, 32], dtype=torch.float32, device= DEVICE).unsqueeze(0)

traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("lenet_script.pt")



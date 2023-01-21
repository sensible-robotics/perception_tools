import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from lenet5.lenet5_class import LeNet5


# Load the model
N_CLASSES = 10
# Check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

root_path = "/home/ana/Research/cnn/perception_tools/lenet"

model = LeNet5(N_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(root_path + "/lenet_model.pt"))

# Get ready to eval
model.eval()


example = torch.zeros([1, 32, 32], dtype=torch.float32, device= DEVICE)
for x in range(5,25):
  example[0][x][15] = 1.0




example_dev = example.unsqueeze(0).to(DEVICE)
traced_script_module = torch.jit.trace(model, example_dev)
traced_script_module.save(root_path + "/lenet_script.pt")


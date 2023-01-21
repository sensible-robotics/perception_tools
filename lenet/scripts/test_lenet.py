import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from lenet5.lenet5_class import LeNet5

root_path = "/home/ana/Research/cnn/perception_tools/lenet"

# Load the model
N_CLASSES = 10

# Check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LeNet5(N_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(root_path + "/lenet_model.pt"))

# Get ready to eval
model.eval()

# Evaluate
ROW_IMG = 10
N_ROWS = 5


# Load datasets for validation
# transforms.ToTensor() automatically scales the images to [0,1] range
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])
valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms)

# Figure
fig = plt.figure()



for index in range(1, ROW_IMG * N_ROWS + 1):
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis('off')
    plt.imshow(valid_dataset.data[index], cmap='gray_r')
    
    with torch.no_grad():
        input_val = ( valid_dataset[index][0].unsqueeze(0) ).to(DEVICE)
        model.eval()
        _, probs = model(input_val)
        
    title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'    
    plt.title(title, fontsize=7)

fig.suptitle('LeNet-5 - predictions');
plt.show()


# Test
print("Run test ------------------")
example = torch.zeros([1, 32, 32], dtype=torch.float32, device= DEVICE)
for x in range(5,25):
  example[0][x][15] = 1.0

example_dev = example.unsqueeze(0).to(DEVICE)
print(type(example_dev))
print(example_dev.size())
_, prob_i = model(example_dev)
print("Prob i --------------")
print(prob_i)
print("OK -----------------")



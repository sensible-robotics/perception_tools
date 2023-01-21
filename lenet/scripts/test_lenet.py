import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from lenet5.lenet5_class import LeNet5


# Load the model
N_CLASSES = 10
model = LeNet5(N_CLASSES)
model.load_state_dict(torch.load("ana_lenet.pt"))

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

print("Val and type")
print(valid_dataset[0][0])
print("Size of vd[n][0]")
print( (valid_dataset[0][0]).size() )
print("type and size of unsqueezed :")
print(type(valid_dataset[0][0].unsqueeze(0)) )
print((valid_dataset[0][0].unsqueeze(0)).size() )

print("Unsqueezed")
print(valid_dataset[10][0].dtype)
print(valid_dataset[10][0])

for index in range(1, ROW_IMG * N_ROWS + 1):
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis('off')
    plt.imshow(valid_dataset.data[index], cmap='gray_r')
    
    with torch.no_grad():
        model.eval()
        _, probs = model(valid_dataset[index][0].unsqueeze(0))
        
    title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'    
    plt.title(title, fontsize=7)

fig.suptitle('LeNet-5 - predictions');
plt.show()




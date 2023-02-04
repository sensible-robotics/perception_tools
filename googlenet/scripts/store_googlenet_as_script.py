import torch
import os

from PIL import Image
from torchvision import transforms

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

# Set location of current script
ABSOLUTE_PATH = os.path.dirname(__file__)
RELATIVE_PATH = "../"
ROOT_PATH = os.path.join(ABSOLUTE_PATH, RELATIVE_PATH)

# Load model
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights='GoogLeNet_Weights.DEFAULT').to(DEVICE)

# Get ready to eval
model.eval()

# Run an example input through the model
filename = os.path.join(ROOT_PATH, "sample_data/un_primo_bonny_224.jpg")
input_image = Image.open(filename)
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485,0.456, 0.406],
                                                      std=[0.229,0.224, 0.225]) ])
input_tensor = preprocess(input_image)
# Create a minibatch
example_batch = input_tensor.unsqueeze(0)

example_dev = example_batch.to(DEVICE)
traced_script_module = torch.jit.trace(model, example_dev)
traced_script_module.save(os.path.join(ROOT_PATH, "googlenet_script.pt"))

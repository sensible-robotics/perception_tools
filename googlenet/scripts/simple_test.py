import torch
import urllib

from PIL import Image
from torchvision import transforms


print("Load model")
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
model.eval()
	
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")

print("Url download image")

try:
  urllib.URLopener().retrieve(url, filename)
except:
  urllib.request.urlretrieve(url, filename)


filename = "sample_data/un_primo_bonny_224.jpg"  
#filename = "dog.jpg"
input_image = Image.open(filename)
preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])	

input_tensor = preprocess(input_image)
# Create a minibatch
input_batch = input_tensor.unsqueeze(0)

# Move the input and model to GPU
if torch.cuda.is_available():
  input_batch = input_batch.to('cuda')
  model.to('cuda')
print("Size of input batch:" )
print(input_batch.dim())
print(input_batch.size())
  
with torch.no_grad():
  output = model(input_batch)
  
print("Output of 1000 classification results")
print(output[0])

probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

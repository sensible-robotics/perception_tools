import torch
import os

from PIL import Image
from torchvision import transforms

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
ABSOLUTE_PATH = os.path.dirname(__file__)
RELATIVE_PATH = "../"
ROOT_PATH = os.path.join(ABSOLUTE_PATH, RELATIVE_PATH)

class ResnetStore:

    def __init__(self):
        print('Init ResnetStore')

    def initialize(self):
        # Load model
        self.model_ = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT').to(DEVICE)

        # Get ready to eval
        self.model_.eval()

        # Read the categories
        with open("imagenet_classes.txt", "r") as f:
            self.categories_ = [s.strip() for s in f.readlines()]


    def get_script(self):
        
        # Run an example input through the model
        filename = os.path.join(ROOT_PATH, "sample_data/gallinita_1.jpg")
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
        traced_script_module = torch.jit.trace(self.model_, example_dev)
        traced_script_module.save(os.path.join(ROOT_PATH, "resnet_script.pt"))

    def test(self, image):
        filename = os.path.join(ROOT_PATH, image)
        input_image = Image.open(filename)
        preprocess = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485,0.456, 0.406],
                                                              std=[0.229,0.224, 0.225]) ])
        input_tensor = preprocess(input_image)
        # Create a minibatch
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(DEVICE)

        with torch.no_grad():
            output = self.model_(input_batch)
        
        probs = torch.nn.functional.softmax(output[0], dim=0)

        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probs, 5)
        for i in range(top5_prob.size(0)):
            print(self.categories_[top5_catid[i]], top5_prob[i].item())


#####################################
def main():
    print("Store script for Resnet")
    
    rs = ResnetStore()
    rs.initialize()
    rs.get_script()

    print("Testing Resnet 1...")
    rs.test("sample_data/un_primo_bonny_224.jpg")
    print("Testing Resnet 2...")
    rs.test("sample_data/gallinita_1.jpg")
    print("Testing Resnet 3...")
    rs.test("sample_data/dog.jpg")
    print("Testing Resnet 4...")
    rs.test("sample_data/kite_2.jpg")
    
#####################################
if __name__ == "__main__":
    main()

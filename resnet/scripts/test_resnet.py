import torch

from PIL import Image
from torchvision import transforms

class ResnetTest:

    def __init__(self):
        print("Init ResnetTest...")

    def initialize(self):
        print("Load model")
        self.model_ = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.IMAGENET1K_V1')
        self.model_.eval()
        
        # Move model to GPU
        if torch.cuda.is_available():
            self.model_.to('cuda')
            
        
        print("Read categories")
        with open("imagenet_classes.txt", "r") as f:
            self.categories_ = [s.strip() for s in f.readlines()]
        
        
    def preprocess_image(self, _filename):
        input_image = Image.open(_filename)
        preprocess = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]) ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        return input_batch
        
    def evaluate(self, _filename):
        input_batch = self.preprocess_image(_filename)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')


        # Evaluate
        with torch.no_grad():
            output = self.model_(input_batch)

        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probs, 5)
        for i in range(top5_prob.size(0)):
            print(self.categories_[top5_catid[i]], top5_prob[i].item())
        


################################
def main():
    print("Testing RESNet")

    filename="sample_data/un_primo_bonny_224.jpg"

    rt = ResnetTest()
    rt.initialize()
    rt.evaluate(filename)


    
################################    
if __name__ == "__main__":
    main()

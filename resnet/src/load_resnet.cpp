#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[])
{
  torch::jit::script::Module model;
  try {
    model = torch::jit::load("/home/ana/Research/perception_tools/resnet/resnet_script.pt");
  } catch (const c10::Error& e) {
    std::cerr << "Error loading the model \n";
    return -1;
  }
  
  std::cout << "Oh yes! "<< std::endl;
  return 0;
}

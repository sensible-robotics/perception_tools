#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, char* argv[])
{
  torch::jit::script::Module model;
  std::string root_path = "/home/ana/Research/cnn/perception_tools/googlenet";

  try {
    model = torch::jit::load(root_path + "/googlenet_script.pt");
  } catch (const c10::Error& e)
  {
    std::cerr << "Error loading the model \n";
    return -1;
  }

  std::cout << "Oh yes!" << std::endl;
  return 0;
}

#include <torch/script.h>
#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>

int main(int argc, const char* argv[])
{
  std::string root_path = std::string("/home/ana/Research/perception_tools/lenet");

  torch::jit::script::Module model;
  try {
    model = torch::jit::load(root_path + "/lenet_script.pt");
  } catch (const c10::Error& e) {
    std::cerr << "Error loading the model \n";
    return -1;
  }
  
  std::cout << "Loaded. Oh yes! "<< std::endl;

  // Test
  std::string image_path = root_path + std::string("/sample_data/02_1.png");
  cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  printf("Loaded image of size: %d and %d \n", img.rows, img.cols);

printf("Model parameters \n");
for (const auto& p : model.parameters()) {
    std::cout << p << std::endl;
  }
printf("Now inputs \n");
  std::vector<c10::IValue> inputs;
  inputs.push_back( torch::randn({1, 32, 32, torch::dtype(torch::kFloat32).device(torch::kCUDA, 1).requires_grad(false)));
  /*
  int index = 0;
  for(int i = 0; i < img.rows; ++i)
  {
    for(int j = 0; j < img.cols; ++j)
    {
       int num = img.at<uchar>(j,i);
       double val = (double)num/255.0;
       inputs[0][index] = torch::IValue(val);
       index++;
    }
  }
*/

  // Execute the model and turn its output into a tensor.
  at::Tensor output = model.forward( inputs ).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/10) << '\n';



  return 0;
}

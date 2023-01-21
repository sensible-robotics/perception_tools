#include <torch/script.h>
#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>

  std::string root_path = "/home/ana/Research/cnn/perception_tools/lenet";


at::Tensor getTensorFromImg(std::string _subpath)
{
 at::Tensor data = torch::zeros({1, 1, 32, 32}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

  std::string image_path = root_path + _subpath;
  cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  printf("Loaded image %s of size: %d and %d \n", _subpath.c_str(), img.rows, img.cols);

for(int i = 0; i < img.rows; ++i)
  {
    for(int j = 0; j < img.cols; ++j)
    {
       int num = 255 - img.at<uchar>(i,j);
       double val = (double)num/255.0;
       printf("%d ", num);
       data[0][0][i][j] = val;
    } printf("\n");
  }

  return data;
}



int main(int argc, const char* argv[])
{
  
  torch::jit::script::Module model;
  try {
    model = torch::jit::load(root_path + "/lenet_script.pt");
  } catch (const c10::Error& e) {
    std::cerr << "Error loading the model \n";
    return -1;
  }
  
  std::cout << "Loaded. Oh yes! "<< std::endl;

at::Tensor d_2, d_3, d_6;
d_2 = getTensorFromImg(std::string("/sample_data/02_1.png"));
d_3 = getTensorFromImg(std::string("/sample_data/03_1.png"));
d_6 = getTensorFromImg(std::string("/sample_data/06_1.png"));


std::vector<torch::jit::IValue> inputs;
inputs.push_back(d_6);
//inputs.push_back(d_3);

  // Execute the model and turn its output into a tensor.
  printf("Forward model \n");
  //at::Tensor output = 
  auto output_tuple = model.forward( inputs ).toTuple();

  
  double max_val = 0;
  int max_ind = -1;
  at::Tensor output = output_tuple->elements()[1].toTensor();
  for(int i = 0; i < 10; ++i)
  {
     double vali;
     vali = output[0][i].item<double>();
     if(vali > max_val)
     {  max_val = vali;
       max_ind = i;
       }
  }
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/10) << '\n';
  std::cout << "Max prob (" << max_val << ") is for value " << max_ind << std::endl;


  return 0;
}

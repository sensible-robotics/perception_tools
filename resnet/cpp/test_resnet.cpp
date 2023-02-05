#include <torch/script.h>
#include <torch/nn/functional/activation.h>

#include <iostream>
#include <memory>
#include <fstream>

#include <opencv2/opencv.hpp>

std::string root_path = "/home/ana/Research/cnn/perception_tools/resnet";

class TestResnet
{
public:
  
  TestResnet()
  {}

  /**
   * @function init
   */
  bool init()
  {
    try {
      model_ = torch::jit::load(root_path + "/resnet_script.pt");
    } catch (const c10::Error& e) {
      std::cerr << "Error loading the model \n";
      return false;
    } // catch

    // Get labels
    labels_ = getLabels();
    
    return !labels_.empty();
  }

  std::vector<std::string> getLabels()
  {
    std::vector<std::string> res;
  
    std::string label_path = root_path + "/imagenet_classes.txt";
    std::fstream input;
 
    input.open(label_path, std::ios::in);
    if(input.is_open())
    {
      std::string data;
      while(std::getline(input, data))
	res.push_back(data);
      input.close();
    }

    return res;
  }

  /**
   * @function getTensorFromImg
   */
  at::Tensor getTensorFromImg(std::string _subpath)
  {
    std::string image_path = root_path + _subpath;
    cv::Mat img1 = cv::imread(image_path, cv::IMREAD_COLOR);
    printf("Loaded image %s of size: %d and %d \n", _subpath.c_str(), img1.rows, img1.cols);
    
    int rows = img1.rows;
    int cols = img1.cols;
    int resized_rows, resized_cols;
    if(rows<cols)
    {
      resized_rows = 224;
      resized_cols = ceil( (double)(cols*224)/(double)rows);
    } else
    {
      resized_cols = 224;
      resized_rows = ceil( (double)(rows*224)/(double)cols);
    }

    cv::Mat img1_res(resized_rows, resized_cols, CV_8UC3);
    cv::resize(img1, img1_res, img1_res.size(), 0, 0);

    
    // Crop
    int row_start = (resized_rows - 224) / 2;
    int col_start = (resized_cols - 224) / 2;
    cv::Mat img = img1_res(cv::Range(row_start, row_start + 224), cv::Range(col_start, col_start + 224) );

    cv::imshow("Display small image", img);
    int k = cv::waitKey(3000);

    
    printf("Img size: %d and %d \n", img.rows, img.cols);
        
    float mean[3] ={0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    
    cv::Mat img2(img.rows, img.cols, CV_32FC3);
    for(size_t i = 0; i < img.rows; i++)
    {
      for(size_t j = 0; j < img.cols; j++)
      {
	cv::Vec3b mi = img.at<cv::Vec3b>(i,j);
	cv::Vec3f oi;
	// OpenCV stores image as BGR rather than RGB
	// hence the 2-k to reorder
	for(size_t k = 0; k < 3; k++)
	  oi.val[k] = ( (float)(mi.val[2-k]) - 255.0*mean[k] ) / 255.0*std[k];
	
	img2.at<cv::Vec3f>(i,j) = oi;
      }
    }
    
    at::Tensor data = torch::zeros({1, 3, 224, 224},
				   torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
        
    for(int i = 0; i < img2.rows; ++i)
    {
      for(int j = 0; j < img2.cols; ++j)
      {
	cv::Vec3f pi = img2.at<cv::Vec3f>(i, j);
	for(int k = 0; k < 3; ++k)
	  data[0][k][i][j] = pi[k];
      }
    }
    
    return data;
  }
  
  /**
   * @function evaluate
   */
  void evaluate(std::string _img_relative_name)
  {
    at::Tensor image = getTensorFromImg(_img_relative_name);
  
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image);
  
    // Execute the model and turn its output into a tensor.
    clock_t ts, tf; double dt;
    ts = clock();
    at::Tensor output = model_.forward( inputs ).toTensor();
    tf = clock();
    dt = (double)(tf-ts)/(double)CLOCKS_PER_SEC;
    printf("Dt for running model: %f \n", dt);

    at::Tensor output_soft = torch::nn::functional::softmax(output, torch::nn::functional::SoftmaxFuncOptions(1));
    
    float min_val, max_val;
    float min_ind, max_ind;
    //printf("Interpreting results. Output size: %d x %d \n", output.size(0), output.size(1));
    for(size_t im = 0; im < output.size(0); im++)
    {
      for(int i = 0; i < output.size(1); i++)
      {
	float val = output[im][i].item<float>();

	if(i == 0)
        {
	  max_val = val; max_ind = i;
	  min_val = val; min_ind = i;
	}
      
	if(val > max_val)
        {
	  max_val = val;
	  max_ind = i;
	}
	if(val < min_val)
	{
	  min_val = val;
	  min_ind = i;
	}
      }
      printf("Maximum class[%s]= %f \n Minimum class[%s]: %f \n",
	     labels_[max_ind].c_str(), max_val,
	     labels_[min_ind].c_str(), min_val);
    } // for im
    
  } // evaluate
  
protected:
  torch::jit::script::Module model_;
  std::vector<std::string> labels_;
};


////////////////////////////////////////
int main(int argc, const char* argv[])
{  
  TestResnet tr;
  if(!tr.init())
    return 1;
  
  tr.evaluate( std::string("/sample_data/dog.jpg") );
  tr.evaluate( std::string("/sample_data/gallinita_1.jpg") );
  tr.evaluate( std::string("/sample_data/un_primo_bonny_224.jpg") );
  tr.evaluate( std::string("/sample_data/kite_1.jpg") );
  tr.evaluate( std::string("/sample_data/kite_2.jpg") );
  tr.evaluate( std::string("/sample_data/un_primo_bonny.jpg") );


  return 0;
}

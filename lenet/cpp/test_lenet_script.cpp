#include <torch/script.h>
#include <iostream>
//#include <memory>


#include <regex>

// it's unnecessary to invoke this function, just enforce library compiled
void dummy() {
    std::regex regstr("Why");
    std::string s = "Why crashed";
    std::regex_search(s, regstr);
}


int main(int argc, const char* argv[])
{
 printf("Create module \n");
 //torch::jit::script::Module model;
 std::string pt_file = std::string("/home/ana/Research/cnn/perception_tools/lenet/ana_lenet_script.pt");
 printf("About to try \n");
 try {
   printf("Loading model....\n");
   auto model = torch::jit::load(pt_file);
 } catch (const c10::Error &e) {
    printf("Error loading the model \n");
    return -1;
 }

  std::cout << "OK" << std::endl;
}


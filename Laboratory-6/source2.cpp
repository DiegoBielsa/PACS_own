#define cimg_use_jpeg
#include <iostream>
#include "CImg.h"
using namespace cimg_library;


int main(){
  CImg<unsigned char> img("image.jpg");  // Load image file "image.jpg" at object img

  std::cout << "Image width: " << img.width() << "Image height: " << img.height() << "Number of slices: " << img.depth() << "Number of channels: " << img.spectrum() << std::endl;  //dump some characteristics of the loaded image
  for (int i = 0; i < img.width(); i++) {
	img(i, img.height() / 2, 0, 0) = 0;
	img(i, img.height() / 2, 0, 1) = 0;
	img(i, img.height() / 2, 0, 2) = 200;
  }

  int i = 53;
  int j = img.height() / 2;
  std::cout << std::hex << (int) img(i, j, 0, 0) << std::endl;  //print pixel value for channel 0 (red) 
  std::cout << std::hex << (int) img(i, j, 0, 1) << std::endl;  //print pixel value for channel 1 (green) 
  std::cout << std::hex << (int) img(i, j, 0, 2) << std::endl;  //print pixel value for channel 2 (blue) 
  
  
  img.display("My first CImg code");             // Display the image in a display window
	
	

  return 0;

}
//g++ source2.cpp -o executable -I "CImg-2" -lm -lpthread -lX11 -ljpeg
//./executable

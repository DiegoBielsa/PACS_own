__kernel void image_flip(
  __global uchar3* image,
  const unsigned int width,
  const unsigned int height){

  int x = get_global_id(0);
  int half_width = width / 2;
  
  int row = (x) / width;
  int pos = (x ) % width;
  uchar3 myColor = {0, 0, 0};
  
  if (x > width * height - 100000) {
    printf(" %i;;", x);
    image[x] =myColor;
  }
/*
  if(x < (width * 20) - 1 && pos < half_width) {
    int mirror_index = row * width + (width - 1 - pos);
    uchar3 temp = image[x];
    image[x] = image[mirror_index];
    image[mirror_index] = temp;
    
  }*/
}
__kernel void image_flip(
  __global unsigned char* image,
  const unsigned int width,
  const unsigned int height){

  int x = get_global_id(0);
  int half_width = width / 2;
  
  int row = (x) / width;
  int pos = (x) % width;
  uchar3 myColor = {0, 0, 0};
  
  if(x < (width * height) - 1 && pos < half_width) {
    int mirror_index = row * width + (width - 1 - pos);
    uchar3 temp = {image[x * 3], image[x * 3 + 1], image[x * 3 + 2]};

    image[x * 3] = image[mirror_index * 3];
    image[x * 3 + 1] = image[mirror_index * 3 + 1];
    image[x * 3 + 2] = image[mirror_index * 3 + 2];

    image[mirror_index * 3] = temp.x;
    image[mirror_index * 3 + 1] = temp.y;
    image[mirror_index * 3 + 2] = temp.z;
    
  }
}
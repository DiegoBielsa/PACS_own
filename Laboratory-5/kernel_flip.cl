__kernel void image_flip(
  __global int *image,
  const unsigned int width,
  const unsigned int height){

  int x = get_global_id(0);
  int div = int(x/(width/2))

  if(x < width * height && div%2 == 0){
    int temp = image[x];
    int dif = (div*width / 2 - x) * 2;
    image[x] = image[dif+x]; 
    image[dif+x] = temp;
  }
}
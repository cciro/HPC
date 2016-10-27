#include <iostream>
#include <assert.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <cuda.h>


#define RED 2
#define GREEN 1
#define BLUE 0
#define CHANNELS 3

using namespace cv;
using namespace std;

__global__
void imgGrayGPU(unsigned char *imageInput, unsigned char *imageOutput, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if ((col < width) and (row < height)) {
    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * CHANNELS;

    unsigned char b = imageInput[rgbOffset + BLUE];
    unsigned char g = imageInput[rgbOffset + GREEN];
    unsigned char r = imageInput[rgbOffset + RED];

    imageOutput[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}

__global__ void sobel(unsigned char *img, int *M1, int *M2, unsigned char *G, int maskWidth, int w, int h)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int aux;
    if(col < w && row < h){
        int Gx = 0;
    int Gy = 0;
    float grad;

        int img_start_col = col - (maskWidth/2);
        int img_start_row = row - (maskWidth/2);
        for (int j = 0; j < maskWidth; j++){
            for (int k = 0; k < maskWidth; k++){
                int curRow = img_start_row + j;
                int curCol = img_start_col + k;
                if(curRow > -1 && curRow < h && curCol > -1 && curCol < w){
                    aux = (int)(img[curRow * w + curCol]);
                    Gx += aux * M1[j*maskWidth+k];
          Gy += aux * (char)M2[j*maskWidth+k];
                }
            }
        }
   
    grad = sqrtf((Gx* Gx)+ (Gy* Gy));
        G[row * w + col] = (unsigned char)grad;
    }
} 



int main() {
  cudaError_t error = cudaSuccess;
  int filtroX[3][3] =  {{-1,0,+1},{-2,0,+2},{-1,0,+1}};
  int filtroY[3][3] =  {{-1,-2,-1},{0,0,0},{+1,+2,+1}};
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  int *d_filtroX, *d_filtroY;
 
 
  Mat image, imgGray, grad;
  image = imread("./inputs/img2.jpg", CV_LOAD_IMAGE_COLOR);

  if (!image.data) {
    cerr << "No image data" << endl;
    return EXIT_FAILURE;
  }
 
  // Versión CPU con open cv ############################################################
  /*
 clock_t startCPU = clock();
  GaussianBlur( image, image, Size(3,3), 0, 0, BORDER_DEFAULT );

  // Convertimos la imágen a escala de grises
  cvtColor( image, imgGray, CV_RGB2GRAY );

 
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  // Gradiente X
  Sobel( imgGray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Gradiente Y
  Sobel( imgGray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradiente
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

 
    imwrite("./outputs/1088316102.png", grad);
 
  clock_t endCPU = clock();
  double cpu_time_used =  double (endCPU - startCPU)  / CLOCKS_PER_SEC;
  printf("Tiempo en CPU: %.10f\n", cpu_time_used);
   
  // Fin version open cv #####################################################################
  */
  // Versión GPU ########################################################################3##
 
  clock_t  startGPU = clock();
  unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput;// Variables para convertir la imágen a escala de grises
  // La variable h_imageOutput es una matriz de datos de la imagen en escala de grises.
  unsigned char *d_grayScale, *h_G, *d_G;
  Size s = image.size();

 
  int width = s.width;
  int height = s.height;
  int size = sizeof(unsigned char) * width * height * image.channels();
  int sizeGray = sizeof(unsigned char) * width * height;
  int sizeFilter = sizeof(int) * 3 * 3;

  dataRawImage = (unsigned char*)malloc(size);
  h_imageOutput = (unsigned char*)malloc(sizeGray);

  error = cudaMalloc((void**)&d_dataRawImage, size);
  if (error != cudaSuccess) {
   cerr << "Error reservando memoria para d_dataRawImage" << endl;
   return EXIT_FAILURE;
  }

  dataRawImage = image.data;
  startGPU = clock();
  error = cudaMemcpy(d_dataRawImage, dataRawImage, size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cerr << "Error copiando los datos de dataRawImage a d_dataRawImage" << endl;
    return EXIT_FAILURE;
  }

  error = cudaMalloc((void**)&d_imageOutput, sizeGray);
  if (error != cudaSuccess) {
    cerr << "Error reservando memoria para d_imageOutput" << endl;
    return EXIT_FAILURE;
  }
  // Convertimos la imágen a escala de grises
  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width / float(blockSize)), ceil(height / float(blockSize)), 1);
  imgGrayGPU<<< dimGrid, dimBlock >>>(d_dataRawImage, d_imageOutput, width, height);
  cudaMemcpy(h_imageOutput, d_imageOutput, sizeGray, cudaMemcpyDeviceToHost);
 

  Mat grayImg;
  grayImg.create(height, width, CV_8UC1);
  grayImg.data = h_imageOutput;
  h_G = (unsigned char*)malloc(sizeGray);
 

 
  error = cudaMalloc((void**)&d_G,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_M");
        exit(0);
    }
  error = cudaMalloc((void**)&d_filtroX,sizeFilter);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_M");
        exit(0);
    }
   
  error = cudaMalloc((void**)&d_filtroY,sizeFilter);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_M");
        exit(0);
    }
   
  error = cudaMalloc((void**)&d_grayScale,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_M");
        exit(0);
    }
 
  
 
  error = cudaMemcpy(d_filtroX, filtroX, sizeFilter, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando datos a d_M");
        exit(0);
    }
   
    error = cudaMemcpy(d_filtroY, filtroY, sizeFilter, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando datos a d_N");
        exit(0);
    }
   
  error = cudaMemcpy(d_grayScale, h_imageOutput, sizeGray, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando datos a d_M");
        exit(0);
    }
 
 
  // Aplicamos el filtro Sobel
  sobel<<<dimGrid,dimBlock>>>(d_grayScale, d_filtroX, d_filtroY, d_G, 3, width, height);
  cudaMemcpy(h_G,d_G,sizeGray,cudaMemcpyDeviceToHost);
  Mat imBorder;
  imBorder.create(height, width, CV_8UC1);
  imBorder.data = h_G;
    imwrite("./outputs/gatosobel.png", imBorder);
    clock_t  endGPU = clock();
  double gpu_time_used = double(endGPU - startGPU) / CLOCKS_PER_SEC;
  printf("Tiempo en GPU: %.10f\n",gpu_time_used);
 
  //Fin versión GPU###########################################################
  return 0;
}

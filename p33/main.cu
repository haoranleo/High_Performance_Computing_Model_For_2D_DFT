/* ECE 6122 Final Project - Cuda Part
* Writer: Haoran Li, Yingqiao Zheng
* Tested on Pace, College of Computing at Georgia Tech
*/

#include <iostream>
#include <cmath>
#include <time.h>
#include <string>
#include <chrono>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "complex.h"
#include "input_image.h"

#define TX 8
#define TY 8
#define PI 3.14159265358979
//#define GRAPHDIS
#define FILE_PATH_MAX 1024

using std::cout;
using std::endl;
using std::string;
using std::to_string;

__global__ void DFTComputeRow(float* d_real_output, float* d_imag_output, float* d_real_data, float* d_imag_data, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row * width + col;
    float temp_real = 0, temp_imag = 0;

    if(row >= 0 && row < height && col >= 0 && col < width) {
        int start_idx = row * width;
        for(int k = 0; k < width; k++) {
            float W_real = cos(2 * PI * col * k / width), W_imag = -sin(2 * PI * col * k / width);
            //printf("Index: %d, row: %d, col: %d, W_real: %f, W_imag: %f, real_data: %f, imag_data: %f", index, row, col, W_real, W_imag, d_real_data[start_idx + k], d_imag_data[start_idx + k]);
            temp_real += W_real * d_real_data[start_idx + k] - W_imag * d_imag_data[start_idx + k];
            temp_imag += W_real * d_imag_data[start_idx + k] + W_imag * d_real_data[start_idx + k];
        }
        d_real_output[index] = temp_real;
        d_imag_output[index] = temp_imag;
        __syncthreads();
    }

}


__global__ void DFTComputeCol(float* d_real_output, float* d_imag_output, float* d_real_data, float* d_imag_data, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row * width + col;
    float temp_real = 0, temp_imag = 0;
    //if(index == 0) printf("height: %d\n", height);
    if(row >= 0 && row < height && col >= 0 && col < width) {
        int start_idx = col;
        for(int k = 0; k < height; k++) {
            float W_real = cos(2 * PI * row * k / height), W_imag = -sin(2 * PI * row * k/ height);
            temp_real += W_real * d_real_data[start_idx + k * width] - W_imag * d_imag_data[start_idx + k * width];
            temp_imag += W_real * d_imag_data[start_idx + k * width] + W_imag * d_real_data[start_idx + k * width];
        }
        d_real_output[index] = temp_real;
        d_imag_output[index] = temp_imag;
        //printf("row: %d, col: %d, index: %d, compute: %f, %f \n", row, col, index, d_real_output[index], d_imag_output[index]);
        __syncthreads();
    }

}


__global__ void iDFTComputeRow(float* d_real_output, float* d_imag_output, float* d_real_data, float* d_imag_data, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row * width + col;
    float temp_real = 0, temp_imag = 0;

    if(row >= 0 && row < height && col >= 0 && col < width) {
        int start_idx = row * width;
        for(int k = 0; k < width; k++) {
            float W_real = cos(2 * PI * col * k / width), W_imag = sin(2 * PI * col * k/ width);
            temp_real += W_real * d_real_data[start_idx + k] - W_imag * d_imag_data[start_idx + k];
            temp_imag += W_real * d_imag_data[start_idx + k] + W_imag * d_real_data[start_idx + k];
        }
        temp_real /= width;
        temp_imag /= width;
        d_real_output[index] = temp_real;
        d_imag_output[index] = temp_imag;
        __syncthreads();
    }

}


__global__ void iDFTComputeCol(float* d_real_output, float* d_imag_output, float* d_real_data, float* d_imag_data, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row * width + col;
    float temp_real = 0, temp_imag = 0;
    if(row >= 0 && row < height && col >= 0 && col < width) {
        int start_idx = col;
        for(int k = 0; k < height; k++) {
            float W_real = cos(2 * PI * row * k / height), W_imag = sin(2 * PI * row * k/ height);
            temp_real += W_real * d_real_data[start_idx + k * width] - W_imag * d_imag_data[start_idx + k * width];
            temp_imag += W_real * d_imag_data[start_idx + k * width] + W_imag * d_real_data[start_idx + k * width];
        }
        temp_real /= width;
        temp_imag /= width;
        d_real_output[index] = temp_real;
        d_imag_output[index] = temp_imag;
        __syncthreads();
    }

}


int main(int argc, char** argv) {
    if(argc != 4) {
        cout << "ERROR! Incorrect input parameters format! Please enter parameter like this:" << endl;
        cout << "time ./NAME [forward/reverse] [INPUTFILE] [OUTPUTFILE]" << endl;
        exit(0);
    }

    string execution = argv[1];
    if(execution != "forward" && execution != "reverse") {
        cout << "ERROR! Invalid execution type, you can only specify either 'forward' or 'reverse'!" << endl;
        exit(0);
    }

    auto start = std::chrono::system_clock::now();
    char buf[FILE_PATH_MAX];
    if(getcwd(buf, FILE_PATH_MAX) == 0) {
        cout << "ERROR! Unable to get current working directory!" << endl;
        exit(0);
    }
    string cur_work_path = buf;
    string file_path = cur_work_path + "/" + argv[2];
    string out_file_path = argv[3];
    InputImage image(file_path.c_str());

    int width = image.get_width();
    int height = image.get_height();
    int size = width * height;
    Complex *data;
    float *real_data;
    float *imag_data;
    float *d_real_data;
    float *d_imag_data;
    float *d_real_output;
    float *d_imag_output;

    data = (Complex*)malloc(size * sizeof(Complex));
    real_data = (float*)malloc(size * sizeof(float));
    imag_data = (float*)malloc(size * sizeof(float));

    data = image.get_image_data();

    for(int cnt = 0; cnt < size; cnt++) {
        real_data[cnt] = data[cnt].real;
        imag_data[cnt] = data[cnt].imag;
    }

    const int BX = (width + TX - 1) / TX;
    const int BY = (height + TY - 1) / TY;
    dim3 blocks(BX, BY);
    dim3 threads(TX, TY);

#ifdef GRAPHDIS
    for(int idx = 0; idx < size; idx++) {
        cout << data[idx] << " ";
        if(idx % width == width - 1) {
            cout << endl;
        }
    }
#endif

    cudaMalloc((void**)&d_real_data, size * sizeof(float));
    cudaMalloc((void**)&d_imag_data, size * sizeof(float));
    cudaMalloc((void**)&d_real_output, size * sizeof(float));
    cudaMalloc((void**)&d_imag_output, size * sizeof(float));

    cudaMemcpy(d_real_data, real_data, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag_data, imag_data, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_real_output, real_data, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag_output, imag_data, size * sizeof(float), cudaMemcpyHostToDevice);

    //DFT
    DFTComputeRow<<<blocks, threads>>>(d_real_output, d_imag_output, d_real_data, d_imag_data, width, height);

    cudaMemcpy(real_data, d_real_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imag_data, d_imag_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_real_data, real_data, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag_data, imag_data, size * sizeof(float), cudaMemcpyHostToDevice);
        
    DFTComputeCol<<<blocks, threads>>>(d_real_output, d_imag_output, d_real_data, d_imag_data, width, height);

    cudaMemcpy(real_data, d_real_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imag_data, d_imag_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    if(execution == "reverse") {   //iDFT
        cudaMemcpy(d_real_data, real_data, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_imag_data, imag_data, size * sizeof(float), cudaMemcpyHostToDevice);

        iDFTComputeRow<<<blocks, threads>>>(d_real_output, d_imag_output, d_real_data, d_imag_data, width, height);

        cudaMemcpy(real_data, d_real_output, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(imag_data, d_imag_output, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_real_data, real_data, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_imag_data, imag_data, size * sizeof(float), cudaMemcpyHostToDevice);
        
        iDFTComputeCol<<<blocks, threads>>>(d_real_output, d_imag_output, d_real_data, d_imag_data, width, height);

        cudaMemcpy(real_data, d_real_output, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(imag_data, d_imag_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    auto end = std::chrono::system_clock::now();

#ifdef GRAPHDIS
    cout << "START PRINT CPU RESULT!!" << endl;
#endif
    for(int cnt = 0; cnt < size; cnt++) {
        data[cnt].real = real_data[cnt];
        data[cnt].imag = imag_data[cnt];
#ifdef GRAPHDIS
        cout << "(" << real_data[cnt] << ", " << imag_data[cnt] << ")" << "  ";
        if(cnt % width == width - 1) {
            cout << endl;
        }
#endif
    }
    std::chrono::duration<double> duration = end - start;
    cout << "System Running Time: " << 1000 * duration.count() << "ms" << endl;
    image.save_image_data(out_file_path.c_str(), data, width, height);

    free(data); free(real_data); free(imag_data);
    cudaFree(d_real_data); cudaFree(d_imag_data); cudaFree(d_real_output); cudaFree(d_imag_output);
    return 0;
}

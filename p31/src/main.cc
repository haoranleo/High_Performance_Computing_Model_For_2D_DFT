#include "input_image.h"
#include "complex.h"
#include <iostream>
#include <thread>
#include <string>
#include <chrono>
#include <math.h>

using namespace std;
using namespace std::chrono;

static const int ERROR = 1;
static const int SUCCESS = 0;
static const int num_threads = 8;
const float PI = 3.14159265358979f;

static const int parts = num_threads - 1;


void fft(Complex* input, Complex* output,
        bool isRow, uint32_t N,
         uint32_t rowl, uint32_t index){
    auto log2n = (uint32_t)log2(N);
    if(isRow){
        //Perform fft computation along a row
        auto rowmult = (uint32_t) rowl*index;
        for(uint32_t a = 0; a < N; ++a){
            uint32_t b = a;
            b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
            b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
            b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
            b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
            b = ((b >> 16) | (b << 16)) >> (32 - log2n);
            output[rowmult + a] = input[rowmult + b];
            /*
            string c = to_string(a) + "-" + to_string(b) + "\n";
            cout << c;
            */
        }

        for(int s = 1; s <= (int) log2n; ++s){
            int m = 1 << s;
            int m2 = m >> 1;
            float theta = -PI/m2;
            Complex w(1.0f);
            Complex wm(cosf(theta), sinf(theta));
            for(int j = 0; j < m2; ++j){
                for(int k = j; k < (int) N; k +=m){
                    Complex t = w*output[rowmult + k + m2];
                    Complex u = output[rowmult + k];
                    output[rowmult + k] = u + t;
                    output[rowmult + k + m2] = u - t;
                }/*
                if(s == 1 && j == 0 && index == 0){
                    for(int i = 0; i < N; ++i){
                        cout << output[i] << ",";
                    }
                }*/
                w = w*wm;
            }
        }


    } else {
        //Perform fft computation along a column
        for(uint32_t a = 0; a < N; ++a){
            uint32_t b = a;
            b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
            b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
            b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
            b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
            b = ((b >> 16) | (b << 16)) >> (32 - log2n);
            output[rowl*a + index] = input[rowl*b + index];
        }

        for(int s = 1; s <= (int) log2n; ++s){
            int m = 1 << s;
            int m2 = m >> 1;
            Complex w(1.0f);
            float theta = -PI/m2;
            Complex wm(cosf(theta), sinf(theta));
            for(int j = 0; j < m2; ++j){
                for(int k = j; k < (int) N; k +=m){
                    int tempidx = index + k*rowl;
                    int tempidx2 = tempidx + m2*rowl;
                    Complex t = w*output[tempidx2];
                    Complex u = output[tempidx];
                    output[tempidx] = u + t;
                    output[tempidx2] = u - t;
                }
                w = w*wm;
            }
        }
    }
}

void dft(Complex* input, Complex* output,
         bool isRow, uint32_t N,
         uint32_t rowl, uint32_t index){

    if(isRow){
        int rowmult = index*rowl;
        for(int i = 0; i < N; ++i){
            output[rowmult + i] = Complex(0.0f);
            for(int k = 0; k < N; ++k){
                double temp = 2*PI*i*k/N;
                output[rowmult + i] =
                        output[rowmult + i] +
                        input[rowmult + k]*
                        Complex(cos(temp), -sin(temp));
            }
        }
    } else {
        for(int i = 0; i < N; ++i){
            output[i*rowl + index] = Complex(0.0f);
            for(int k = 0; k < N; ++k){
                double temp = 2*PI*i*k/N;
                output[i*rowl + index] =
                        output[i*rowl + index] + input[k*rowl + index]*
                        Complex(cos(temp), -sin(temp));
            }
        }
    }

}

void threaded_dft(InputImage img, Complex *intermImg, int tid, bool isRow) {

    if(isRow){
        int low_bound = tid*img.get_height()/num_threads;
        int hi_bound = (tid+1)*img.get_height()/num_threads-1;
        string output = to_string(tid) + ", "
                + to_string(low_bound) + ", " + to_string(hi_bound) + "\n";
        for(int row = low_bound; row <= hi_bound; ++row){
            dft(img.get_image_data(), intermImg, isRow,
                (uint32_t) img.get_width(),
                (uint32_t) img.get_width(), (uint32_t) row);
        }
    } else {
        int low_bound = tid*img.get_width()/num_threads;
        int hi_bound = (tid+1)*img.get_width()/num_threads-1;
        for(int col = low_bound; col <= hi_bound; ++col){
            dft(intermImg, img.get_image_data(), isRow,
                (uint32_t) img.get_height(),
                (uint32_t) img.get_width(), (uint32_t) col);
        }
    }
}

void dft2d(InputImage inputImg){
    std::thread t[parts];
    auto *intermImg = (Complex *) malloc(sizeof(Complex)*inputImg.get_height()*inputImg.get_width());

    //Launch a group of threads
    for (int i = 0; i < parts; ++i) {
        t[i] = thread(threaded_dft,
                      ref(inputImg), intermImg, i, true);
    }

    threaded_dft(ref(inputImg), intermImg, parts, true);

    //Join the threads with the main thread
    for (auto& v:t) {
        v.join();
    }

    thread t2[parts];
    for (int i = 0; i < parts; ++i) {
        t2[i] = thread(threaded_dft,
                       ref(inputImg), intermImg, i, false);
    }

    threaded_dft(ref(inputImg), intermImg, parts, false);

    for (auto& v:t2){
        v.join();
    }

    free(intermImg);

}

void idft2d(InputImage inputImg){
    for (int i = 0; i < inputImg.get_height()*inputImg.get_width(); ++i) {
        inputImg.get_image_data()[i] = inputImg.get_image_data()[i].conj();
    }
    dft2d(inputImg);
    Complex normalize(1.0f/inputImg.get_height()/inputImg.get_width());
    for (int i = 0; i < inputImg.get_height()*inputImg.get_width(); ++i) {
        inputImg.get_image_data()[i] = inputImg.get_image_data()[i].conj()*normalize;
    }
}

int main(int argc, char *argv[]) {

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    if(argc < 4){
        cout << "Not enough arguments provided. Exiting.\n";
        return ERROR;
    }

    InputImage inputImg(argv[2]);
    cout << "Image:\n Height = " << inputImg.get_height() <<
         ", Width = " << inputImg.get_width() << endl;

    if(argv[1][0] == 'F' || argv[1][0] == 'f'){
        dft2d(inputImg);
        inputImg.save_image_data(argv[3], inputImg.get_image_data(),
                                 inputImg.get_width(), inputImg.get_height());
    } else {
        idft2d(inputImg);
        inputImg.save_image_data_real(argv[3], inputImg.get_image_data(),
                                 inputImg.get_width(), inputImg.get_height());
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();

    cout << "Execution time: " << duration << " ms" << endl;

    return SUCCESS;
}
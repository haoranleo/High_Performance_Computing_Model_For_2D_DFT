#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include "complex.h"
#include "input_image.h"
#include <chrono>
#include <string>
const float PI = 3.14159265358979f;

using namespace std::chrono;
using std::string;


void divide(Complex* buffer, int start, int end) {
	int n = end - start + 1;
	Complex* temp = (Complex *)malloc(sizeof(Complex) * n / 2);  // get temp heap storage
	for (int i = 0; i < n / 2; i++)    // copy all odd elements to heap storage
		temp[i] = buffer[start + i * 2 + 1];
	for (int i = 0; i < n / 2; i++)    // copy all even elements to lower-half of a[]
		buffer[start + i] = buffer[start + i * 2];
	for (int i = 0; i < n / 2; i++)    // copy all odd (from heap) to upper-half of a[]
		buffer[start + i + n / 2] = temp[i];
	free(temp);                 // delete heap storage
}

void fft(Complex* buffer, int start, int end) {
	int N = end - start + 1;
	if (N >= 2) {
		divide(buffer, start, end);      // all evens to lower half, all odds to upper half
		fft(buffer, start, start + N / 2 - 1);   // recurse even items
		fft(buffer, start + N / 2, end);   // recurse odd  items
										   // combine results of two half recursions
		for (int k = start; k < start + N / 2; k++) {
			Complex even(buffer[k].real, buffer[k].imag);
			Complex odd(buffer[k + N / 2].real, buffer[k + N / 2].imag);
			// w is the "twiddle-factor"
			Complex w(cos(2 * PI * (k - start) / N), -sin(2 * PI * (k - start) / N));
			buffer[k] = even + w * odd;
			buffer[k + N / 2] = even - w * odd;
		}
	}
}

int main(int argc, char* argv[]) {

	if (argc < 4) {
		std::cout << "Please input correct number of arguments!" << std::endl;
		exit(1);
	}

	string direction = argv[1];
	InputImage input(argv[2]);		//read the input
	string outputfile = argv[3];
	

	// Initialize MPI environment
	int rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) {
		printf("ERROR INITIALIZING MPI PROGRAM\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}
	
	int rank, processorNum;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &processorNum);

	int length = input.get_width();
	int num_pp = input.get_width() / 8;	//number of rows each processors need to calculate

	MPI_Datatype dt_complex;
	MPI_Type_contiguous(2, MPI_FLOAT, &dt_complex);
	MPI_Type_commit(&dt_complex);

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	if (direction[0] == 'f' || direction[0] == 'F') {
		Complex* buffer_input = (Complex *)malloc(sizeof(Complex)*length * num_pp);


		MPI_Scatter(input.get_image_data(), length * num_pp, dt_complex, buffer_input, length * num_pp, dt_complex, 0, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);

		//calculation
		for (int j = 0; j < num_pp; j++) {
			fft(buffer_input, j * length, (j + 1) * length - 1);
		}

		MPI_Gather(buffer_input, length*num_pp, dt_complex, input.get_image_data(), length*num_pp, dt_complex, 0, MPI_COMM_WORLD);


		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == 0) {
			//send data to other processors
			Complex* temp = (Complex*)malloc(sizeof(Complex) * length * length);
			for (int i = 0; i < length; i++) {
				for (int j = 0; j < length; j++) {
					temp[j * length + i] = input.get_image_data()[i * length + j];
				}
			}
			MPI_Scatter(temp, length * num_pp, dt_complex, buffer_input, length * num_pp, dt_complex, 0, MPI_COMM_WORLD);
		}
		else {
			MPI_Scatter(NULL, length * num_pp, dt_complex, buffer_input, length * num_pp, dt_complex, 0, MPI_COMM_WORLD);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		//calculation
		for (int j = 0; j < num_pp; j++) {
			fft(buffer_input, j * length, (j + 1) * length - 1);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Gather(buffer_input, length * num_pp, dt_complex, input.get_image_data(), length * num_pp, dt_complex, 0, MPI_COMM_WORLD);

		if (rank == 0) {
			Complex *temp1 = (Complex*)malloc(sizeof(Complex) *length *length);
			for (int i = 0; i < length; i++) {
				for (int j = 0; j < length; j++) {
					temp1[i * length + j] = input.get_image_data()[j * length + i];
				}
			}
			for (int i = 0; i < length * length; i++) {
				input.get_image_data()[i] = temp1[i];
			}
			free(temp1);
		}

		MPI_Barrier(MPI_COMM_WORLD);



		if (rank == 0) {
			input.save_image_data(outputfile.c_str(), input.get_image_data(), length, length);
		}

		free(buffer_input);
	}
	else {
		Complex* buffer_input = (Complex *)malloc(sizeof(Complex)*length * num_pp);

		// reverse fft
		if (rank == 0) {
			for (int i = 0; i < input.get_width() * input.get_height(); i++) {
				input.get_image_data()[i] = input.get_image_data()[i].conj();
			}
		}

		MPI_Scatter(input.get_image_data(), length * num_pp, dt_complex, buffer_input, length * num_pp, dt_complex, 0, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);

		//calculation
		for (int j = 0; j < num_pp; j++) {
			fft(buffer_input, j * length, (j + 1) * length - 1);
		}

		MPI_Gather(buffer_input, length*num_pp, dt_complex, input.get_image_data(), length*num_pp, dt_complex, 0, MPI_COMM_WORLD);


		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == 0) {
			//send data to other processors
			Complex* temp = (Complex*)malloc(sizeof(Complex) * length * length);
			for (int i = 0; i < length; i++) {
				for (int j = 0; j < length; j++) {
					temp[j * length + i] = input.get_image_data()[i * length + j];
				}
			}
			MPI_Scatter(temp, length * num_pp, dt_complex, buffer_input, length * num_pp, dt_complex, 0, MPI_COMM_WORLD);
		}
		else {
			MPI_Scatter(NULL, length * num_pp, dt_complex, buffer_input, length * num_pp, dt_complex, 0, MPI_COMM_WORLD);
		}


		MPI_Barrier(MPI_COMM_WORLD);
		//calculation
		for (int j = 0; j < num_pp; j++) {
			fft(buffer_input, j * length, (j + 1) * length - 1);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Gather(buffer_input, length * num_pp, dt_complex, input.get_image_data(), length * num_pp, dt_complex, 0, MPI_COMM_WORLD);

		if (rank == 0) {
			Complex *temp1 = (Complex*)malloc(sizeof(Complex) *length *length);
			for (int i = 0; i < length; i++) {
				for (int j = 0; j < length; j++) {
					temp1[i * length + j] = input.get_image_data()[j * length + i];
				}
			}
			for (int i = 0; i < length * length; i++) {
				input.get_image_data()[i] = temp1[i];
			}
			free(temp1);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == 0) {
			Complex normalize = Complex(1.0f / input.get_width() / input.get_height());
			for (int i = 0; i < input.get_width() * input.get_height(); i++) {

				input.get_image_data()[i] = input.get_image_data()[i].conj()*normalize;
				//input.get_image_data()[i] = input.get_image_data()[i].conj();
			}

			input.save_image_data_real(outputfile.c_str(), input.get_image_data(), length, length);
		}

		free(buffer_input);
	}

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t2 - t1).count();
	std::cout << "Excution Time: " << duration << " ms" << std::endl;

	MPI_Finalize();


}

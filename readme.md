## ECE 6122 Final Project

This project aims to implement different computing methods to solve 2D DFT problem efficiently.

See project description here: [Project Description](./FinalProject.pdf)

## File list
p31: C++ thread - DFT

p32: C++ MPI - FFT

p33: C++ CUDA - DFT

p34: C++ thread - FFT

## Usage:
Use following command, and four executable file (p31, p32, p33, p34) will be generated.
```
	cd build
	cmake ..
	make
```
p31~p34 usage:
```
./p31 forward/reverse [Inputfile][Outputfile]
```
ex: **./p31 forward Tower256.txt Output256.txt** will perform 2 2D DFT using Tower256.txt as input and producing an output file named Output256.txt.

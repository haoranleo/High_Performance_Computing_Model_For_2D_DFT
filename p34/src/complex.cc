//
// Created by brian on 11/20/18.
//

#include "complex.h"

#include <cmath>

const float PI = 3.14159265358979f;

Complex::Complex() : real(0.0f), imag(0.0f) {}

Complex::Complex(float r) : real(r), imag(0.0f) {}

Complex::Complex(float r, float i) : real(r), imag(i) {}

Complex Complex::operator+(const Complex &b) const {
    return Complex((real+b.real), (imag+b.imag));
}

Complex Complex::operator-(const Complex &b) const {
    return Complex((real-b.real), (imag-b.imag));
}

Complex Complex::operator*(const Complex &b) const {
    return Complex((this->real*b.real - this->imag*b.imag),
                   (this->real*b.imag + this->imag*b.real));
}

float Complex::mag() const {
    return sqrt(real*real + imag*imag);
}

float Complex::angle() const {
    return atan(imag/real);
}

Complex Complex::conj() const {
    return Complex(real, -imag);
}

std::ostream& operator<< (std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}
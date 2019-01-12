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
    Complex tmp;
    tmp.real = b.real + this->real;
    tmp.imag = b.imag + this->imag;
    return tmp;
}

Complex Complex::operator-(const Complex &b) const {
    Complex tmp;
    tmp.real = this->real - b.real;
    tmp.imag = this->imag - b.imag;
    return tmp;
}

Complex Complex::operator*(const Complex &b) const {
    Complex tmp;
    tmp.real = this->real * b.real - this->imag * b.imag;
    tmp.imag = this->real * b.imag + this->imag * b.real;
    return tmp;
}

float Complex::mag() const {
    float mag;
    mag = sqrtf(this->real * this->real + this->imag * this->imag);
    return mag;
}

float Complex::angle() const {
    float angle;
    angle = (float)atan (this->imag / this->real) * 180 / PI;
    return angle;
}

Complex Complex::conj() const {
    Complex tmp;
    tmp.real = this->real;
    tmp.imag = - this->imag;
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
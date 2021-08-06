# sfft
Image Subtraction in Fourier Space 

## backends
sfft has three different backends to perform the subtraction, which are Numpy-based, Pycuda-based and Cupy-based, respectively. 

a) Numpy backend: sfft will totally run on the CPU devices. 
b) Pycuda backend: GPU device(s) with double-precision support are required. The core functions of sfft are written in CUDA, Pycuda and Scikit-Cuda.
c) Cupy backend: GPU device(s) with double-precision support are required. The core functions of sfft are written in CUDA and Cupy.

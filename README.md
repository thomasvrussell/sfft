# sfft
Image Subtraction in Fourier Space 

## Prerequisites
sfft has three different backends to perform the subtraction, which are Numpy-based, Pycuda-based and Cupy-based, respectively. 

Using Numpy backend sfft will totally run on the CPU devices. 
While the later two backends require GPU device(s) and the core functions are written by CUDA (though already wrapped in Python).

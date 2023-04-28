# Image to Pencil Sketch with Python

This project aims to create a pencil sketch of an image using Python.

## Objective
The objective of this project is to use image processing techniques in Python to create a pencil sketch of an input image.

## Approach
The following steps are involved in creating a pencil sketch of an image using Python:

1. Read the input image in RGB format.

2. Convert the RGB image to a grayscale image using the cv2.cvtColor() function.

3. Invert the grayscale image to obtain a negative image.

4. Blur the negative image using the GaussianBlur() function from the cv2 library.

5. Invert the blurred image.

6. Divide the grayscale image by the inverted blurred image to obtain the final pencil sketch.


## Dataset
No external dataset is required for this project as the input image can be any image of choice.

Reference - https://thecleverprogrammer.com/2020/09/30/pencil-sketch-with-python/

## Results
The output of this project is a pencil sketch of the input image. The final pencil sketch has a black and white effect and is similar to a hand-drawn sketch.

## Conclusion
This project demonstrates the use of image processing techniques in Python to create a pencil sketch of an image. The steps involved are straightforward and can be easily implemented using the cv2 library. The final output is an aesthetically pleasing image that can be used for various applications.

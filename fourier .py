import cv2
import numpy as np


def edge_detection_fourier(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    f_transform = np.fft.fft2(gray_image)

    f_transform_shifted = np.fft.fftshift(f_transform)

    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    f_transform_shifted[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    f_transform_inverse = np.fft.ifftshift(f_transform_shifted)
    image_filtered = np.fft.ifft2(f_transform_inverse)
    image_filtered = np.abs(image_filtered)

    _, binary_edges = cv2.threshold(image_filtered, 30, 255, cv2.THRESH_BINARY)

    return binary_edges


image = cv2.imread(r"C:\Users\zbook 17 g3\Desktop\runs\parked.jpg")

edges = edge_detection_fourier(image)

cv2.imshow('Original Image', image)
cv2.imshow('Edge Detection (Fourier Transform)', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
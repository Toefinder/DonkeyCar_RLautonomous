import cv2
import numpy as np

# def detect_edge(image, kernel_size=7, low_threshold=80, high_threshold=90):
#     im_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
#     im_edge = cv2.Canny(np.uint8(im_blurred), low_threshold, high_threshold)

#     return im_edge

def detect_edge(image, kernel_size=5, low_threshold=80, high_threshold=200):
    im_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    im_edge = cv2.Canny(np.uint8(im_blurred), low_threshold, high_threshold)

    return im_edge
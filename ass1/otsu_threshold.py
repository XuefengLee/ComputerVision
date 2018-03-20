import cv2
import numpy as np

image =cv2.imread('ductile_iron2-0.jpg',0)
sorted_pix = image.ravel()
length = len(sorted_pix)
sums = np.cumsum(sorted_pix)
reverse_sums = np.cumsum(sorted_pix[::-1])[::-1]
candidates = []

for i in range(255):
    backs = sorted_pix > i
    noduels = sorted_pix <= i
    
    sum1 = np.sum(backs)
    sum2 = np.sum(noduels)
    if sum1 == 0 or sum2 == 0:
        candidates.append(0)
        continue
    
    w1 = sum1/length
    w2 = sum2/length
    u1 = np.sum(backs*sorted_pix)/sum1
    u2 = np.sum(noduels*sorted_pix)/sum2
    result = w1*w2*(u1-u2)**2
    candidates.append(result)

threshold = np.argmax(candidates)
(thresh,image2) = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
image = (image > threshold)*255
cv2.imshow('image',np.array(image, dtype=np.uint8))
cv2.waitKey(0)

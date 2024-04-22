import cv2
import matplotlib.pyplot as plt
import easyocr
# instance text detector
reader = easyocr.Reader(['en'], gpu=False)
# read image
image_path = '/content/bw.jpeg'

img = cv2.imread(image_path)
# converting the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# inverting the grayscale image
inverted_image = 255 - gray_image
# Apply threshold (pixel values less than 200 become zero)
threshold_value = 180
_, thresholded_image = cv2.threshold(inverted_image, threshold_value, 255, cv2.THRESH_BINARY)
# Inverting the tresholded image
inverted_image = 255 - thresholded_image
# converting thresholded image to 3-channel image
thresholded_3channel = cv2.merge([inverted_image] * 3)
# applying Gaussian blur
thresholded_3channel = cv2.GaussianBlur(thresholded_3channel, (5, 5), 0)  # Kernel size (5, 5) can be adjusted


# detect text on image
text_ = reader.readtext(thresholded_3channel)

threshold = 0.25
# draw bbox and text
for t_, t in enumerate(text_):
    print(t)

    bbox, text, score = t

    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 255), 2)
        cv2.putText(img, text, [bbox[0][0]+10,bbox[0][1]+20], cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
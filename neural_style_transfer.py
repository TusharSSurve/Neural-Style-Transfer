import cv2 

model = cv2.dnn.readNetFromTorch('Neural Style Transfer/models/eccv16/starry_night.t7')

img = cv2.imread('Resources/Face2.jpg')
blob = cv2.dnn.blobFromImage(img,1.0,(img.shape[1], img.shape[0]),(103.939, 116.779, 123.680), swapRB=False, crop=False)

model.setInput(blob)
output = model.forward()

output = output.reshape((3, output.shape[2], output.shape[3]))
output[0] += 103.939
output[1] += 116.779
output[2] += 123.680
output /= 255.0
output = output.transpose(1, 2, 0)

cv2.imshow("Input", img)
cv2.imshow("Output", output)
cv2.waitKey(0)
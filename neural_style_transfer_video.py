import cv2
import os

path = 'Neural Style Transfer/models/eccv16/'
path1 = 'Neural Style Transfer/models/instance_norm/'

models = [path+i for i in os.listdir(path) if i.endswith('.t7')]
models.extend([path1+i for i in os.listdir(path1) if i.endswith('.t7')])

i = 0 
model = cv2.dnn.readNetFromTorch(models[i])
cap = cv2.VideoCapture(0)
while True:
    succ,img = cap.read()
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
    key = cv2.waitKey(1) & 0xff
    if key==ord('n'):
        if i<len(models)-1:
            i+=1
            model = cv2.dnn.readNetFromTorch(models[i])
    if key==ord('p'):
        if i>0:
            i-=1
            model = cv2.dnn.readNetFromTorch(models[i])
    if key==ord('q'):
        break
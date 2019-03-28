import cv2
import pandas as pd 
import numpy as np 


facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
StudentFaceCount = 0;

StudentRoll = int(input("Enter Student Roll Number : "));
StudentName = input("Enter Student Name : ");

cam = cv2.VideoCapture(0)

while(True):
	ret,img = cam.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = facedetect.detectMultiScale(gray,1.3,5)

	for(x,y,w,h) in faces:
		StudentFaceCount = StudentFaceCount + 1;
		cv2.imwrite('studentsData/'+str(StudentName)+'_'+str(StudentFaceCount)+'.jpg',gray[y:y+h,x:x+w])
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.waitKey(100)
	cv2.imshow('face',img)
	cv2.waitKey(1)
	if(StudentFaceCount>=100):
		break

cam.release()
cv2.destroyAllWindows()
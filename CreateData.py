import cv2
import numpy as np
import os 
import csv      

Id=input('Enter ID : ')
name=input('Name Of Student : ')
os.makedirs("dataset"+'/'+Id)
	
harcascadePath = "haarcascade_frontalface_default.xml"
detector=cv2.CascadeClassifier(harcascadePath)
pic_no=0
ret=True
#x=input('enter') To Create Data From A Photo
cap = cv2.VideoCapture(0)#To Take Input From Default WebCam
while ret:
	ret,frame=cap.read()
	gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=detector.detectMultiScale(gray_img,1.15,8)
	for (x,y,w,h) in faces:
		cropped=frame[y:y+h,x:x+w]
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		pic_no=pic_no+1
		#cv2.imwrite("TrainingImage\ "+Id +'.'+ str(pic_no) + ".jpg", cropped)
		cv2.imwrite("dataset"+'/'+Id+'/'+str(pic_no)+'.jpg',cropped)
	cv2.imshow('frame',frame)
	cv2.waitKey(50)
	if(pic_no>50):
		break
cap.release()
cv2.destroyAllWindows() 
row = [Id , name]
with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
	writer = csv.writer(csvFile)
	writer.writerow(row)
csvFile.close()

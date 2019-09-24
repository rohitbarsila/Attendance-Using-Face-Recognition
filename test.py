import cv2,os
import shutil
import csv
import numpy as np
import pandas as pd
import datetime
import time
import faceRecognition as fr

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainDATA.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
df=pd.read_csv("StudentDetails\StudentDetails.csv")
x=str(input('Enter Location Of Students Photo (Without ["]): '))
cap = cv2.VideoCapture(x)
col_names =  ['Id','Name','Date','Time']
attendance = pd.DataFrame(columns = col_names)
currentDate = time.strftime("%d_%m_%y")
df=pd.read_csv("StudentDetails\StudentDetails.csv")
while True:
	ret,test_img=cap.read()
	faces_detected,gray_img=fr.faceDetection(test_img) 
	for (x,y,w,h) in faces_detected:
		cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5) 
	cv2.imshow('face detection Tutorial',test_img)
	cv2.waitKey(2000)
	for face in faces_detected:
		(x,y,w,h)=face
		roi_gray=gray_img[y:y+h,x:x+w]
		Id,confidence=face_recognizer.predict(roi_gray)
		fr.draw_rect(test_img,face)
		if confidence>0:
			ts = time.time()      
			date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
			timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
			aa=df.loc[df['Id'] == Id]['Name'].values
			tt=str(Id)+"-"+aa
			attendance.loc[len(attendance)]=[Id,aa,date,timeStamp]            
			cv2.putText(test_img,'text',(x+w,y+h),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
		attendance=attendance.drop_duplicates(subset=['Id'],keep='first',inplace=False)    
	if (cv2.waitKey(1)==ord('q')):
			break
	ts = time.time()      
	date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
	timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
	Hour,Minute,Second=timeStamp.split(":")
	fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
	attendance.to_csv(fileName,index=False)
	print(attendance)
cap.release()
cv2.destroyAllWindows()
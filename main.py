import cv2
import os
import logging as log
import datetime as dt
from time import sleep
import pyautogui
import threading

def f():
    while True:
        while stop_event.is_set():
            print("Waiting for Event")
            sleep(1)
        print("No Face")
        sleep(5)
        if stop_event.is_set():
            continue
        pyautogui.hotkey('winleft', 'l')


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

x = threading.Thread(target=f)
stop_event = threading.Event()
stop_event.set()
x.start()

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
    if len(faces) == 0:
        stop_event.clear()
    else:
        stop_event.set()


    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Video', frame)

video_capture.release()
cv2.destroyAllWindows()

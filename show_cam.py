import cv2

    
while True:
    video_capture = cv2.VideoCapture(3)
    frame = video_capture.read()[1]
    cv2.imshow("frame", frame)
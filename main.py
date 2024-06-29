import cv2 # type: ignore
#   ================ Face Detection ==============
face_cap = cv2.CascadeClassifier("C:/Python/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cam = cv2.VideoCapture(0)
while (True):
    ret, video_data = video_cam.read()
    col = cv2.cvtColor(video_data,cv2.COLOR_RGB2GRAY)  # color of the image..
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow("Camera",video_data)  # yha pr viedo/picture ka frame bana k liya use hota hai imshow.add()
    if cv2.waitKey(10) == ord("a"):   # a sa camera off hota hai... 
        break
video_cam.release()    
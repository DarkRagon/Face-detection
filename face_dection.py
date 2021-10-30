import cv2

#Load some pre trained data on face frontals from opencv
trained_face_data=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#choose an image to detect faces in
#img=cv2.imread('rdj1.jpg')

#Capturing video from webcam
webcam=cv2.VideoCapture(0)

#Iterate forever over the frame
while True:
    successful_frame_read, frame=webcam.read()

    #Must convert to grayscale
    grayscaled_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)

    #Drawing the rectangle around the face dynamically
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y), (x+w , y+h), (0,255,0), 2)

    #image detection message displaying the images
    cv2.imshow('Image detected',frame)
    #holding the screen
    key=cv2.waitKey(1)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break

#Release the video capture
webcam.release()

print("code completed")




"""
#Detect faces
face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)

#Drawing the rectangle around the face dynamically

for(x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y), (x+w , y+h), (0,255,0), 2)

#print(face_coordinates)

#image detection message displaying the images
cv2.imshow('Image detected',img)

#holding the screen
cv2.waitKey()

print("code completed")
"""
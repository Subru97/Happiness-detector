import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(grayscaleImage, originalImage):
    faces = face_cascade.detectMultiScale(grayscaleImage, scaleFactor= 1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(originalImage, (x, y), (x+w, y+h), (255, 0, 0), 2)
        region_of_interest_gray = grayscaleImage[y:y+h, x:x+w]
        region_of_interest_original = originalImage[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(region_of_interest_gray, scaleFactor= 1.1, minNeighbors=3)
        smiles = smile_cascade.detectMultiScale(region_of_interest_gray, scaleFactor= 1.7, minNeighbors=22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(region_of_interest_original, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(region_of_interest_original, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return originalImage

video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    grayScaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(grayScaleFrame, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
import cv2 as cv
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#img = cv.imread("face.jpg")
cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    flipped = cv.flip(src=frame, flipCode=1)
    greyscale = cv.cvtColor(src=flipped, code=cv.COLOR_BGR2GRAY)
    faces = detector(greyscale)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        landmarks = predictor(image=greyscale, box=face)
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv.circle(img=flipped, center=(x, y), radius=1, color=(0, 255, 0), thickness=-1)

    cv.imshow(winname="Face", mat=flipped)

    if cv.waitKey(delay=1) == 27:
        break

cap.release()
cv.destroyAllWindows()


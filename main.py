import dlib
from skimage import io
import cv2


# Load models
predictor_path = 'shape_predictor_5_face_landmarks.dat' 
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


img = io.imread('/home/doombuggy_/Pictures/face_test.jpg')

# Detect faces
dets = detector(img, 1)

num_faces = len(dets)

# Loop through each face detected
for k, d in enumerate(dets):
    # Get the landmarks/parts for the face in box d.
    shape = sp(img, d)
    
    # Compute the 128D vector that describes the face in img identified by shape.
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    
    # Draw rectangles around detected faces
    x1 = d.left()
    y1 = d.top()
    x2 = d.right() 
    y2 = d.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2) 
    
    # Add text for number of faces detected
    cv2.putText(img, "Number of faces detected: {}".format(num_faces), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show image
cv2.imshow('image', img)
cv2.waitKey(0)

cv2.destroyAllWindows()
import face_recognition
import docopt
from sklearn import svm
import numpy as np
import os
import joblib
  
def train_faces(dir):
    encodings = []
    names = []
  
    # Training directory
    if dir[-1]!='/':
        dir += '/'
    train_dir = os.listdir(dir)
  
    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir(dir + person)
  
        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(
                dir + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)
  
            # If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image 
                # with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img + " can't be used for training")
  
    # Create and train the SVC classifier
    clf = svm.SVC(gamma ='scale', probability=True)
    clf.fit(encodings, names)
    joblib.dump(clf, "face_model.pkl")
    joblib.dump(encodings, "encoding.pkl")

def recognize(test):
    test_image = face_recognition.load_image_file(test)
  
    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    print("Number of faces detected: ", no)
  
    # Predict all the faces in the test image using the trained classifier
    print("Found:")
    clf=joblib.load("face_model.pkl")
    enc=joblib.load("encoding.pkl")
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        matches = face_recognition.compare_faces(enc, test_image_enc)
        faceDis = face_recognition.face_distance(enc, test_image_enc)
        print("matches",matches)
        print("faceDis",faceDis)
        print(matches[np.argmin(faceDis)])
        name = clf.predict([test_image_enc])
        score = clf.predict_proba([test_image_enc])
        print(*name)
        print(score)
  
def main():
    train_dir ='faces'
    test_image='face2.jpg'
    # train_faces(train_dir)
    recognize(test_image)
  
if __name__=="__main__":
    main()
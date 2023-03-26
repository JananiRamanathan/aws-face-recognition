import face_recognition
import docopt
from sklearn import svm
import numpy as np
import os
import joblib
  
def train_faces(dir):
	name = os.path.basename(dir)
	if dir[-1]!='/':
		dir += '/'
	train_dir = os.listdir(dir)

	for person_img in train_dir:
		if person_img.endswith('.jpg'):
			face = face_recognition.load_image_file(
				dir + "/" + person_img)
			face_bounding_boxes = face_recognition.face_locations(face)

			if len(face_bounding_boxes) == 1:
				face_enc = face_recognition.face_encodings(face)[0]
				append_data_to_pickle(name, face_enc)
			else:
				print(name + "/" + person_img + " can't be used for training")

def append_data_to_pickle(name, encoding):
	if (os.path.exists("data/models/names.pkl") and os.path.exists("data/models/encoding_model.pkl")):
		with open("data/models/names.pkl", "rb") as f:
			names = joblib.load(f)
		
		with open("data/models/encoding_model.pkl", "rb") as f:
			encodings = joblib.load(f)
	else:
		names = []
		encodings = []

	names.append(name)
	encodings.append(encoding)
	joblib.dump(names, "data/models/names.pkl")	
	joblib.dump(encodings, "data/models/encoding_model.pkl")



def recognize_face(test_img):
	test_image = face_recognition.load_image_file(test_img)

	enc=joblib.load("data/models/encoding_model.pkl")
	names=joblib.load("data/models/names.pkl")

	test_image_enc = face_recognition.face_encodings(test_image)[0]
	matches = face_recognition.compare_faces(enc, test_image_enc, tolerance=0.5)

	if True in matches:
		result = names[matches.index(True)]
		return result
	else:
		return ["Unable to recognize face"]
  

train_faces('data/train_faces/Janani')
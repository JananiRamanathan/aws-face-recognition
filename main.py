import cv2
import os
import re
cap = cv2.VideoCapture(0)

i=0
# # Deleting jpg files
# directory = 'faces'
# for filename in os.listdir(directory):
#     if filename.endswith('.jpg') and re.fullmatch("face\d{1,5}\.jpg", filename) == None:
#         print(os.path.join(directory, filename))
#         os.remove(os.path.join(directory, filename))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == False:
#         break
#     cv2.imwrite('faces/face'+str(i)+".jpg", frame)
#     i+=1
#     # if i == 3:
#     #     break
def crop_faces(filename, cropped_filename):
    print(filename)
    img = cv2.imread(filename)
  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('haarcascade.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        cv2.imwrite(cropped_filename, faces)
        
def capture_faces():
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        # cv2.imwrite('faces/face'+str(i)+".jpg", frame)
        cv2.imwrite('face'+str(i)+".jpg", frame)
        i+=1
        if i == 3:
            break

    cap.release()
    cv2.destroyAllWindows()

# directory = 'faces/santhosh'
# for filename in os.listdir(directory):
#     i+=1
#     if filename.endswith('.jpg'):
#         crop_faces(os.path.join(directory, filename), 'cropped_faces/Santhosh/face'+str(i)+'.jpg')
capture_faces()
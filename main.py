import cv2

cap = cv2.VideoCapture(0)

i=0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('faces/face'+str(i)+".jpg", frame)
    i+=1
    # if i == 3:
    #     break

cap.release()
cv2.destroyAllWindows()
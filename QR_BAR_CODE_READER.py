import cv2
import pyzbar.pyzbar as pyzbar              ##Decoder for Bar-code and QR-Code

cap=cv2.VideoCapture(0)

while True:
    _,img=cap.read()
    dec=pyzbar.decode(img)				##Continuosly reads the frames supplied if there is a available qr aor bar code then print it.

    for obj in dec:
        cv2.putText(img,str(obj.data),(10,130),cv2.FONT_HERSHEY_SIMPLEX,0.9,(200,2,246),3)
    cv2.imshow('image',img)
    if cv2.waitKey(1)==ord('q'):
        break
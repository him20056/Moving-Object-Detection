import cv2

first_frame=None
vid=cv2.VideoCapture(0)

while(True):
    retval,image=vid.read()

    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    imag=cv2.GaussianBlur(image,(21,21),0)

    if first_frame is None:
        first_frame=imag

    imag=cv2.absdiff(first_frame,imag)

    thres_fram=cv2.threshold(imag,30,255,cv2.THRESH_BINARY)[1]
    thres_fram=cv2.dilate(thres_fram, None, iterations=2)

    (_,cnts,_)=cv2.findContours(thres_fram.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour)<1000:
            continue
        (x,y,z,w)=cv2.boundingRect(contour)
        cv2.rectangle(image,(x,y),(x+z,y+w),(0,255,255),3)
    
    cv2.imshow("Him",image)
    cv2.imshow("Gupta",thres_fram)
    h=cv2.waitKey(1)
    if(h==ord('q')):
        break

vid.release()
cv2.destroyAllWindows()

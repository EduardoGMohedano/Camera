import numpy as np
import cv2

kernel_size = 3
cap = cv2.VideoCapture(0)

# define the list of boundaries HSV
boundaries = [([0, 100, 100], [10, 255, 255])]
boundariesHSV = [[106, 100, 100], [179, 255, 255]]
lower = [[0, 100, 100]]
upper = [[10, 255, 255]]

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame2 = cv2.medianBlur(frame2, kernel_size)
    hist = cv2.calcHist([frame2],[2],None,[16],[0,255])

    for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(frame2, lower, upper)
    output = cv2.bitwise_and(frame2, frame2, mask = mask)
    #using a delay
    cv2.waitKey(100)
    
    # Display the resulting frame
    cv2.imshow("images", np.hstack([frame, frame2]))
    #cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

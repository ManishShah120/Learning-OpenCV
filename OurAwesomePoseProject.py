import cv2
from PoseModule import poseDetector

cap = cv2.VideoCapture("videos/Yoga_Video.mp4")
detector = poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)

    # For marking the right elbow
    if len(lmList) != 0:
        # print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 7, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Detecting Right Elbow", img)  # For naming the OpenCV Panel Screen
    cv2.waitKey(1)

import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                   smooth_landmarks=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("PoseVideos/0.mp4")
pTime = 0

while True:
    success, img = cap.read()
    if success:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = pose.process(imgRGB)
        
        # print(results.pose_landmarks)
        if results.pose_landmarks:
            mpDraw = mp.solutions.drawing_utils
            mpDraw.draw_landmarks(img,
                                results.pose_landmarks,
                                mpPose.POSE_CONNECTIONS)
            
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm.x, lm.y, lm.z)
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
    
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
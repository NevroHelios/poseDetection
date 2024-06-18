import cv2
import mediapipe as mp
import time

class poseDetector():
    
    def __init__(self, mode = False,
                 smooth = True,
                 detectionCon = 0.5,
                 trackCon = 0.5) -> None:
        """
        Initializes a new instance of the `poseDetector` class.

        Args:
            mode (bool, optional): Specifies whether the pose detection should be performed in static image mode. 
                                   Defaults to False.
            smooth (bool, optional): Specifies whether to apply smoothing to the landmarks. Defaults to True.
            detectionCon (float, optional): Specifies the minimum confidence value for pose detection. 
                                            Defaults to 0.5.
            trackCon (float, optional): Specifies the minimum confidence value for pose tracking. Defaults to 0.5.

        Returns:
            None
        """
        
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw = True):
        """
        Finds the pose in an image and optionally draws the landmarks on the image.

        Args:
            img (numpy.ndarray): The input image in which to find the pose.
            draw (bool, optional): Whether to draw the landmarks on the image. Defaults to True.

        Returns:
            numpy.ndarray: The input image with the landmarks drawn, if draw is True.
        """
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,
                                           self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img
    
    def getPosition(self, img, draw = True):
        """
        Get the position of the landmarks in an image.

        Args:
            img (numpy.ndarray): The input image.
            draw (bool, optional): Whether to draw the landmarks on the image. Defaults to True.

        Returns:
            list: A list of lists containing the landmark ID, x-coordinate, y-coordinate, and z-coordinate.
        """
        lmList = []
        if self.results.pose_landmarks:   
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                
                lmList.append([id, cx, cy, cz])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    
        return lmList
    
def main():
    """
    Runs the main loop of the program, capturing video from a file and performing pose estimation on each frame.
    
    This function initializes a `poseDetector` object and starts a loop that reads frames from a video file using `cv2.VideoCapture`.
    For each frame, it calls the `findPose` method of the `poseDetector` object to find the pose in the frame and draw the landmarks on the frame.
    It then calls the `getPosition` method of the `poseDetector` object to get the position of the landmarks in the frame.
    The function prints the position of the landmarks if they exist.
    It also calculates and displays the frames per second (fps) of the video.
    
    The loop continues until the user presses the 'q' key or there is an error reading a frame from the video file.
    
    """
    cap = cv2.VideoCapture("PoseVideos/0.mp4")
    pTime = 0
    detector = poseDetector()
    
    while True:
        success, img = cap.read()
        if success:
            img = detector.findPose(img)
            lmList = detector.getPosition(img, draw=True) # set it to True to draw all
            if lmList: print(lmList)
            # cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
            
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            
            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)
            cv2.imshow("Image", img)
            
            if cv2.waitKey(1)  & 0xFF == ord('q'):
                break
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()
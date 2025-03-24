import cv2
import mediapipe as mp

class handDetector():


    def __init__(self,mode=False,maxHands=1,complexity=1,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.complexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils





    def findHands(self,img,draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)


        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img





    def findPosition(self,img,id = tuple(range(21)),draw=False,color=(255,0,0)):
        if isinstance(id,int):
            id = (id,)
        lmListAll = []
        if self.results.multi_hand_landmarks:

            for myHand in self.results.multi_hand_landmarks:  # Loop through all detected hands
                lmList = []

                for ID, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([ID, cx, cy])
                    if draw and ID in id :
                        cv2.circle(img, (cx, cy), 7, color, cv2.FILLED)
                lmListAll.append(lmList)


        return lmListAll

import cv2
import mediapipe as mp
import time
import numpy as np
from tensorflow.keras.models import load_model
import time

xp, yp = 0, 0
class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        

    def findHands(self, img, model, draw=True, seq_length = 30, seq=[], action_seq=[]):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        actions = ['one', 'standby', 'input']
        #seq_length = 30
        #seq = []
        #action_seq = []
        action = 0
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                joint = np.zeros((21,4))
                for j, lm in enumerate(handLms.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
            
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree
                d = np.concatenate([joint.flatten(), angle])
                seq.append(d)

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                
                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                
                y_pred = model.predict(input_data).squeeze()
                
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.7:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                if action :
                    cv2.putText(img, 
                                f'{this_action.upper()}', 
                                org=(int(handLms.landmark[0].x * img.shape[1]), int(handLms.landmark[0].y * img.shape[0] + 20)), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale=1, 
                                color=(255, 0, 0), 
                                thickness=2)
        action_seq3 = action_seq[-4:-1]
        #print(action_seq3)
        return img, action, action_seq3

    def findPosition(self, img, handNo=0, draw = True):
        lmList = []
        pen_on = 0
        erase_on = 0
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
            
                if draw:
                    if id == 8:
                        len_8 = ((lmList[8][1]-lmList[5][1])**2 + (lmList[8][2]-lmList[5][2])**2)
                        if len_8 > 3500:
                            cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
                            pen_on = 1
                    if id == 12:
                        len_12 = ((lmList[12][1]-lmList[9][1])**2 + (lmList[12][2]-lmList[9][2])**2)
                        if len_12 > 3500:
                            cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
                            pen_on = 0
                            erase_on = 1

        return lmList, pen_on, erase_on

    def drawCanvas(self, img, imgCanvas, lmList, pen_on, erase_on, action, mode = 'pen') :
        x1, y1, x1p, y1p, x2, y2, x2p, y2p= 0, 0, 0, 0, 0, 0, 0, 0
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

        if (pen_on == 1) & (action == 'one'):
            if x1p == 0 and y1p == 0:
                x1p, y1p = x1, y1
            
            cv2.line(img, (x1p,y1p), (x1,y1), (255,0,0), 3)
            cv2.line(imgCanvas, (x1p,y1p), (x1,y1), (255,0,0), 30)
            x1p, y1p = x1, y1

        elif erase_on == 1:
            if x1p == 0 and y1p == 0 and x2p == 0 and y2p == 0:
                x1p, y1p = x1, y1
                x2p, y2p = x2, y2
            
            cv2.line(img, (x1p,y1p), (x1,y1), (0,0,0), 3)
            cv2.line(img, (x2p,y2p), (x2,y2), (0,0,0), 3)
            cv2.line(imgCanvas, (x1p,y1p), (x1,y1), (0,0,0), 50)
            cv2.line(imgCanvas, (x2p,y2p), (x2,y2), (0,0,0), 50)
            x1p, y1p = x1, y1
            x2p, y2p = x2, y2

        #print(x1, y1, xp, yp)    

        return imgCanvas
    
    def save_image(self, imgCanvas, action_seq3):
        if len(action_seq3) > 3:
                    
            if (action_seq3[-1] == 'input') & (action_seq3[-2] == 'stanby') & (action_seq3[-3] == 'stanby'):
                imgCanvas = 255 - imgCanvas
                cv2.imwrite('image/image01.png', imgCanvas)
                time.sleep(0.5)
                imgCanvas = np.zeros((480, 640, 3), np.uint8)
        return imgCanvas

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 2)
    # def change_res(width, height):
    #     cap.set(3, width)
    #     cap.set(4, height)
    
    # change_res(640,480)

    detector = handDetector()
    while True:
        success, img = cap.read()
        
        img = cv2.flip(img, 1)

        img = detector.findHands(img, draw=True)
        lmList, _ = detector.findPosition(img, draw=True)
        # if len(lmList) !=0:
        #     print(lmList[8], lmList[5] )
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Image", cv2.resize(img, (320, 240)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        
if __name__ == "__main__":
    main()
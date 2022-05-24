import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()
imgCanvas = np.zeros((480, 640, 3), np.uint8)

## gesture 및 한글인식 모델
model = load_model('models/model.h5')
model_KCL = load_model('models/Korean_commercial_letter_2350_20epoch_test2.h5')

## KCL 데이터(라벨) 저장
file_dir_KCL = 'KCL_data/'
file_name = 'Korean_commercial_letters_2350.txt'
with open(file_dir_KCL + file_name, "r", encoding='cp949') as file: ## encoding='cp949'(or 'euc-kr) ==> 한글 인코딩 
    strings = file.readlines()
KCL = list(*strings)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    img, action, action_seq3 = detector.findHands(img, model, draw=True)
    lmList, pen_on, erase_on = detector.findPosition(img, draw=True)
    imgCanvas = detector.drawCanvas(img, imgCanvas, lmList, pen_on, erase_on, action)

    # if len(lmList) !=0:
    #     print('') # pen_on, lmList[8])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    
    _, imgInv = cv2.threshold(imgGray, 25, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    
    #img = cv2.addWeighted(img, 1, imgCanvas, 2, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)
    if len(action_seq3) >= 3:
        if (action_seq3[2] == 'input') & (action_seq3[0] == 'standby') & (action_seq3[1] == 'standby'):
            # imgCanvas = 255 - imgCanvas
            # imgCanvas = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            # imgCanvas = cv2.threshold(imgCanvas, 25, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite('image/image01.png', imgInv)
            
            time.sleep(0.5)

            img_KCL = Image.open("image/image01.png").convert('L')

            ## 모델에 맞는 형식으로 변환
            img_resize = img_KCL.resize((28, 28))
            imgArray = np.array(img_resize)
            imgArray = (255-imgArray)/255
            imgArray = imgArray[np.newaxis, :]
            
            ## 예측
            pred = model_KCL.predict(imgArray)
            #print(np.argmax(pred))
            print(KCL[np.argmax(pred)])
                        
            ## 출력이미지 초기화
            imgCanvas = np.zeros((480, 640, 3), np.uint8)

    #imgCanvas = detector.save_image(imgCanvas, action_seq3)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
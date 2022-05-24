import numpy as np
from PIL import Image, ImageChops
from tensorflow.keras.models import load_model

## 손글씨 예측 모델 로딩
model_KCL = load_model('models/Korean_commercial_letter_2350_20epoch.h5')

## 라벨 데이터 로딩
file_dir_KCL = 'KCL_data/'
file_name = 'Korean_commercial_letters_2350.txt'
with open(file_dir_KCL + file_name, "r", encoding='cp949') as file: ## encoding='cp949'(or 'euc-kr) ==> 한글 인코딩 
    strings = file.readlines()
KCL = list(*strings)

def crop(im):
    background = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, background)
    diff = ImageChops.add(diff, diff, 2.0, -35)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        print('Failure!')
        return

## 예측 이미지 로딩
img_KCL = Image.open("image/image01.png").convert('L')

## 이미지 전처리
img_KCL = crop(img_KCL)

## 전처리 이미지 저장
img_KCL.save("image/image02.png",'png')

## 모델에 맞는 형식으로 변환
img_resize = img_KCL.resize((28, 28))
imgArray = np.array(img_resize)
imgArray = (255-imgArray)/255
imgArray = imgArray[np.newaxis, :]

## 예측
pred = model_KCL.predict(imgArray)
#print(np.argmax(pred))
print(KCL[np.argmax(pred)])
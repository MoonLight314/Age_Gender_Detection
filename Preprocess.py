import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import cv2


MODEL_FILE = "opencv_face_detector_uint8.pb"
CONFIG_FILE = "opencv_face_detector.pbtxt"
SIZE = 300
CONFIDENCE_FACE = 0.9
MARGIN_RAIO = 0.2


# 선글라스 , 모자 , 모자 + 뿔테안경을 착용한 경우는 모두 사용하지 않는다.
use_accessaries = ['S001' , 'S002' , 'S003']

# 적절한 밝기의 조명 사진만 사용한다
use_lights = ['L1' , 'L2' , 'L3' , 'L4' , 'L8' ,'L9' ,'L12' ,'L13' ,'L22']

# 얼굴이 제대로 보이지 않는 카메라 위치는 사용하지 않는다.
use_cam_pos = ['C4','C5','C6','C7','C8','C9','C10','C19','C20']

# 찡그린 표정의 사진은 사용하지 않는다.
use_looks = ['E01','E02']


dir_path = "../Dataset/High_Resolution"
data_file_path = []

def GetTrainDataFileList():
    for (root, directories, files) in tqdm(os.walk(dir_path)):
        for file in files:
            
            if '.jpg' in file:
                append = False
                
                file_path = os.path.join(root, file)
                for u in use_accessaries:
                    if u in str(file_path):
                        append = True
                        break

                if append:
                    append = False
                    for u in use_lights:
                        if u in str(file_path):
                            append = True
                            break

                if append:
                    append = False
                    for u in use_cam_pos:
                        if u in str(file_path):
                            append = True
                            break

                if append:
                    append = False
                    for u in use_looks:
                        if u in str(file_path):
                            append = True
                            break

                if append:
                    data_file_path.append( file_path )


    print( len(data_file_path) )

    meta_data = pd.DataFrame( data_file_path , columns=['file_path'])
    meta_data.to_csv("meta_data_K-Face.csv",index=False)

GetTrainDataFileList()

meta_data = pd.read_csv("meta_data_K-Face.csv")

file_path = meta_data['file_path'].tolist()
len(file_path)

filename = []
left_list = []
right_list = []
top_list = []
bottom_list = []

net = cv2.dnn.readNetFromTensorflow( MODEL_FILE , CONFIG_FILE )

for file in tqdm(file_path):

    img = cv2.imread(file)
    rows, cols, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1.0, (SIZE, SIZE))

    net.setInput(blob)
    detections = net.forward()

    for detection in detections[0, 0]:

        score = float(detection[2])

        if score > CONFIDENCE_FACE:

            if detection[3] >= 1.00 or detection[4] >= 1.00 or detection[5] >= 1.00 or detection[6] >= 1.00 or detection[3] <= 0 or detection[4] < 0 or detection[5] <= 0 or detection[6] <= 0:
                filename.append(np.NaN)
                left_list.append( np.NaN )
                right_list.append( np.NaN )
                top_list.append( np.NaN )
                bottom_list.append( np.NaN )

            else:
                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)

                left = left - int((right - left) * MARGIN_RAIO)
                top = top - int((bottom - top) * MARGIN_RAIO)
                right = right + int((right - left) * MARGIN_RAIO)
                bottom = bottom + int((bottom - top) * MARGIN_RAIO / 2)

                if left < 0:
                    left = 0

                if right > cols:
                    right = cols

                if top < 0:
                    top = 0

                if bottom > rows:
                    bottom = rows
                    
                filename.append(file)
                left_list.append( left )
                right_list.append( right )
                top_list.append( top )
                bottom_list.append( bottom )

coor = pd.DataFrame( list(zip(filename , left_list , right_list , top_list , bottom_list)) , columns=['file_path' , 'left' , 'right' , 'top' , 'bottom'] )

coor.head()

coor.info()

coor.to_csv("coor.csv" , index=False)

def get_ID(file_path):
    ID = str(file_path)[27:35]
    return int(ID)

coor.head()

coor['ID'] = coor['file_path'].apply(get_ID)

coor.head()

coor.info()

additional_info = pd.read_csv("KFace_data_information_Folder1_400.csv",encoding='CP949')
additional_info = additional_info[['ID','연령대','성별']]
print(additional_info.head())
print(additional_info.info())

merged_meta_data = pd.merge(coor,
         additional_info,
         how='left',
         on='ID'
        )

merged_meta_data.tail()

merged_meta_data.to_csv('meta_data_face_coor_K-Face.csv' , index=False)
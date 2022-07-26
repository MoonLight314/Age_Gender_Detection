from numpy.lib.function_base import average
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D , BatchNormalization , Dropout , Dense
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint , LearningRateScheduler , EarlyStopping

BATCH_SIZE = 32
DROP_OUT_RATE = 0.2

dataset_info = pd.read_csv("meta_data_face_coor_K-Face.csv")
dataset_info

gender = dataset_info['성별'].tolist()

le_gender = LabelEncoder()
le_gender.fit(gender)
print(le_gender.classes_)

age_band = dataset_info['연령대'].tolist()

le_age_band = LabelEncoder()
le_age_band.fit(age_band)
print(le_age_band.classes_)


data_file_path = dataset_info[['file_path' , 'left' , 'right' , 'top' , 'bottom' , '연령대' , '성별']]

dataset_info['merged_class'] = dataset_info['연령대']+dataset_info['성별']

dataset_info['merged_class']

dataset_info['merged_class'].value_counts()

merged_class = dataset_info['merged_class'].tolist()


le_merged_class = LabelEncoder()
le_merged_class.fit(merged_class)
print(le_merged_class.classes_)
merged_class = le_merged_class.transform(merged_class)
merged_class = tf.keras.utils.to_categorical(merged_class , num_classes=8)


file_path_train, file_path_val, y_train, y_val = train_test_split(data_file_path, merged_class, 
                                                                  test_size=0.25, 
                                                                  random_state=777, 
                                                                  stratify = merged_class)


print( len(file_path_train) , len(y_train) , len(file_path_val) , len(y_val) )

file_path_train


train_left = file_path_train['left'].tolist()
train_right = file_path_train['right'].tolist()
train_top = file_path_train['top'].tolist()
train_bottom = file_path_train['bottom'].tolist()
train_file_path = file_path_train['file_path'].tolist()


age_band = file_path_train['연령대'].tolist()

age_band = le_age_band.transform(age_band)
train_age_band = tf.keras.utils.to_categorical(age_band , num_classes=4)
train_age_band


gender = file_path_train['성별'].tolist()

gender = le_gender.transform(gender)
train_gender = tf.keras.utils.to_categorical(gender , num_classes=2)
train_gender

val_left = file_path_val['left'].tolist()
val_right = file_path_val['right'].tolist()
val_top = file_path_val['top'].tolist()
val_bottom = file_path_val['bottom'].tolist()
val_file_path = file_path_val['file_path'].tolist()


age_band = file_path_val['연령대'].tolist()

age_band = le_age_band.transform(age_band)
val_age_band = tf.keras.utils.to_categorical(age_band , num_classes=4)
val_age_band

gender = file_path_val['성별'].tolist()

gender = le_gender.transform(gender)
val_gender = tf.keras.utils.to_categorical(gender , num_classes=2)
val_gender


def load_image( image_path , left , right , top , bottom , label_age_band , label_gender ):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)   
    img = tf.image.crop_to_bounding_box( img , top , left, bottom - top , right - left )
    
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    
    return img , (label_gender , label_age_band)


train_dataset = tf.data.Dataset.from_tensor_slices( (train_file_path , 
                                                     train_left , 
                                                     train_right , 
                                                     train_top , 
                                                     train_bottom , 
                                                     train_age_band,
                                                     train_gender) )

val_dataset = tf.data.Dataset.from_tensor_slices( (val_file_path , 
                                                   val_left , 
                                                   val_right , 
                                                   val_top , 
                                                   val_bottom ,
                                                   val_age_band,
                                                   val_gender) )


train_dataset = train_dataset.shuffle(buffer_size=len(train_file_path)).map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_dataset.shuffle(buffer_size=len(val_file_path)).map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

input = tf.keras.Input(shape=(224, 224, 3), name="input")

r = tf.keras.applications.resnet.ResNet50(
    weights=None,
    input_shape=(224, 224, 3),
    include_top=False)(input)

g = GlobalAveragePooling2D()(r)
g = Dropout(DROP_OUT_RATE)(g)
g = BatchNormalization()(g)
g = Dense(128,activation="relu")(g)
g = Dropout(DROP_OUT_RATE)(g)
g = BatchNormalization()(g)


output_gender = Dense(2, activation='softmax' , name="output_gender")(g)

output_age_band = Dense(4, activation='softmax' , name="output_age_band")(g)

model = tf.keras.Model(
    inputs=input,
    outputs=[output_gender, output_age_band],
)


model.summary()


initial_learning_rate = 0.01

def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * np.math.exp(-k*epoch)

lr_scheduler = LearningRateScheduler(lr_exp_decay, verbose=1)


log_dir = os.path.join('Logs')
CHECKPOINT_PATH = os.path.join('CheckPoints_K-Face_Gender_Age_Band_F1_Score')
tb_callback = TensorBoard(log_dir=log_dir)


cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, 
                     monitor='val_output_age_band_accuracy',
                     #monitor='val_F1_metric',
                     save_best_only = True,
                     verbose = 1)

es = EarlyStopping(monitor = 'val_output_age_band_loss', patience = 2, mode = 'auto')

F1_metric = tfa.metrics.F1Score(num_classes=2 , average=None)


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),    
    
    loss={'output_gender':'binary_crossentropy',
          'output_age_band':'categorical_crossentropy'},    
    
    metrics={'output_gender':'accuracy' , 
             'output_age_band' : 'accuracy'}
)


hist = model.fit(train_dataset,
                 validation_data=val_dataset,
                 callbacks=[lr_scheduler , cp , tb_callback , es],
                 epochs = 10,
                 verbose = 1
                )

plt.plot(hist.history['output_gender_accuracy'])
plt.plot(hist.history['val_output_gender_accuracy'])
plt.title('Gender Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(hist.history['output_age_band_accuracy'])
plt.plot(hist.history['val_output_age_band_accuracy'])
plt.title('Age Band Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
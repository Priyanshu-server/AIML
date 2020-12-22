import cv2

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0],True)

model = load_model('SceneDetection.h5')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _,frame = cap.read()
    cv2.rectangle(frame,(45,380),(200,418),[0,0,0],-1,cv2.LINE_AA)
    new_frame = cv2.resize(frame,(64,64)) #[64,64]
    new_frame = np.expand_dims(new_frame,0)
    prediction = model.predict(new_frame)

    try:
        prediction_list = prediction.reshape(-1,).tolist()
        prediction_list = prediction_list.index(1.0) #[1.0,0,0,0]
        class_indices = {0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}
        prediction = class_indices[prediction_list]
        print(prediction)

        cv2.putText(frame,prediction,(50,407),cv2.FONT_HERSHEY_SIMPLEX,1,[0,255,0],1,cv2.LINE_AA)
    except:
        cv2.putText(frame,"No prediction",(50,407),cv2.FONT_HERSHEY_SIMPLEX,1,[255,255,255],1,cv2.LINE_AA)


    cv2.imshow('window',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

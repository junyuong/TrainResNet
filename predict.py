#학습된 모델 불러와서 데이터 확인하기 위한 코드

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
json_file = open("resnet_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("240x240_ResNet_weight.h5")
loaded_model.save('teem.hdf5')
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


def callImage(str):

    img = image.load_img(str, target_size=(224, 224))
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.

    pred = loaded_model.predict(img_tensor)
    print(np.max(pred))

    f = open("label.txt", 'r')
    line = f.readline()
    f.close()
    for i in range(0,2350):
        if (pred[0][i]==np.amax(pred)):
            inde = i
            break
    print(line[inde])
    print(inde)


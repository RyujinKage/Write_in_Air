import numpy as np
from keras.models import load_model
from skimage import transform

# dimensions of images
img_width, img_height = 32, 32

# load model
model = load_model('../weights/final_model_weight.h5')


label = ['+','-','0','1','2','3','4','5','6','7','8','9']

def finalPreProcess(img_):
   np_image = img_
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (img_width, img_height, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def prediction(img):
    imag = finalPreProcess(img)
    predict_classes = model.predict(imag)
    predicted_class = np.argmax(predict_classes)  #find the class with max prob
    return label[predicted_class] # return corresponding label for the class

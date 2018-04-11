import numpy as np
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.optimizers import SGD
import cv2

np.random.seed(9) 

def VGG_16(weights_path=None): 
    model = Sequential()  
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',activation='relu',kernel_initializer='uniform',name='block1_conv1'))  
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform',name='block1_conv2'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform',name='block2_conv1'))  
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform',name='block2_conv2'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform',name='block3_conv1'))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform',name='block3_conv2'))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform',name='block3_conv3'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform',name='block4_conv1'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform',name='block4_conv2'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform',name='block4_conv3'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform',name='block5_conv1'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform',name='block5_conv2'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform',name='block5_conv3'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Flatten())  
    model.add(Dense(4096,activation='relu',name='fc1'))  
    model.add(Dropout(0.5))  
    model.add(Dense(4096,activation='relu',name='fc2'))  
    model.add(Dropout(0.5))  
    model.add(Dense(1000,activation='softmax',name='predictions')) 
    
    if weights_path:
        model.load_weights(weights_path)
    return model

if __name__ == "__main__":
    im = cv2.resize(cv2.imread('I:/Face_Liveness_Code/PythonCode/src/cat.jpg'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_16('I:/VGG/VGG_kaggle/weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5')#网络参数路径
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print (np.argmax(out))

import numpy as np
from flask import Flask,request,jsonify,render_template,url_for,send_from_directory
import pickle
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

#PreProcessing training Set
#transformation of images to avoid overfitting (Feature Scaling)
#train_datagen = ImageDataGenerator(
#        rescale=1./255,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True)
#train_df = train_datagen.flow_from_directory(
#        'datasett/training_set',
#        target_size=(64,64),
#        batch_size=32,
#        class_mode='binary')

#Preprocessing the Test Set
#test_datagen = ImageDataGenerator(rescale=1./255)
#test_df = test_datagen.flow_from_directory(
#        'datasett/test_set',
#        target_size=(64, 64),
#        batch_size=32,
#        class_mode='binary')

#Initialisation the CNN
#cnn = tf.keras.models.Sequential()

#Convolution
#cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=(64, 64, 3)))
#Pooling
#cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#Second Convolution Layer
#cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
#cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#Flattening
#cnn.add(tf.keras.layers.Flatten())
#Full Connection
#cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
#cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
#cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
#cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#cnn.fit(x=train_df,validation_data=test_df,epochs=5)

#cnn.save_weights("model.h5")

#cnn.load_weights('model.h5')
#cnn.save('model.pkl')

#mm._make_predict_function()
#graph = tf.get_default_graph()

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/home', methods=['POST'])
def home():
	mm = tf.keras.models.load_model('model.pkl')
	#print(mm.summary())
	global COUNT
	img = request.files['image']
	img.save('static/{}.jpg'.format(COUNT))
	test_image = image.load_img('static/{}.jpg'.format(COUNT),target_size=(64,64))    #img_arr = img_arr / 255.0
	#img_arr = img_arr.reshape(1,64,64,3)
	print(test_image)
	test_image = image.img_to_array(test_image)
	print(test_image)
	test_image = np.expand_dims(test_image,axis=0)
	print(test_image)
	result = mm.predict(test_image)
	COUNT+=1
	print(result)

	if result[0][0] == 1:
		prediction = 'dog'
	else:
		prediction = 'cat'

	print(prediction)
	return render_template('prediction.html', data=prediction)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)

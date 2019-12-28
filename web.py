from flask import Flask , render_template , request
import cv2
import os
import numpy as np
import keras 
from keras.models import model_from_json
import tensorflow
import pandas as pd

app = Flask(__name__ , static_folder = 'static')


def read(id):
	path = r'J:\\dataset\\presentation\\Project\\static'
	path = path.strip("‪u202a")
	id = str(id)+'.mp4'
	cap = cv2.VideoCapture(os.path.join(path,id))
	file = []
	while(cap.isOpened()):
		ret , frame = cap.read()
		if(ret == False):
			break
		frame = cv2.resize(frame , (60,60))
		file.append(frame)
	cap.release()
	cv2.destroyAllWindows()
	file = np.array(file)
	return file

def preprocess(file):
	d = len(file)//60
	video = []
	for j in range(0,len(file)):
		video.append(file[j])
		j+=d;
		if(len(video)==60):
			break
	return video

model_path = '‪J:\\dataset\\model_archi.json'
json_file = open(model_path.strip("‪u202a"), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
weights = '‪J:\\dataset\\model_w.h5'
model.load_weights(weights.strip("‪u202a"))


def model_predict(file):
	
	pred = model.predict([file])

	filename = r'‪J:\\dataset\\training_data.csv'

	data = pd.read_csv(filename)
	pred = data[id]['target']
	actual = data[id]['target']

	return ( actual , pred)

@app.route('/')
def home():
	return render_template('menu.html')

@app.route('/file/<int:id>')
def file(id):
	file = read(id)
	file = preprocess(file)
	(actual , predicted) = model_predict(file)
	id = str(id)+'.mp4'
	return render_template('menu.html' , actual = actual , predicted = predicted , filename = id )

if __name__ == '__main__':
	app.run(  debug = True )
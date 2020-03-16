import os
import cv2
from sklearn.model_selection import train_test_split
import shutil


data_path = '/home/okhrimenko/Work/furniture/classifier/data/'
images_ext = ['jpg', 'png', 'jpeg']
train_path = 'train_path'
test_path = 'test_path'
test_size = 0.1    # percentage of data to create a test sample


fnames = []
labels = []
for folder in os.listdir(data_path):
	current_path = os.path.join(data_path, folder)
	for el in os.listdir(current_path):
		fname = os.path.join(current_path, el)
		img = cv2.imread(fname)

		if img is not None and img.size != 0:
			fnames.append(fname)
			labels.append(folder)
		else:
			print(fname)

print(len(fnames))			

X_train, X_test, y_train, y_test = train_test_split(fnames, labels, test_size=test_size, stratify=labels)

for el in X_train:
	train_cls = el.split('/')
	dest_train_folder = os.path.join(train_path, train_cls[-2])
	dest_train_name = os.path.join(train_path, train_cls[-2], train_cls[-1])
	if os.path.exists(dest_train_folder) is False:
	    os.makedirs(dest_train_folder)
	shutil.copy(el, dest_train_folder)

for el in X_test:
	test_cls = el.split('/')
	dest_test_folder = os.path.join(test_path, test_cls[-2])
	dest_test_name = os.path.join(test_path, test_cls[-2], test_cls[-1])
	if os.path.exists(dest_test_folder) is False:
	    os.makedirs(dest_test_folder)
	shutil.copy(el, dest_test_folder)

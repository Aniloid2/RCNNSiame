import sys
import os
import numpy as np
from shutil import copyfile

directory = '../Data'



Save_dir_t = '../Gate_sis/Training_data'

try:
	os.mkdir(Save_dir_t)
except Exception as e:
	print (e)


Save_dir_v = '../Gate_sis/Validation_data'
try:
	os.mkdir(Save_dir_v)
except Exception as e:
	print (e)


Save_dir = '../Gate_sis/'


# FIRST HALF OF THE ARRAY IS THE TRAINING< SECOND HALF ITS THE VALIDATION

angles = ['Silhouette_000-00','Silhouette_030-00', 'Silhouette_015-00', 'Silhouette_045-00' ] # e.g for training(00001(00,30),00002(00,30)), validation(00001(14,45),00002(15,45))


# get all the 200 individuals
all_individuals = []
all_individuals_path = []
for angle in angles:
	path_join_dir_angle = os.path.join(directory, angle)
	list_indiviuals = os.listdir(path_join_dir_angle)[:200] # Take a maximum of 200 indiviuals from the data directory

	individuals_path = []
	for i in range(len(list_indiviuals)):
		individuals_path.append(os.path.join(path_join_dir_angle, list_indiviuals[i]))

	all_individuals.append(list_indiviuals)
	all_individuals_path.append(individuals_path)




wanted_individuals = []
wanted_individuals_path = []
i = 0

# make sure each angle has the same individual, reject the sample if it dosn't.
requirements = True
setimo = {}
for angle in range(len(all_individuals)):
	for j in range(len(all_individuals[angle])):
		if all_individuals[angle][j] not in setimo:
			setimo[all_individuals[angle][j]] = [1]*(len(all_individuals)+1)
		else:
			setimo[all_individuals[angle][j]][0] += 1 


		setimo[all_individuals[angle][j]][angle+1] = len(os.listdir(all_individuals_path[angle][j]))


# make sure each individual has at least 50 images
ammount_of_images = []
for ind, data in setimo.items():
	ammount_of_images = data[1:]
	min_amm_img = min(ammount_of_images)
	if min_amm_img >50 and data[0] == len(all_individuals):
		wanted_individuals.append(ind)
		wanted_individuals_path.append(os.path.join(os.path.join(  directory   ,   angles[angle]   ),    ind     ))

# For training data
for i in angles[:2]:



	for ind in wanted_individuals:
		ind_path = os.path.join(Save_dir_t, ind)
		try:
			os.mkdir(ind_path)
		except Exception as e:
			print (e)
		angle_images = os.path.join(directory, i) # where we find the images
		ind_images = os.path.join(angle_images, ind) # the individual image path
		ind_all_angles_path = os.path.join(ind_path, i) # path we want to put the images in
		try:
			os.mkdir(ind_all_angles_path)
		except Exception as e:
			print (e)
		pictures = os.listdir(ind_images)[30:50]
		for pic in pictures:
			from_image = os.path.join(ind_images,pic)
			put_images_in = os.path.join(ind_all_angles_path,pic)
			print (from_image, put_images_in)

			copyfile(from_image, put_images_in)


# For the validation data
for i in angles[2:]:
	for ind in wanted_individuals:
		ind_path = os.path.join(Save_dir_v, ind)
		try:
			os.mkdir(ind_path)
		except Exception as e:
			print (e)
		angle_images = os.path.join(directory, i) # where we find the images
		ind_images = os.path.join(angle_images, ind) # the individual image path
		ind_all_angles_path = os.path.join(ind_path, i) # path we want to put the images in
		try:
			os.mkdir(ind_all_angles_path)
		except Exception as e:
			print (e)
		pictures = os.listdir(ind_images)[30:50]
		for pic in pictures:
			from_image = os.path.join(ind_images,pic)
			put_images_in = os.path.join(ind_all_angles_path,pic)
			print (from_image, put_images_in)

			copyfile(from_image, put_images_in)


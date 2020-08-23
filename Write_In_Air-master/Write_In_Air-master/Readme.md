# WRITING IN AIR

> Human Computer Interaction: It is the interaction between humans and computers. And it is one of the most important aspects of Human Computer Interaction-Gesture Recognition as it has gained massive popularity in the past decade. Writing in Air is one such gesture recognition problem which has been previously solved using some complex devices like depth and motions sensors, using external objects for fingertip detection such as wearing gloves or using some markers etc. This project proposes an effective and easy to use approach for writing in air using just the index finger's fingertip. This system uses an ordinary computer webcam to track the trajectory of the fingertip. A CNN model is used for object detection to classify the trajectory into 12 classes, with 10 being digits(0-9) and 2 being mathematical operators i.e. both addition and subtraction. Finally, these predictions are used for evaluating the mathematical expression.

## Files Details
##### 1. src :  This directory contains the source code of this project.
* main.py : It Contains the main program for finger detection to evaluating the expression.
* predict.py : It Contains the driver code for predictions of trajectory using the trained models.
* CNNModel.ipynb : It Contains the Code for training the CNN model.
##### 2 .weights : This directory contains the trained weigths of the model.
* final_weights.h5 : It Contains trained weigths of the model.
* final_model_weight.h5 : It Contains trained weights of the model.
##### 3. data.zip : This directory contains the dataset for the training of model.
##### 4. outputs : This folder contains the output of the program in video and image format.
##### 5. requirements.txt : This file contains all the requirements needed for the system.

## How To Use this Project.

### While using in Ubuntu

1. Clone the repository from github 
```sh
$ git clone https://github.com/RyujinKage/Write_in_Air.git
```
2. Create a Python virtual invironment
```sh
$ python3 -m venv Write_in_Air
$ cd Write_in_Air
$ source bin/activate
```
3. Install the requirements by using requirement.txt
```sh
$ pip3 install -r requirements.txt 
```
4. Run the program
```sh 
$ python3 main.py
```
4. Keep your hand in the generated rectangle and start writing using fingertip.

### While using in Windows
Install the requirement provided in requirements.txt in Anaconda and run the program. If in case the versions are to be downgrade then kindly create another environment for this as it will be easy.

### ### `Special Attention: While using this code please use a Clear Backgroud with enough Bright Lighting conditions(or use some sheets to create a clear background).`

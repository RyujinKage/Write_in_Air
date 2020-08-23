# WRITE IN AIR

> Human Computer Interaction is the interaction between humans and computers. One of the most important parts of human computer interaction-gesture recognition has gained massive popularity in the past decade. Writing in Air is one such gesture recognition problem which has been traditionally solved using complex devices like depth and motions sensors and using external objects for fingertip detection. This paper proposes an effective approach for writing in air using just the index fingertip. Our system uses an ordinary computer webcam to track the fingertip trajectory. A CNN model does object detection to classify the trajectory into 12 classes, 10 digits and 2 mathematical operators for addition and subtraction. Finally, these predictions are used for mathematical expression evaluation.

## About the files
##### 1. src :  this directory contains source code
* main.py : Contains main program for finger detection to expression evaluation
* predict.py : Contains driver code for predictions of trajectory using trained model
* CNNModel.ipynb : Code for training the CNN model
##### 2 .weights : This directory contains trained weigths and model
* final_weights.h5 : Contains trained weigths of the model
* final_model_weight.h5 : Contains trained weights and model
##### 3. data.zip
##### 4. outputs : This folder contains the output of our program in video format and image format
##### 5. requirements.txt : This file contains all the requirements

## How To Use

### On Ubuntu

1. clone the repository from github 
```sh
$ git clone https://github.com/Rajpra786/Write_In_Air.git
```
2. Create a Python vertual invironment
```sh
$ python3 -m venv Write_In_Air
$ cd Write_In_Air
$ source bin/activate
```
3. Install requirements by using requirement.txt
```sh
$ pip3 install -r requirements.txt 
```
4. Run the program
```sh 
$ python3 main.py
```
4. Keep your hand in the given rectangle and start writing

### On Windows
Install the requirement provided in requirements.txt in Anaconda and run the program.

### ### `Note: For this code to work properly, clear backgroud and bright lighting is necessary.`

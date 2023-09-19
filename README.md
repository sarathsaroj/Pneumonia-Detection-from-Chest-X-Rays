# Pneumonia-Detection-from-Chest-X-Rays

A convolutional neural network (CNN) is a type of deep learning artificial neural network commonly used for image processing tasks. In this section, I will load a pretrained network, finetune it for a new task, and evaluate it to achieve a good image classification accuracy on the Chest X-Ray. The task is to classify the X-Ray as Pneumonia or Normal

The code involves 2 parts:
1.	Load a pretrained ResNet34
2.	Replace the last layer of the network so that the output dimension matches the number of image classes (2). 

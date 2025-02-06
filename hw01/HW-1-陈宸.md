# **HW-1-陈宸**

## **Report**

**1.The background and problems to be solved by DNN** 

​	LeNet and AlexNet are both concrete deep neural network (DNN) architectures that are part of convolutional neural networks (CNN). They were proposed at different time points and achieved significant success in their respective application fields.

​	LeNet was first developed by Yann LeCun and his colleagues in 1998,which was one of the earliest convolutional neural networks. According to his paper,it was designed to recognize handwritten digits, specifically for the MNIST dataset. This innovation aimed to automate the process of digit recognition, which was crucial for applications like postal mail sorting and bank check processing. It solved the problem related to feature extraction and pattern recognition in images, which were significant hurdles in the field of computer vision at the time.

​	AlexNet was introduced by Alex Krizhevsky and his team in 2012 intended to participate in the ImageNet Large Scale Visual Recognition Challenge. It demonstrated the potential of deep learning for large-scale image classification tasks, also addressed the challenge of  classifying millions of high-resolution images into 1000 different classes, significantly improving the state-of-the-art in image recognition.

**2.The difference between Layer,activation etc.**

​	**Layers: **According to paper, LeNet-5 consists of 7 layers,  including 3 convolutional layers, 2 subsampling layers, 1 full connection and 1 output layers. According to the code from github, it consists 10 layers.

​	AlexNet consists of 8 layers,  including 5 convolutional layers, 2 subsampling layers and 1 output layer.

​	**Activation Functions**: LeNet uses the sigmoid activation function. AlexNet utilizes the ReLU (Rectified Linear Unit) activation function.

​	**Pooling**: LeNet: Employs average pooling.AlexNet: Uses max pooling.

**3.The result of running LeNet code**

After 5000 iteration, the accuracy is 0.968750, The loss is 4.755514 

**4.The parameters of LeNet and AlexNet**

​	According to paper, LeNet-5 has approximately 60,000 parameters(can be trained). Calculating from the code  in  github, it has 55,750 parameters.

​	The amount of AlexNet: 34944 + 307456 + 885120 + 663936 + 442624 + 37752832 + 16781312 + 4096000 = 60964224
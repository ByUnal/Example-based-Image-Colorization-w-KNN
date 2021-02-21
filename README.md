# Image-Colorization with KNN
Generally, classical colorization methods can be broadly separated into three categories:

- scribble-based colorization  (generally optimization based)
- example-based colorization  (generally optimization based and/or machine learning based)
- learning-based colorization  (generally machine learning based)

In this project, we will implement 'example-based colorization'

## Example-based Colorization

Lets assume that you are given with below image:

![1](https://user-images.githubusercontent.com/43930582/108631360-aa7bbc00-747a-11eb-811f-2ccf7e2ef82d.png)

Then you find a similar color image (or it is given to you):

![2](https://user-images.githubusercontent.com/43930582/108631476-1e1dc900-747b-11eb-9ae1-96a48f42c973.png)

So you are given an gray scale image, and a similar but colored image.
    
Then you will convert similar color image to gray scale image to have an input data (grayscaled similar image) and output data (given color similar image).

![3](https://user-images.githubusercontent.com/43930582/108631527-5fae7400-747b-11eb-8c57-e456f6fb6ac5.png) 
![4](https://user-images.githubusercontent.com/43930582/108631528-6210ce00-747b-11eb-8cd1-f873db59ea7e.png)

Note that, converting a color image to gray scale image is piece of cake.

Then you will develop your Machine Learning model with two given image and one genrated image.

And you will apply your ML model to given gray scale image to obtain a colored version. To see the performance, you can compare with the groundtruth image as well (and using error metric such as MSE we can measure the error):

![5](https://user-images.githubusercontent.com/43930582/108631530-64732800-747b-11eb-93b2-12e6bbd61126.png)

So this image is unknown true color image (also known as groundtuth)

### _Simple Example_

You are given a grayscale image: p007, b_target.png

Your aim is to generate colorized version of p007, b_target.png
You are also given a similar color (exampler) image p007, a_source.png

![MicrosoftTeams-image (1)](https://user-images.githubusercontent.com/43930582/108631758-7608ff80-747c-11eb-9ee4-2554c9052ab1.png)

Think image p007, a_source.png as y (output data)
Generate grayscale version of image p007, a_source.png  (which is a trivial task)
Think grayscaled(image p007, a_source.png) as X (input data)

Now you have X and y, so you can train a Machine Learning model.

Once you trained your Machine Learning model then you can apply this model on p007, b_target.png to create y_hat
Here y_hat is colorized version of p007, b_target.png

So during prediction your X is

![MicrosoftTeams-image (2)](https://user-images.githubusercontent.com/43930582/108631760-773a2c80-747c-11eb-9de4-a09e93606589.png) 
and then you will find y_hat (colorized image).


In order to test the performance of your colorizing method y_hat will be compared with below goundtruth image (normally you do not have this data):

![MicrosoftTeams-image (3)](https://user-images.githubusercontent.com/43930582/108631761-786b5980-747c-11eb-86a0-d4fe02a2fe01.png)
p007, c_groundtruth.png (groundtruth)

We will use Mean Absolute Error (MAE) between y_hat and groundtruth. If you find zero error then you have a perfect colorizer method.

## IDEA is simple

_Hint: You can test your method with a trivial case by testing it on train data._

- You have only a single color image C.
- Then obtain G which is grayscale version of C.
- Then extract features from G (grayscale image) and define this features as input data X.
- Color image C is output data y.
- Train your model with this X and y.  Then do prediction on X again to find a colorized version of G, name this as y_hat.

Find MAE between y_hat (colorized image) and C (color image).

So, you are training and testing on same data. If your method is not able to produce good result with this trivial testing approach then it is not probable that you will have good results on actual test data.

## Critical Parts in these kind of projects

- Features that are extracted from the grayscale image (as X, input data) 
- Feature vector dimension can be high so dimension reduction can be useful (i.e. for KNN), one can use PCA for that purpose
- Number of data can be huge for some methods (i.e. SVM, or can be slow in KNN) then you can reduce the number of data (i.e. using a clustering approach in an intelligent manner)
- Choice of machine learning model (probably you will need an approach that can deal with nonlinear data, in real-world data is mostly nonlinear)
	- note that colorization is a regression problem where each pixel contains red, green, blue (RGB) that takes values between 0 and 255
	- output is a vector with 3 dimension:  y_hat_i = [red_i, green_i, blue_i]  for i_th pixel
	- you may consider using classification algorithms to guide your regression method (not easy to imagine, be creative)
- Execution speed is quite important, for example KNN and SVM can be quite slow. There are ways to deal with this computational burden, think about it
- Your method can produce a color version of a pixel directly, or you can determine the color of a pixel in an iterative manner (i.e. initialy pixel is [gray_i, gray_i, gray_i] then iteratively it converges to [red_i, green_i, blue_i]

## Contributers

- Ahmet Oğuz Şenocak -> https://github.com/ahmetoguz1
- Aytaç Öntürk -> https://github.com/aytaconturk
- Yavuz Akın -> https://github.com/yavuzakin

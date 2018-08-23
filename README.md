# Drowsiness Detection
This is a Convolutional Neural Network (CNN) project developed as a summer interniship project at Bennet Univeristy, Greater Noida.
The project detects the drowsiness state of any person by monitoring the frequnency of his/her eye blinks.

The project comprising of 8 layers of Neural Networks for the eye blink detection -
* Layer-1 -> Convolution 2D Layer (To detect type-1 edges in images)
* Layer-2 -> Convolution 2D Layer (To detect type-2 edges in images)
* Layer-3 -> MaxPooling 2D Layer (To reduce the contrast of the images)
* Layer-4 -> Dropout(0.25) (To drop out the less weighted neurons to reduce over-fitting of model)
* Layer-5 -> Flatten
* Layer-6 -> Dense
* Layer-7 -> Dropout(0.5)
* Layer-8 -> Dense

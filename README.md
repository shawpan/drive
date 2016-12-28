# drive
Predict steering angle of a self driven car using behaviorial cloning 


# Simulation
## Track 2
<a href="http://www.youtube.com/watch?feature=player_embedded&v=D5FvSXcjfEg
" target="_blank"><img src="http://img.youtube.com/vi/D5FvSXcjfEg/0.jpg" 
alt="Track 2" width="608" height="480" border="10" /></a>
## Track 1
<a href="http://www.youtube.com/watch?feature=player_embedded&v=tkAR-Uqi4LU
" target="_blank"><img src="http://img.youtube.com/vi/tkAR-Uqi4LU/0.jpg" 
alt="Track 1" width="608" border="10" /></a>

# Data Collection
1. Data provided [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) by udacity is used with real time augmetation. data folder contains the contents given below:
  1. `IMG` folder - this folder contains all the frames of your driving. 
  2. `driving_log.csv` - each row in this sheet correlates your image with the steering angle, throttle, brake, and speed of  your car. You'll mainly be using the steering angle.
2. `driving_log.csv` looks like
<img src="driving_log_snap.png" width="1512" alt="Combined Image" />
3. Alternatively data can be collected via training phase of simulator which needs careful driving because data should be balanced between proper driving as well as recovery from mistakes. 

# Data Generation 
1. `driving_log.csv` contains only `n = 8036` rows which may not be sufficient to train the model well. So the left and right camera images are also used by adding `0.25` to the steering angle from `center` to produce `left` camera data and subtracting `0.25` to produce `right` camera data. Now we have `3n`  data from original source. 
```python
steering_angle = float(row['steering'])

# Generate data using left image
x.append(row['left'].strip())
y.append(steering_angle + 0.25)

# Generate data using right image
x.append(row['right'].strip())
y.append(steering_angle - 0.25)
```

# Real Time Data Augmentation
1. Images are kept in file and processed in batches using `batch_data_generator` method which is a python generator method and fed into `fit_generator` of the model in real time to avoid memory overflow. 
2. For each batch one flipped image on Y axis is produced for each input to produce more data.
```python
flipped_image = cv2.flip(image, 1)
x_data = np.append(x_data, np.array([flipped_image]), axis = 0)
```
Corresponding steering angle is also flipped as `-1 * steering_angle`
```python
y_data = np.append(y_data, -1 * y[i])
```
This way we have `2 * 3n`, ~40k data which covers most of the scenarios. 
So for initial batch size of 16 it returns 32 after batch processing which is the effective batch size. 
3. Input image is cropped above and below to remove unnecessary portions such as sky and back side of the car
```python
image = image[40:130]
```
4. Input image is resized to half to reduce computation.
```python
image = cv2.resize(image, (160,45))
```

# Normalization
1. Input values are normalized between `-0.5` and `0.5`. This is done using a `Lambda` layer in the model.

# Model Architecture
The model used here is a slight modification of the http://comma.ai/ model. It has 13 layers. 1 preprocessing Lambda layer
, 3 Conv layers having 16, 32 and 64 filters with size (8,8) - strides (4,4) ,size (5,5) - strides(2,2), size (5,5) - strides (2,2) and 1 Fully Connected layers having 512 hidden nodes and 1 output layer. For non-linearity activation Exponential Linear Unit, ELU is used. To prevent overfitting 2 dropout layer is used 

| Layer (type)                    |  Output Shape       | Param #    | Connected to          |
| ------------------------------- |:-------------------:| ----------:| ---------------------:|
| lambda_1 (Lambda)               | (None, 45, 160, 3)  | 0          | lambda_input_1[0][0]  |
| convolution2d_1 (Convolution2D) | (None, 12, 40, 16)  | 3088       | lambda_1[0][0]        |
| elu_1 (ELU)                     | (None, 12, 40, 16)  | 0          | convolution2d_1[0][0] |
| convolution2d_2 (Convolution2D) | (None, 6, 20, 32)   | 12832      | elu_1[0][0]           |
| elu_2 (ELU)                     | (None, 6, 20, 32)   | 0          | convolution2d_1[0][0] |
| convolution2d_3 (Convolution2D) | (None, 3, 10, 64)   | 51264      | elu_2[0][0]           |
| flatten_1 (Flatten)             | (None, 1920)        | 0          | convolution2d_3[0][0] |
| dropout_1 (Dropout)             | (None, 1920)        | 0          | flatten_1[0][0]       |
| elu_3 (ELU)                     | (None, 1920)        | 0          | dropout_1[0][0]       |
| dense_1 (Dense)                 | (None, 512)         | 983552     | elu_3[0][0]           |
| dropout_2 (Dropout)             | (None, 512)         | 0          | dense_1[0][0]         |
| elu_4 (ELU)                     | (None, 512)         | 0          | dropout_2[0][0]       |
| dense_2 (Dense)                 | (None, 1)           | 513        | elu_4[0][0]           |
 **Total params: 1051249**


# Training
comma.ai and NVIDIA have already designed successful models and tested on real life scenarios for self driven car. NVIDIA model has more number of parameters. At first tried both but running NVIDIA model was more time consuming. So chose comma.ai model but with sufficient modification to work on our data or camera images. It is a very simple model compared to others regarding architecture and number of parameters. I had to build and train the model without GPU support so it has been a good choice and many of the hyperparameters have also been tuned keeping it on mind. The training phase has been based on mostly trial and error. The significant part of the process is 

1. The initial data is used to generate ~40k inputs using the process described in data generation and data augmentation. 
2. 15% of the entire data was kept as validation set. 
3. Cropped image so that only the desired portion of the road is fed to train the model removing sky and back side of the car. This is important because noisy data may mislead the car.
4. Resized the input image to half so that total number of trainable parameters is reduced. It significantly reduced the computation time without hampering performance.
5. Used Python generator to supply training data in small batches since loading all the images in memory and augmenting is not feasible. `fit_generator` method of keras model is specifically built for this optimization.
6. Chose batch size to be 16 which is spawned to become 32 after augmentation because it took less time as well as better performance. Before it, tried 64 and 128 which took a longer time as expected but did not improve much.
7. Used adam optimizer with default learning_rate = 0.001 for minimizing loss. Both of the dropout layers were assigned keep_probabilty of 50%. 
8. Intially trained the model for 20 Epochs and kept the best set of weights using `ModelCheckpoint` provided by keras. It took about 2 hours with `batch_size=32`. Applied this weight to simulator and the car started driving well for a few seconds with a fixed `throttle=0.2` before going astray. For the batch size 64 and 128, it was about 1 hour per 5 epoch. Clearly at this point, the model had a bias towards 0 angle because of imbalanced data of excessive 0 steering angle.
9. Repeated the process after reduceing the straight angles from training data but then the car become biased towards frequent turns. So decided to keep all the data having 0 steering_angles for smooth driving.
9. Then started tuning the weights reducing the learning rate by a factor of 10 and repeatedly used the same approach for 5 epochs with preloaded weights of previous best.
10. By this time, the car was successful to drive well through the track except a nasty turn after the bridge. The combination of high speed and the predicted angle was failing it to keep its tire on the road.
11. Kept reducing the learning rate and training with preloaded weights of previous best validation loss. But only for 1 epoch. It helped me to save time as each epoch takes about 6 minutes and I was able to check the simulation for the new weights.
12. Finally the car was running better on the track and able to complete the lap. At this point the training loss was 0.0147 and validation loss was 0.0172
13. At the end to make the driving smoother, the throttle in `drive.py` was reduced to 0.15 when the car was driving appearently straight i.e., `-0.1< steering_angle < 0.1` and 0 otherise which acted as a break.

# drive
Predict steering angle of a self driven car using behaviorial cloning 

# Data Collection
1. Data provided [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) by udacity is used with real time augmetation. data folder contains the contents given below:
  1. `IMG` folder - this folder contains all the frames of your driving. 
  2. `driving_log.csv` - each row in this sheet correlates your image with the steering angle, throttle, brake, and speed of  your car. You'll mainly be using the steering angle.
2.`driving_log.csv` looks like
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
x.append(row['left'].strip())
y.append(steering_angle + 0.25)
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

# Simulation
<a href="http://www.youtube.com/watch?feature=player_embedded&v=tkAR-Uqi4LU
" target="_blank"><img src="http://img.youtube.com/vi/tkAR-Uqi4LU/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="1074" height="556" border="10" /></a>

Here’s the text formatted for easy copying:

---

# Autonomous Driving Projects Overview

My involvement in ten core industry-sponsored self-driving car projects, including collaborations with leaders like NVIDIA and Mercedes, significantly honed my skills and deepened my domain knowledge in EV and autonomous driving software. These projects were pivotal in advancing cutting-edge technologies in this field. You can view my work on these critical projects here.

This repository showcases a collection of projects focused on various aspects of autonomous driving, such as lane detection, vehicle control, localization, and path planning. Each project employs advanced algorithms and machine learning techniques to address key challenges in the development of autonomous vehicles.

## Projects

### 1. Advanced Lane Finding

This project develops a robust pipeline for detecting lane lines on the road, particularly in complex scenarios involving curves. Key techniques used include:

- **Camera Calibration**: Correcting lens distortion to ensure accuracy.
- **Image Processing**: Applying color and gradient thresholding to highlight lane lines.
- **Perspective Transform**: Converting images to a bird's-eye view for better lane detection.
- **Lane Detection**: Fitting polynomials to lane lines, accommodating road curvature.
- **Lane Metrics**: Calculating lane curvature and vehicle offset from the lane center.
- **Overlay Visualization**: Projecting detected lanes back onto the original road image.

### 2. Behavioral Cloning

This project involves the autonomous control of a vehicle in a simulated environment using a Convolutional Neural Network (CNN) trained to predict steering angles. The workflow includes:

- **Data Collection**: Simulating optimal driving behavior for training data.
- **Model Development**: Building and training a CNN to predict steering based on camera input.
- **Training and Validation**: Ensuring the model generalizes well across different driving scenarios.
- **Simulation Testing**: Evaluating the model's ability to drive autonomously on a virtual track.

### 3. Extended Kalman Filter

In this project, an Extended Kalman Filter (EKF) is implemented to fuse data from lidar and radar sensors, providing accurate state estimation for a moving object. The focus areas include:

- **Sensor Fusion**: Combining data from multiple sensors to enhance accuracy.
- **State Estimation**: Tracking the object's position and velocity over time.
- **Performance Evaluation**: Validating the filter's accuracy against set benchmarks.

### 4. Kidnapped Vehicle

This project implements a 2D Particle Filter for vehicle localization. The filter uses a map, initial localization data, and sensor/control inputs to estimate the vehicle's position. The core tasks include:

- **Particle Filtering**: Estimating the vehicle's position using a probabilistic approach.
- **Sensor Data Integration**: Processing noisy sensor data for more accurate localization.
- **Localization Precision**: Continuously refining the vehicle's position estimate.

### 5. Path Planning

The goal of this project is to navigate a vehicle on a highway safely and efficiently, accounting for other traffic. Key objectives include:

- **Localization**: Utilizing sensor data for precise vehicle positioning.
- **Speed Control**: Maintaining a target speed while avoiding collisions.
- **Lane Management**: Ensuring the vehicle stays within its lane and changes lanes when necessary.

### 6. PID Controller

This project involves the design and tuning of a PID controller to steer the vehicle based on Cross Track Error (CTE). The process includes:

- **Controller Design**: Implementing the PID algorithm to minimize CTE.
- **Parameter Tuning**: Adjusting PID gains to achieve optimal steering control.
- **Simulation Testing**: Verifying the controller's performance in a simulated driving environment.

### 7. Traffic Sign Recognition

This project focuses on developing a Convolutional Neural Network (CNN) to classify traffic signs using the German Traffic Sign Dataset. The approach includes:

- **Data Preprocessing**: Preparing and augmenting the dataset for training.
- **Model Training**: Developing a CNN to accurately classify traffic signs.
- **Model Evaluation**: Testing the model's performance on unseen traffic sign images.

## Getting Started

Each project folder contains detailed instructions on setup, execution, and evaluation. Please refer to the respective READMEs within each folder for specific guidance.

## Dependencies

The projects rely on various dependencies, including Python libraries (e.g., OpenCV, TensorFlow), C++ tools, and simulation environments. Detailed dependency lists and installation instructions are provided in the individual project folders.
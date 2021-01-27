# Path Planning Model Description
This document describes how the path planning algorithm is modeled for the Udacity CarND-Path-Planning-Project.

## Problem Statement 

This project's objective is to design a path planner so that the host vehicle is able to drive at least 4.32 miles without incident while the car does not exceed a total of 10 m/s^2 and a jerk of 10 m/s^3, obeying 50 mph speed limit. The path planning algorithm is tested under the highway driving condition with all vehicles except the host being autonomous in the Udacity's driving simulator. The simulator runs at 50 freq and works by feeding in host vehicle's projected position points in global.

From the simulator, the following data are subscribed from the simulator:
1. host vehicle's position x,y in global frame
2. host vehicle's position s,d in frenet coordinate
3. host vehicle's yaw rate information
4. host vehicle's speed
5. host vehicle's previous path x,y in global frame
6. host vehicle's end path s,d in frenet coordinate
7. target vehicle's information from sensor fusion 
   7.1 speed x,y
   7.2 position s,d

## Criteria

## Max Acceleration and jerk are not Exceeded
the host car mainly drives along by setting reference velocity, which is initialized as 0 mph. This relatively low initial speed helps the car's acceleration and jerk under each 10 m/s^2 and 10 m/s^3. This then is updated based on speed of the preceding vehicle and the nearby vehicles on the neighbor lanes. 

## The car drives according to the speed limit
Simple speed control logic is implemented so that the speed is gradually decreased once the car reaches 95% of the max speed limit of 50 mph. The current apporach subtracts down the speed increment value of 0.224 per cycle.

## Car does not have collisions
Similar to the above, once the car approaches too close to the preceding vehicle (within 5m), the next cycle's speed is updated by the current speed subtracted from 3 times the speed increment value of 0.224, and for the other cases regarding the preceding vehicle being close(with in 15m), the updated speed is subtracted by 1 time the speed increment value. For the close-by vehicles in the neighboring lanes, analogously, the speed is subtracted by 1 time the speed increment value, preventing any collisions from lane change.

## The car stays in its lane, except for the time between changing lanes
The method of host vehicle following and staying in its lane can simplified using Frenet coordinate system. the data regarding all vehicles' longitudinal and lateral distance in Frenet coordinate can be subscribed and used from the simulator. Lateral distance, d is very useful when feeding in lane change information eg) left lane can be described by d - lane width(4m in this case) while right lane being described by d + lane width. For making the car stay in its lane, without changing d, future longitudinal distance, s + alpha(some integer to describe the forward distance) is changed and fed into the simulator.


## The car is able to change lanes
Analogous to the approach above, adding or subtracting the lateral distance, d value by lane width and feeding into the simulator essentially plots the new path along the left/right plan-to-change lane and enables changing lanes. This 'plot' portion is handled by using spline header(spline.h), which takes in a series of points, enabling the interpolation along the curve. The spline takes in two x,y location information from the previous path planner and the three points along 30m, 60m, and 90m along longitudinal direction in the Frenet coordinate(later converted to global once feeding into the simulator), then it at max generates 50 points by interpolation, taking some from the previous points.   

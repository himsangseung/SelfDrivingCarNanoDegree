#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "spline.h"
#include "json.hpp"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  int lane = 1;       // start lane
  double ref_vel = 5; // reference velocity at start (mph)

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy, &lane, &ref_vel]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side 
          // of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

 	  int prev_size = previous_path_x.size(); 
	  
	  // if previous path exists, track the end path point as current
	  if (prev_size > 0)
	    car_s = end_path_s;

          /*
           * define a path made up of (x,y) points that the car will visit
           * sequentially every .02 seconds
           */ 
	  bool laneChangePrep  = false;
	  bool laneChangePrepL = true;  bool laneChangePrepR = true;
	  bool tooCloseFront   = false; bool tooCloseLR      = false;
	  vector<double> dist;

	  // checking if the preceding vehicle is too close
	  // if not, prepare for lane change
	  for (int i = 0; i < sensor_fusion.size(); i++) 
	  {
	    double vx = sensor_fusion[i][3]; // velocity along x in global
	    double vy = sensor_fusion[i][4]; // velocity along y in global
	    double  s = sensor_fusion[i][5]; // long dist in Frenet
	    double  d = sensor_fusion[i][6]; // lat dist in Frenet
	    double sp = sqrt(pow(vx,2) + pow(vy,2)); // speed in global
	    s += prev_size * period * sp; // updated long dist - prev.path 
	    double distBtw = s - car_s; // diff in dist btw host and targe 

	    // within safety distance threshold -too close -> slowDown Flag
	    // outside the threshold -> prepare for lane change  
	    if (inLane(lane, d) && inSafetyDistFront(s, car_s)) 
	    {   
		std::cout << "** Preceding vehicle too close **\n" <<
		"checking for lane change availability\n" << std::endl; 
	        tooCloseFront = true;
	        laneChangePrep = true; 
	    }
	    if (inLane(lane, d) && distBtw > 0)
	      dist.emplace_back(distBtw);
	  }
	  
	  // if the lane change prep flag is on, check if the vehicle in
	  // the left/right lane is within the safety distance threshold
	  // and set the flag accordingly, simliar to the above
	  if (laneChangePrep) 
	  {
	   for (int i = 0; i < sensor_fusion.size(); i++)
	   {
	     double vx = sensor_fusion[i][3]; // velocity along x in global
	     double vy = sensor_fusion[i][4]; // velocity along y in global
	     double  s = sensor_fusion[i][5]; // long dist in Frenet
	     double  d = sensor_fusion[i][6]; // lat dist in Frenet
	     double sp = sqrt(pow(vx,2) + pow(vy,2)); // speed in global
	     s += prev_size * period * sp;// updated long dist - prev.path

	     // left lane
	     if (inLane(lane - 1, d))
	     {
	       if (inSafetyDistLR(s, car_s))
	       {
	         tooCloseLR = true;
   		 laneChangePrepL = false;		 
	       }
	     } 
	     // right Lane
	     else if (inLane(lane + 1, d))
             {
               if (inSafetyDistLR(s, car_s))
	       {
		 tooCloseLR = true;
	 	 laneChangePrepR = false;	 
	       }
	     } 
	   } 
	  }

 	  /* Perform lane change feed in added lateral offset in Frenet */
	  if (laneChangePrep && laneChangePrepL && lane > 0)
	  {  
	    std::cout << "Perform Lane Change - Left" << std::endl;
	    std::cout << "----------------------------------=" <<std::endl;
	    lane -= 1;
	  }
	  else if (laneChangePrep && laneChangePrepR && lane < 2)
	  {
	    std::cout << "Perform Lane Change - Right" << std::endl;
	    std::cout << "-----------------------------------" <<std::endl;
            lane += 1;
	  }
	 
	  /* simple speed control */
	  // closest vehicle in the host lane < 5m, emergency break 
	  if ((dist.size() != 0 && 
	   *std::min_element(dist.begin(), dist.end()) < slowDownDist))
	    ref_vel -= velIncrement * 3.0;
	  // keep the speed below the limit
	  else if (car_speed > speedLimit * 0.95)
	    ref_vel -= velIncrement;
	  // too-close and speed limit flag, gradually slowing down
	  else if (tooCloseFront || tooCloseLR) 
	    ref_vel -= velIncrement;
	  // within speed limit, gradually speed up
	  else if (car_speed < speedLimit * 0.8)
	    ref_vel += velIncrement; 
	  else 
	    ref_vel += velIncrement;

	  vector<double> ptsx;
	  vector<double> ptsy;
	  // reference x,y, yawrate
	  // either we will reference the starting point as where the car
	  // is or at the previous paths end points
	  double ref_x = car_x;
	  double ref_y = car_y;
	  double ref_yaw = deg2rad(car_yaw);
	
	  if (prev_size < 2) 
	  {
	    // use two points that make the path tangent to the car
	    double prev_car_x = car_x - cos(car_yaw);
	    double prev_car_y = car_y - sin(car_yaw);

	    ptsx.emplace_back(prev_car_x);
	    ptsx.emplace_back(car_x);
	   
	    ptsy.emplace_back(prev_car_y);
	    ptsy.emplace_back(car_y);
	  }
	  // use the previous path's end point as starting reference
	  else 
	  {
	    //redefine reference state as previous path end point
	    ref_x = previous_path_x[prev_size-1];
	    ref_y = previous_path_y[prev_size-1];

	    double ref_x_prev = previous_path_x[prev_size-2];
    	    double ref_y_prev = previous_path_y[prev_size-2];
	    ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

	    // use two points that make the path tangent to the previous
	    // path's end point
	    ptsx.emplace_back(ref_x_prev);
	    ptsx.emplace_back(ref_x);

	    ptsy.emplace_back(ref_y_prev);
	    ptsy.emplace_back(ref_y);	    
	  }
	
	  // In Frenet add evenly 30m spaced points ahead of the starting
	  // reference
	  vector<double> next_wp0 = getXY(car_s+30, laneWidth*(1/2+lane), 
	      map_waypoints_s, map_waypoints_x, map_waypoints_y);
	  
	  vector<double> next_wp1 = getXY(car_s+60, laneWidth*(1/2+lane),
              map_waypoints_s, map_waypoints_x, map_waypoints_y);

	  vector<double> next_wp2 = getXY(car_s+90, laneWidth*(1/2+lane), 
              map_waypoints_s, map_waypoints_x, map_waypoints_y);
	  
	  ptsx.emplace_back(next_wp0[0]);
	  ptsx.emplace_back(next_wp1[0]);
	  ptsx.emplace_back(next_wp2[0]);
	  
	  ptsy.emplace_back(next_wp0[1]);
	  ptsy.emplace_back(next_wp1[1]);
	  ptsy.emplace_back(next_wp2[1]);

	  for (int i = 0; i < ptsx.size(); i++)
	  {
	    //shift car reference angle to 0 degrees
	    double shift_x = ptsx[i]-ref_x;
	    double shift_y = ptsy[i]-ref_y;

	    ptsx[i] = (shift_x * cos(0-ref_yaw)-shift_y * sin(0-ref_yaw));
	    ptsy[i] = (shift_x * sin(0-ref_yaw)+shift_y * cos(0-ref_yaw));
	  }
	  // create a spline instance
	  tk::spline s;

	  // set (x,y) anchor points to the spline
	  s.set_points(ptsx,ptsy);

	  // define the actual (x,y) points we will use for the planner
	  vector<double> next_x_vals;
	  vector<double> next_y_vals;

	  // start with all of the previous path points from last time
	  // indicating future paths left from the previous paths 
	  // for continuity
	  for (int i = 0; i < previous_path_x.size(); i++)
	  {
	    next_x_vals.emplace_back(previous_path_x[i]);
	    next_y_vals.emplace_back(previous_path_y[i]);
	  }

	  // calculate how to break up spline points so that we travel
	  // at our desired reference velocity
	  double target_x = 30.0;
	  double target_y = s(target_x); // f(30) = target_y
	  double target_dist = sqrt(pow(target_x,2)+pow(target_y,2));

	  double x_add_on = 0; // where we start -result from coord shift

	  // fill up the rest of our path planner after filling it with
	  // previuos points, here we will always output 50 points
	  for (int i = 1; i <= 50 - previous_path_x.size(); i++)
	  {
	    double N = (target_dist/(period * ref_vel / 2.24));//2.24:mph-mps
	    double x_point = x_add_on + (target_x) / N;
	    double y_point = s(x_point);

	    x_add_on = x_point;

	    double x_ref = x_point;
	    double y_ref = y_point;

	    // rotate back to global from local coordinate
	    x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
	    y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));

	    x_point += ref_x;
	    y_point += ref_y;

	    next_x_vals.emplace_back(x_point);
	    next_y_vals.emplace_back(y_point);
	  }
	  
	  // // testing straight line const velocity
	  //double dist_inc = 0.5;
	  //for (int i = 0; i < 50; i++) {
	  //  double next_s = car_s + (i+1)*dist_inc;
	  //  double next_d = 6;
	  //  vector<double> xy = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
	  //  next_x_vals.emplace_back(xy[0]);
	  //  next_y_vals.emplace_back(xy[1]);
	  //}

	  json msgJson;
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}

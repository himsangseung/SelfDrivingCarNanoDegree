/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::normal_distribution<double> normDist_x (x, std[0]);
  std::normal_distribution<double> normDist_y (y, std[1]);
  std::normal_distribution<double> normDist_theta (theta, std[2]);

  num_particles = 300;  // TODO: Set the number of particles
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = normDist_x(gen);
    p.y = normDist_y(gen);
    p.theta = normDist_theta(gen);
    p.weight = 1.0;
    std::cout <<"p.x : " <<  p.x << std::endl; 
    particles.emplace_back(p);
  } 
  // gen = reinterpret_cast<std::default_random_engine>(gen);
  initialized();
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

   double yawLimit = 1e-5;
   for (int i = 0; i < num_particles; i++) {
     
     double predictionX, predictionY, predictionT;

     double previousX = particles[i].x;
     double previousY = particles[i].y;
     double previousT = particles[i].theta;
     	 
     if (std::fmax(abs(yaw_rate), yawLimit) == yawLimit) {
       predictionT = previousT + (yaw_rate * delta_t);
       predictionX = previousX + (velocity / yaw_rate) * 
	             (sin(predictionT) - sin(previousT));
       predictionY = previousY + (velocity / yaw_rate) *
	       	     (cos(previousT) - cos(predictionT));
     } else {
       predictionT = previousT;
       predictionX = previousX + (velocity * delta_t * cos(previousT));
       predictionY = previousY + (velocity * delta_t * sin(previousT));
     }

     std::normal_distribution<double> normDist_x (predictionX, std_pos[0]);
     std::normal_distribution<double> normDist_y (predictionY, std_pos[1]);     std::normal_distribution<double> normDist_theta (predictionT, std_pos[2]);
  
     particles[i].x = normDist_x(gen);
     particles[i].y = normDist_y(gen);
     particles[i].theta = normDist_theta(gen);
   }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (auto & obs : observations) {
    double min_dist = std::numeric_limits<double>::max();

    for (const auto & pred_obs : predicted) {
      double d = dist(obs.x, obs.y, pred_obs.x, pred_obs.y);
      if (d < min_dist) {
	obs.id = pred_obs.id;
	min_dist = d;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i = 0; i < num_particles; i++) {

    double particleX = particles[i].x;
    double particleY = particles[i].y;
    double particleT = particles[i].theta;    
    
    vector<LandmarkObs> predictionLandVec;
    auto & landmarkList = map_landmarks.landmark_list;
    for (size_t j = 0; j < landmarkList.size(); j++) {
      int    landID = landmarkList[j].id_i;
      double landX  = double(landmarkList[j].x_f);
      double landY  = double(landmarkList[j].y_f);
	    
      double distance = dist(particleX, particleY,
		             landX, landY); 
      
      if (sensor_range > distance) {
        LandmarkObs predictionLand;
        predictionLand.id = landID;
	predictionLand.x  = landX; 
	predictionLand.y  = landY;

	predictionLandVec.emplace_back(predictionLand);	
      }
    }	  

    vector<LandmarkObs> observedLandVec;
    for (size_t k = 0; k < observations.size(); k++) {
      LandmarkObs observedLand;
      observedLand.x = particleX + cos(particleT) * observations[k].x - 
	      		           sin(particleT) * observations[k].y;
      observedLand.y = particleY + sin(particleT) * observations[k].x +
	     			   cos(particleT) * observations[k].y;
      
      observedLandVec.emplace_back(observedLand);
    }

    dataAssociation(predictionLandVec, observedLandVec);

    double particleLikelihood = 1.0;
    double mu_x; double mu_y;

    for (const auto & obs : observedLandVec) {
      for (const auto & land : predictionLandVec) {
	if (obs.id == land.id) {
 	  mu_x = land.x;
	  mu_y = land.y;
	  break;
	}
      double norm_factor = 2 * M_PI * std_landmark[0] * std_landmark[1];
      double prob = exp(-(pow(obs.x - mu_x, 2)/(2*std_landmark[0]*
			      std_landmark[0]) + pow(obs.y - mu_y,2) /
			      (2*std_landmark[1] * std_landmark[1])));
      particleLikelihood *= prob / norm_factor; 
    }
    particles[i].weight = particleLikelihood;
  }
  double norm_factor = 0.0;
  for (const auto & particle : particles) 
    norm_factor += particle.weight;

  for (auto & particle : particles)
     particle.weight /= (norm_factor + std::numeric_limits<double>::epsilon());
}

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  weights.clear();
  for (const auto & particle : particles)
    weights.emplace_back(particle.weight);

  std::discrete_distribution<int> DistWeight (weights.begin(), weights.end());

  vector<Particle> resampledParticles;
  for (int i = 0; i < num_particles; i++) {
    resampledParticles.emplace_back(particles[DistWeight(gen)]);
  }

  particles = resampledParticles;
  for (size_t i = 0; i < particles.size(); i++) {
    particles[i].weight = 1.0;
  }
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

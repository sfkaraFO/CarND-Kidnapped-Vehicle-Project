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
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  // Initialize all particles to initial measurements. Add Gaussian noise to corresponding values. 
  // Set all weights to 1 since we do not have further information about the weights yet. 
  num_particles = 100;  // TODO: Set the number of particles
  for (int i=0; i<num_particles; ++i){
    Particle particle;
    particle.id = i;
    particle.x = NormalRandom(x, std[0]);
    particle.y = NormalRandom(y, std[1]);
    particle.theta = NormalRandom(theta, std[2]);
    particle.weight = 1.0;
    particles.push_back(particle);
  }
  is_initialized = true;
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

  // Predict the particles using yaw_rate and velocity information. 
  // If the yaw_rate is close to zero use simple model.
  // Else more complex model is used.
  for (auto& particle: particles){

    if (abs(yaw_rate) <= 0.00001) {

      particle.x = particle.x + delta_t*velocity*cos(particle.theta);
      particle.y = particle.y + delta_t*velocity*sin(particle.theta);
    }
    else {
      
      particle.x = particle.x + velocity/yaw_rate*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta));
      particle.y = particle.y + velocity/yaw_rate*(cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t));
      particle.theta = particle.theta + yaw_rate*delta_t;
    }

    // Insert the process noise to the particles.
    particle.x = NormalRandom(particle.x, std_pos[0]);
    particle.y = NormalRandom(particle.y, std_pos[1]);
    particle.theta = NormalRandom(particle.theta, std_pos[2]);

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

  // Assign the predicted landmarks to transformed observations.
  // Find the minimum distance between points.
  for (auto& transObservation: observations){
    
    double minDistance = std::numeric_limits<double>::max();
    
    for (auto landmark: predicted){

      double distance = dist(transObservation.x, transObservation.y, landmark.x, landmark.y);
      if ( distance < minDistance ){
        
        transObservation.id = landmark.id;
        minDistance = distance;
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

  // Reset the particle weights if the resampling is performed. Else use the previous values.
  if (resamplingPerformed){

    for (auto& particle : particles){
      
      particle.weight = 1.0;
    }
  }


  for (auto& particle: particles){
    
    // Find the closest landmarks to the corresponding particles.
    vector<LandmarkObs> predictedLandmarks;
    for (auto landmark: map_landmarks.landmark_list){
      if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) < sensor_range) {
        
        LandmarkObs lob;
        lob.x = landmark.x_f;
        lob.y = landmark.y_f;
        lob.id = landmark.id_i;
        predictedLandmarks.push_back(lob);
      }
    }

    // Transform the observations to the map coordinates.
    vector<LandmarkObs> transObservations;
    for (auto obs: observations){

      LandmarkObs transObs;
      transObs.x = cos(particle.theta)*obs.x - sin(particle.theta)*obs.y + particle.x;
      transObs.y = sin(particle.theta)*obs.x + cos(particle.theta)*obs.y + particle.y;

      transObservations.push_back(transObs);
    }

    // Associate the closests landmarks to the transformed measurements.
    dataAssociation(predictedLandmarks, transObservations);

    double likelihood = 1.0;

    // Get the transformed measurement's landmark id.
    for (auto transObs: transObservations){

      double x;
      double y;
      for (auto landmark: predictedLandmarks) {
        
        if (landmark.id == transObs.id) {
          
          x = landmark.x;
          y = landmark.y;
          break;
        }
      }

      // Calculate the likelihood value using this id.
      double obs_w = ( 1/(2*M_PI*std_landmark[0]*std_landmark[1]))
                     * exp( -( pow(transObs.x - x, 2) / (2*pow(std_landmark[0],2)) +
                               pow(transObs.y - y, 2) / (2*pow(std_landmark[1],2)) ) );

      // Multiply each measurements likelihood since they are independent.
      likelihood = likelihood * obs_w;

    }

    particle.weight = particle.weight*likelihood;

  }

  // Calculate sum of the weights and mormalize them.
  double sumOfWeights = 0.0;
  for (auto& particle : particles) {

    sumOfWeights = sumOfWeights + particle.weight;
  }

  for (auto& particle : particles) {

    particle.weight = particle.weight / sumOfWeights;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Calculate the sum of squares.
  double sumOfSquareWeights = 0.0;
  vector<double> weights;
  for (auto particle: particles) {

    sumOfSquareWeights += particle.weight*particle.weight;
    weights.push_back(particle.weight);
  }

  // Calculate the effective number of particles.
  double Neff = 1/sumOfSquareWeights;
  if ( Neff < ((double)num_particles/2) ){
  // if (true) {

    // Pick a random index.
    int index = UniformIntRandom(0, num_particles);

    // Find the maximum valued weight.
    auto it = max_element(weights.begin(), weights.end());
    double max_weight = *it;

    double beta = 0.0;

    vector<Particle> newParticles;

    for (int i = 0; i < num_particles; ++i) {
      // Generate uniform reel number
      beta = beta + UniformReelRandom(.0, 2*max_weight);
      while (beta > weights[index]) {
        beta = beta - weights[index];
        index = (index + 1) % num_particles;
      }
      newParticles.push_back(particles[index]);
    }

    particles = newParticles;

    resamplingPerformed = true;
  }
  else{

    resamplingPerformed = false;
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
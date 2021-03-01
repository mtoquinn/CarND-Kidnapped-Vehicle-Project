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

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
  //Only do if not already initialized
  if (!is_initialized)
  {
    num_particles = 100;  //number of particles
  	std::normal_distribution<double> dist_x(x, std[0]);
  	std::normal_distribution<double> dist_y(y, std[1]);
  	std::normal_distribution<double> dist_theta(theta, std[2]);
    
    for (int i = 0; i < num_particles; i++)
    {
      Particle p; //New particle
      p.id = i; //particle id 
      p.x = dist_x(gen);
      p.y = dist_y(gen);
      p.theta = dist_theta(gen);
      p.weight = 1.0; //initial weight
      particles.push_back(p);
      weights.push_back(p.weight);
    }
    
    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) 
{
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]); 
  
  for (int i = 0; i < particles.size(); i++)
  { 
    double theta = particles[i].theta;

    if (fabs(yaw_rate) >= 0.00001)
    {
      particles[i].x += (velocity/yaw_rate) * (sin(theta + (yaw_rate * delta_t)) - sin(theta));
      particles[i].y += (velocity/yaw_rate) * (cos(theta) - cos(theta + (yaw_rate * delta_t)));
      particles[i].theta += (yaw_rate * delta_t);
    }
    else
    {
      particles[i].x += velocity * delta_t * cos(theta);
      particles[i].y += velocity * delta_t * sin(theta);
    }
    
    particles[i].x += dist_x(gen);
  	particles[i].y += dist_y(gen);
  	particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) 
{
  double smallest;
  double distance;
  
  for (unsigned int i = 0; i < observations.size(); i++)
  {
    smallest = 1000000; //arbitrarily large number for initialization
    
    for (unsigned int j = 0; j < predicted.size(); j++)
    {
      distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      
      if (distance < smallest)
      {
        smallest = distance;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) 
{
  double particle_x, particle_y, particle_theta;
  double predicted_x, predicted_y, exponent, weight;
  double transformed_x, transformed_y;
  double total_weight, distance;
  double weight_sum = 0.0;
  double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  
  for (unsigned int i = 0; i < particles.size(); i++)
  {
    particle_x = particles[i].x;
    particle_y = particles[i].y;
    particle_theta = particles[i].theta;
    total_weight = particles[i].weight = 1.0;
    std::vector<LandmarkObs> predicted;
    
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) 
    {
      distance = dist(map_landmarks.landmark_list[j].x_f, 
                      map_landmarks.landmark_list[j].y_f,
                      particle_x, particle_y);
      //Make sure it is in range
      if (distance <= sensor_range)
      {
      	predicted.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, 
                                        map_landmarks.landmark_list[j].x_f,
                                        map_landmarks.landmark_list[j].y_f});
      }
    }
    
    std::vector<LandmarkObs> transformed_obs;
    
    for (unsigned int k = 0; k < observations.size(); k++)
    {
      transformed_x = particle_x + (cos(particle_theta) * observations[k].x) - (sin(particle_theta) * observations[k].y);
      transformed_y = particle_y + (sin(particle_theta) * observations[k].x) + (cos(particle_theta) * observations[k].y);
      transformed_obs.push_back(LandmarkObs{-1, transformed_x, transformed_y});
    }
    
    dataAssociation(predicted, transformed_obs);
    
    for (unsigned int l = 0; l < transformed_obs.size(); l++)
    {
      for (unsigned int m = 0; m < predicted.size(); m++)
      {
        if(predicted[m].id == transformed_obs[l].id)
        {
          predicted_x = predicted[m].x;
          predicted_y = predicted[m].y;
          break;
        }
      }
      
      exponent = (pow(transformed_obs[l].y - predicted_y, 2) / (2 * pow(std_landmark[1], 2))) + (pow(transformed_obs[l].x - predicted_x, 2) / (2 * pow(std_landmark[0], 2)));
      
      weight = gauss_norm * exp(-exponent);
      total_weight *= weight;
    }
      particles[i].weight = weights[i] = total_weight;
      weight_sum += total_weight;
  }
  
  if (weight_sum != 0.0) //prevent division by zero
  {
    for (unsigned int i = 0; i < particles.size(); i++)
    {
      particles[i].weight /= weight_sum;
      weights[i] /= weight_sum;
    }
  }
}

void ParticleFilter::resample() 
{  
  std::vector<Particle> resampled;
  std::discrete_distribution<int> random_weights(weights.begin(), weights.end());
  
  for (unsigned int i = 0; i < particles.size(); i++)
  {
    resampled.push_back(particles[random_weights(gen)]);
  }
  
  particles = resampled;
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
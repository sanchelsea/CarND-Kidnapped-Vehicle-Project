/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    this->num_particles = 100;
    double gps_x = x;
    double gps_y = y;
    double gps_theta = theta;
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];
    
    //Create a normal (Gaussian) distribution for x, y and theta
    normal_distribution<double> dist_x(gps_x, std_x);
    normal_distribution<double> dist_y(gps_y, std_y);
    normal_distribution<double> dist_theta(gps_theta, std_theta);

    for (int i = 0; i < this->num_particles; ++i) 
    {
        double sample_x, sample_y, sample_theta;
        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);	 
        
        Particle par;
        par.id =i+1;
        par.x = sample_x;
        par.y = sample_y;
        par.theta = sample_theta;
	par.weight = 1;
        this->particles.push_back(par);
        
        // Print your samples to the terminal.
        cout << "Sample " << i + 1 << " " << sample_x << " " << sample_y << " " << sample_theta << endl;
    }
    
    this->is_initialized = true;


}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    
    //Create a normal (Gaussian) distribution for x, y and theta
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    
    for (int i = 0; i < this->num_particles; ++i) 
    {
        
         if (fabs(yaw_rate) < 0.00001) 
         { 
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
         }
         else
         {
            double theta_f = particles[i].theta + (yaw_rate * delta_t);
            particles[i].x += ((velocity/yaw_rate)*(sin(theta_f) - sin(particles[i].theta)));
            particles[i].y += ((velocity/yaw_rate)*(cos(particles[i].theta) - cos(theta_f)));
            particles[i].theta = theta_f;
         }       
               
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    for (unsigned int i = 0; i < observations.size(); ++i) 
    {
        double min_dist = numeric_limits<double>::max();
        for (unsigned int j = 0; j < predicted.size(); ++j) 
        {
            double dist_calc = dist(predicted[j].x,predicted[j].y,observations[i].x,observations[i].y);
            if (dist_calc < min_dist)
            {
                min_dist = dist_calc;
                observations[i].id = predicted[j].id;
            }
        }
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    double gauss_norm = 0.5 * M_PI * std_landmark[0] * std_landmark[1];
    this->weights.clear();
    
    for (int i = 0; i < this->num_particles; ++i) 
    {
        std::vector<LandmarkObs> observations_map; 
    
        for (unsigned int j = 0; j < observations.size(); ++j) 
        {
            LandmarkObs obs_map;
            obs_map.x = particles[i].x + (cos(particles[i].theta) * observations[j].x) -  (sin(particles[i].theta)*observations[j].y);
            obs_map.y = particles[i].y + (sin(particles[i].theta) * observations[j].x) +  (cos(particles[i].theta)*observations[j].y);
            observations_map.push_back(obs_map);
        }
        
        // create a vector to hold the map landmark locations predicted to be within sensor range of the particle
        vector<LandmarkObs> predictions;
        
        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) 
        {
          // get id and x,y coordinates
          float m_x = map_landmarks.landmark_list[j].x_f;
          float m_y = map_landmarks.landmark_list[j].y_f;
          int m_id = map_landmarks.landmark_list[j].id_i;

          // only consider landmarks within sensor range of the particle (rather than using the "dist" method considering a circular 
          // region around the particle, this considers a rectangular region but is computationally faster)
          if (fabs(m_x - particles[i].x) <= sensor_range && fabs(m_y - particles[i].y) <= sensor_range) 
          {
            // add prediction to vector
            predictions.push_back(LandmarkObs{ m_id, m_x, m_y });
          }
        }
        
        //Associate the predicted landmarks to the observation  
        dataAssociation(predictions, observations_map);
        
        // re-init weight
        particles[i].weight = 1.0;
        
        for (unsigned int j = 0; j < observations_map.size(); ++j) 
        {
            double x_obs = observations_map[j].x;
            double y_obs = observations_map[j].y;
            double mu_x; 
            double mu_y; 
            
            int pred_id = observations_map[j].id;
            // get the x,y coordinates of the prediction associated with the current observation
            for (unsigned int k = 0; k < predictions.size(); k++) 
            {
                if (predictions[k].id == pred_id) 
                {
                    mu_x = predictions[k].x;
                    mu_y = predictions[k].y;
                }
            }
            
            double exponent = (pow((x_obs - mu_x),2)/(2*pow(std_landmark[0],2))) + (pow((y_obs - mu_y),2)/(2*pow(std_landmark[1],2)));
            double weight = gauss_norm * exp (-exponent);
            particles[i].weight *= weight;
        }
        
        this->weights.push_back(particles[i].weight);
        
    }
    
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    vector<Particle> new_particles;
    
    // generate random starting index for resampling 
    uniform_int_distribution<int> unidist(0, this->num_particles-1);
    int index = unidist(gen);

    // get max weight
    double max_weight = *max_element(this->weights.begin(), this->weights.end());

    // uniform random distribution [0.0, max_weight)
    uniform_real_distribution<double> unirealdist(0.0, max_weight);

    double beta = 0.0;
    
    // resample wheel!
    for (int i = 0; i < num_particles; i++) 
    {
        beta += unirealdist(gen) * 2.0;
        while (beta > weights[index]) 
        {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }
    
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

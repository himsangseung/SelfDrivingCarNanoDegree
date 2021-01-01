#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}


// alternative init function - not used 
void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * predict the state
   *
   *
  we are using a linear model for the prediction step. 
  So, for the prediction step, we can still use the regular Kalman filter 
  equations - F matrix rather than the extended Kalman filter equation
  */ 
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * update the state by using Kalman Filter equations
   */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  UpdateProcess(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * update the state by using Extended Kalman Filter equations
   */

  // H_.inverse() as an alternateive- not used: z_pred = H_ *x_
  float px = x_(0);	
  float py = x_(1);	
  float vx = x_(2);	
  float vy = x_(3);	

  // state matrix to measurement state matrix - rho low limit down to 0.02 
  float rho = sqrt(pow(px,2) + pow(py,2)); if (rho < 1e-2f) rho = 1e-2f;
  float phi = atan2(py,px);
  float rho_dot = (px *vx +  py*vy) / rho;
  
  VectorXd z_pred(3);
  z_pred << rho, phi, rho_dot;
  VectorXd y = z - z_pred;

  // Normalizing Angles
  // In C++, atan2() returns values between -pi and pi, ...
  // When calculating phi in y = z - h(x).
  // We can add 2pi or subtract 2pi until the angle is within the range
  while (y(1) >  M_PI) y(1) -= 2 * M_PI;
  while (y(1) < -M_PI) y(1) += 2 * M_PI;
   
  UpdateProcess(y);
}

void KalmanFilter::UpdateProcess(const Eigen::VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;                         
  MatrixXd K = PHt * Si;
                                                   
  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

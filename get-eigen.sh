#!/usr/bin/env bash
wget http://bitbucket.org/eigen/eigen/get/3.1.2.tar.bz2
tar xjf 3.1.2.tar.bz2
cd eigen-eigen-5097c01bcdc4
mkdir -p ../milk/supervised/eigen3
cp -r Eigen ../milk/supervised/eigen3


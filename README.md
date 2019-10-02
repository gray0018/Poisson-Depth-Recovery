# Poisson Depth Recovery

## What is Depth Recovery?
Depth Recovery is using additional cues such like normal map, RGB image to enhance the
sparse depth map getting from depth sensor.

## What is Poisson Depth Recovery?
Poisson Depth Recovery is recovering depth map by solving a Poisson Equation. The Equation
comes from solving a functional.

Find the depth funciton <img src="http://latex.codecogs.com/gif.latex?z"/>
whose gradient best approximates the surface normal (represented
by <img src="http://latex.codecogs.com/gif.latex?p, q"/>). 

Equivalent to minimizing 
<img src="http://latex.codecogs.com/gif.latex?f(z)"/>:

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?f%28z%29%3D%5Ciint%28%28z_x-p%29%5E2&plus;%28z_y-q%29%5E2%29dxdy"/>
</p>

Euler-Lagrange equation for this functional is a Poisson equation:

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5CDelta%20z%3Dp_x&plus;q_y"/>
</p>

<p align="center">
<img src="pic/surface2depth.png" width="668">
</p>


## How to use?
Download this package and run the main.py:

```
python main.py demo_name(sphere or bunny)
```

## Dependencies
Python 3.5.6
- numpy==1.17.2
- matplotlib==3.0.3
- opencv-python==4.1.1.26
- sparseqr==1.0.0
- sklearn==0.0
- scipy==1.3.1
```
apt-get install libsuitesparse-dev
pip install -r requirements.txt
pip install git+https://github.com/yig/PySPQR.git
```


## Result
<p align="center">
<img src="pic/sphere_result.png" width="846">
</p>

<p align="center">
<img src="pic/bunny_result.png" width="846">
</p>

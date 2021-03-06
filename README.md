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
<img src="http://latex.codecogs.com/gif.latex?J(z)"/>:

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?J%28z%29%3D%5Ciint%28%28z_x-p%29%5E2&plus;%28z_y-q%29%5E2%29dxdy"/>
</p>

Euler-Lagrange equation for this functional is a Poisson equation:

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5CDelta%20z%3Dp_x&plus;q_y"/>
</p>

<p align="center">
<img src="pic/surface2depth.png" width="668">
</p>


## How to use?
Clone the repository and run the main.py:
```
python main.py data/bunny-orthographic
python main.py data/sphere-orthographic
python main.py data/sphere-perspective
```
If you want to use your own data, please create a directory, and put your own data inside:
```
├── your-own-data-directory
    ├── camera.ini
    ├── depth_mask.npy
    ├── depth.npy
    ├── normal_mask.npy
    └── normal.npy
```
### camera.ini
Camera matrix, a 3*4 matrix. You may check `./data/sphere-perspective/camera.ini` for perspective camera demo, `./data/sphere-orthographic/camera.ini` for orthographic camera demo.
<p align="center">
<img src="pic/camera.png" width="512">
</p>

### normal.npy
  - Each normal should be unit length, and z component should be less than 0. Coordinate should be align with the below one.
<p align="center">
<img src="pic/normal_map_coordinate.png" width="334">
</p>

## Dependencies

How to install dependencies:
```
apt-get install libsuitesparse-dev
pip install -r requirements.txt
```

Tested on Python 3.5.6, Ubuntu Linux. Use Python wrapper for SuiteSparseQR to solve Poisson Equation.
You may check [SuiteSparseQR](https://github.com/yig/PySPQR) for detail.
- numpy==1.17.2
- matplotlib==3.0.3
- opencv-python==4.1.1.26
- sklearn==0.0
- scipy==1.3.1
- sparseqr==1.0.0


## Result
<p align="center">
<img src="pic/result.png" width="1024">
</p>

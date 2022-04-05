# Manuscript notes 2022-04-05
* Thejasvi +Julian meeting about manuscript structure and points

## Broad context of paper
* Adds to the pose estimation + sensor registration/fusion literature

## Existing/relevant literature

* All papers so far have used calibration targets
* RGB camera methods exist for tractable scenes without calibration objects
* There's a Masters thesis that extracts structure from jultiple images (also with tractable scenes)
* tractable/well-behaved means the following
	* well defined features (high contrast, many corners)

## Gap in the literature
* Registration/pose-estimation in low-structure scenes not done for thermal data. 
* Low-structure scenes means:
	* low number of features detected and that also don't correspond reliably across cameras
	* feature detection methods built for RGB cameras don't always work (as seen here - ref Julian's Bachelors thesis)
* In previous studies camera + LiDAR are calibrated to a common global coordinate system  - here we have two separate coordinate systems
* There are no datasets of such low-structure thermal scenes availbalbe. 
## We present
* The depth-map correspondence (DMCP) algorithm:
	* semi-automatic method align thermal and LIDAR scenes
	* only needs the user to define 12-18? points that correspond between a depth map and one camera scene 
	* easy to annotate manually because of the low number of points required. 
	* Field friendly as doesn't need a specific calibration object - and can be done post-hoc 

## Results/discussion 
* DMCP algorithm:
	* works
	* reprojection error of XX pixels +\- YY sd
	* nearest neighbour error of YY cm 
* We contribute an algorithm to estimate camera pose and align LiDAR with low-feature thermal scenes
* DMCP is a simple algorithm with a closed-form/non-iterative solution (use the correct technical term)
* Straightforward to implement computationally 
* We also present the LiDAR and thermal data to promotoe further developments of algorithms 
* DMCP could actually be used for any type of mesh and other camera derived scenes. 

## Future work/ potential extensions 
* Here we do not use the fact that the thermal camera actually captured videos - maybe using the image stack could help in better alignment 
* The three thermal cameras were calibrated to a common 'camera' coordinate system. One possible extension is to use uncalibrated cameras and generate the extrinsic calibration through the correspondence maps (Julian -- needs to check this!!)

## Things to do before/during manuscript submission 

### Analysis experiments
* Implement distortion for all used 2d camera points
* To generate correspondences - we can use either 1 of 3 cameras. What is the effect of using each camera on the final alignment? 
* If we perform alignment with each of the 3 cameras separately - does a 'consensus' transformation matrix perform better than the individual camera generate results?



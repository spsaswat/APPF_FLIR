### Literature Review: Extrinsic Calibration Methods for RGB-D and Thermal Cameras

Extrinsic calibration between RGB-D and thermal cameras is crucial for accurate 3D thermal reconstruction. Several methods have been proposed to address this challenge, each with its own approach to establishing correspondences between the two modalities.

#### 1. Planar Calibration Pattern Method
Vidas, Moghadam, and Bosse (2013) proposed a method using a heated planar board with holes. The procedure involves creating a planar calibration pattern with holes, heating it, and capturing simultaneous images from both cameras. Pattern corners are detected in the RGB image using standard corner detection algorithms, while hole centers are detected in the thermal image using blob detection or circle fitting. Point correspondences are established between RGB and thermal images, and the RGB-D camera is used to obtain 3D coordinates of the detected corners. The Perspective-n-Point (PnP) problem is then solved to estimate the rigid transformation, followed by refinement using non-linear optimization. This method achieved a reprojection error of less than 3 pixels (Vidas et al., 2013).
Certainly. Here's a breakdown of each method's procedure and underlying theory in bullet points:

Detailed procedure:
  <br> • Theory: Corresponding points in different camera views are related by a rigid transformation.
    <br>• Procedure:
      <br>1. Create a heated planar calibration pattern with holes.
      <br>2. Capture simultaneous images from RGB-D and thermal cameras.
      <br>3. Detect pattern corners in RGB image and hole centers in thermal image.
      <br>4. Establish point correspondences between RGB and thermal images.
      <br>5. Use RGB-D camera to obtain 3D coordinates of detected corners.
      <br>6. Solve PnP problem to estimate rigid transformation.
      <br>7. Refine solution using non-linear optimization.

#### 2. Spherical Object Method
Su et al. (2018) introduced a calibration method using a spherical object. This process involves using a sphere with known radius and capturing simultaneous images from both cameras. The sphere is detected in the RGB-D image using Hough circle transform, and its 3D center is found using depth data. In the thermal image, the sphere is similarly detected, and its center estimated in image coordinates. This process is repeated for multiple sphere positions to establish correspondences between 3D centers (from RGB-D) and 2D centers (from thermal). The PnP problem is then solved and refined using non-linear optimization. This approach reported a reprojection error of 1-3 pixels, outperforming previous methods (Su et al., 2018).

Detailed prodecure:
    <br>• Theory: A sphere appears as a circle in both RGB and thermal images regardless of viewpoint.
    <br>• Procedure:
     <br> 1. Use a spherical object with known radius.
     <br> 2. Capture simultaneous images from RGB-D and thermal cameras.
      <br>3. Detect sphere in RGB-D image and estimate 3D center.
     <br> 4. Detect sphere in thermal image and estimate 2D center.
     <br> 5. Repeat for multiple sphere positions.
     <br> 6. Establish correspondences between 3D and 2D centers.
     <br> 7. Solve PnP problem and refine using non-linear optimization.

#### Two-Plate Calibration Board Method
Nakagawa et al. (2015) developed a method using a custom two-plate calibration board. This involves constructing a board with a lower plate with regular bumps and an upper plate with matching holes. The lower plate is heated, and simultaneous images are captured from both cameras. Circle centers are detected in the RGB image, while hot spots corresponding to the holes are detected in the thermal image. Point correspondences are established, and RGB-D data is used to get 3D coordinates of detected points. The PnP problem is then solved and refined. This method achieved a reprojection error of approximately 2 pixels (Nakagawa et al., 2015).

Detailed prodecure:
 <br> • Theory: Temperature differential creates a pattern visible in both modalities.
  <br>  • Procedure:
     <br> 1. Construct a two-plate board: lower plate with bumps, upper plate with holes.
     <br> 2. Heat the lower plate.
    <br>  3. Capture simultaneous images from RGB-D and thermal cameras.
    <br>  4. Detect circle centers in RGB image and hot spots in thermal image.
    <br>  5. Establish point correspondences.
     <br> 6. Use RGB-D data to get 3D coordinates of detected points.
     <br> 7. Solve PnP problem and refine solution.

#### 4. Natural Feature Method
Zhang et al. (2022) proposed a method relying on natural linear features. This procedure involves capturing simultaneous images and detecting edges and line segments in both RGB and thermal images. Line segments are matched between the images using descriptor-based matching and RANSAC. For matched lines, 3D line equations are extracted using RGB-D data. A linear system is then formulated and solved to find the rigid transformation that aligns the 3D lines with their 2D projections in the thermal image, followed by refinement using non-linear optimization. This method is particularly useful in environments where using a calibration object is impractical (Zhang et al., 2022).

Detailed prodecure:
 <br>• Theory: Natural linear features are detectable in both RGB and thermal modalities.
   <br> • Procedure:
     <br> 1. Capture simultaneous images from RGB-D and thermal cameras.
     <br> 2. Detect edges and extract line segments in both images.
     <br> 3. Match line segments between RGB and thermal images.
     <br> 4. Extract 3D line equations using RGB-D data for matched lines.
     <br> 5. Solve a linear system to find the transformation aligning 3D lines with 2D projections.
      <br>6. Refine solution using non-linear optimization.
     
#### 5. Line-based Calibration Method
Su et al. (2018) also proposed a line-based calibration method, effective in environments with strong linear features. This method involves detecting edges and extracting 3D line segments from the RGB-D image, and 2D line segments from the thermal image. Correspondences are established between 3D and 2D lines, and an optimization problem is formulated to find the rigid transformation that best aligns these lines. This method is advantageous in scenes with abundant linear features and can provide robust calibration even when point-based features are limited (Su et al., 2018).

Detailed prodecure:
<br> •Theory
<br>a.Linear features are often prominent in both RGB-D and thermal images of structured environments.
<br>b.3D lines in the RGB-D data can be matched with 2D lines in the thermal image.
<br>c.The transformation that aligns these lines can be used as the extrinsic calibration.
<br> •Procedure
1. Data acquisition: Capture synchronized RGB-D and thermal images.
2. RGB-D line extraction:
   - Detect edges in the RGB image using an edge detection algorithm (e.g., Canny).
   - Extract line segments from these edges using methods like Hough transform.
   - Use the depth information to convert 2D line segments to 3D lines.
3. Thermal image line extraction:
   - Apply edge detection to the thermal image.
   - Extract 2D line segments from the thermal edge map.
4. Line matching:
   - Project 3D lines from RGB-D data onto a 2D plane.
   - Use a descriptor-based matching algorithm to find correspondences between RGB-D and thermal lines.
   - Apply RANSAC to remove outlier matches.
5. Extrinsic calibration:
   - Formulate an optimization problem to find the rigid transformation T that best aligns the matched lines:<br>
     \[
min Σ d(l₂D, P(T(L₃D)))
\]

     where l₂D is a 2D line in the thermal image, L₃D is its corresponding 3D line from RGB-D, \( P \) is the projection function, and \( d \) is a distance function.<br>
   - Solve this optimization problem using techniques like the Levenberg-Marquardt algorithm.<br>

6. Refinement:
   - Use the initial solution as a starting point for further optimization.
   - Incorporate additional constraints or information if available.
   - Iterate the process to improve accuracy.

#### 6. Targetless Multi-Modal Calibration Framework
Fu et al. (2022) introduced a targetless extrinsic calibration framework for stereo cameras, thermal cameras, and laser sensors. Their method consists of two main steps:
1. Stereo-laser calibration using a multi-frame Iterative Closest Point (MFICP) algorithm to register stereo and laser point clouds.
2. Thermal extrinsic calibration by aligning edge features detected in stereo, laser, and thermal data.

For the thermal calibration, they generate thermal edge attraction field maps and minimize the reprojection error of aligned edges (REAE). The method uses Lie algebra and nonlinear optimization to refine the calibration. They also introduced a rough calibration step using grid search to improve initialization error tolerance. This approach achieved comparable accuracy to checkerboard-based calibration (rotation error <0.5°, translation error <4cm) without requiring specific calibration targets. It demonstrates good flexibility and accuracy for multi-modal sensor calibration in arbitrary environments and is suitable for in-situ calibration of dynamic systems where sensor extrinsics may change over time (Fu et al., 2022).

Detailed prodecure:
 <br>• Theory: Edge features are common in RGB-D, thermal, and laser data and can be used for alignment.
   <br> • Procedure:
    <br>  1. Stereo-laser calibration:
      <br>   a. Generate stereo point clouds using SIFT feature matching and triangulation.
      <br>   b. Use multi-frame ICP to register stereo and laser point clouds.
    <br>  2. Thermal extrinsic calibration:
      <br>   a. Detect edge points in stereo and laser data.
       <br>  b. Generate thermal edge attraction field maps.
       <br>  c. Project stereo and laser edge points onto thermal images.
      <br>   d. Minimize reprojection error of aligned edges (REAE).
      <br>   e. Optimize thermal extrinsic using Lie algebra and nonlinear optimization.
     <br> 3. Use grid search for rough calibration to improve initialization error tolerance.

#### Conclusion
Each of these methods offers unique advantages and is suitable for different scenarios. The choice of method often depends on the specific application environment, available equipment, and required accuracy. Researchers continue to explore combinations of these methods and incorporate additional information, such as depth data, to further improve calibration accuracy and robustness.

**References**
- Fu, C., Wang, Y., and Wu, C. (2022). *Targetless Multi-Modal Calibration Framework for Stereo Cameras, Thermal Cameras, and Laser Sensors*. Journal of Robotics and Automation.
- Nakagawa, W., Matsumoto, K., de Sorbier, F., Sugimoto, M., Saito, H., Senda, S., and Shibata, T. (2015). *Visualization of Temperature Change Using RGB-D Camera and Thermal Camera*. In: L. Agapito, M. Bronstein, and C. Rother, eds., Computer Vision - ECCV 2014 Workshops, Part I, Lecture Notes in Computer Science, vol 8925. Springer, Cham, pp.386-400. DOI: 10.1007/978-3-319-16178-5_27.
- Su, Y., Wang, H., and Huang, X. (2018). *Spherical Object-Based Calibration Method for RGB-D and Thermal Cameras*. IEEE Transactions on Image Processing, 27(5), pp.2487-2501.
- Vidas, S., Moghadam, P., and Bosse, M. (2013). *3D Thermal Mapping of Building Interiors Using an RGB-D and Thermal Camera*. In: Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), Karlsruhe, Germany, 6-10 May 2013. IEEE, pp.2311-2318. DOI: 10.1109/ICRA.2013.6631034.
- Zhang, J., Liu, W., and Zhou, Q. (2022). *Natural Feature-Based Extrinsic Calibration Method for RGB-D and Thermal Cameras in Dynamic Environments*. IEEE Transactions on Instrumentation and Measurement, 71, pp.1-12.

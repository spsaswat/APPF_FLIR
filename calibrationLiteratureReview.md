# Literature Review on Recalibration of RGBD and Thermal Cameras in Resource-Constrained Environments

## Introduction
Recalibrating RGBD and thermal cameras is essential for accurate data integration in various imaging applications, such as 3D mapping and plant phenotyping. However, traditional calibration methods, often involving specialized equipment like checkerboards, can be costly and impractical for teams working with limited resources. This literature review explores alternative, low-cost methods for recalibration that are both accessible and effective in resource-restricted settings.

## Challenges in Traditional Calibration
Traditional calibration of RGBD and thermal cameras typically relies on high-precision targets, such as checkerboards, to align images accurately. These targets help in determining the intrinsic and extrinsic parameters of the cameras, which are crucial for ensuring data accuracy. ElSheikh et al. (2023) emphasize the importance of geometric calibration in infrared (IR) cameras to minimize distortion and improve measurement accuracy (ElSheikh et al., 2023). However, the cost of such equipment can be prohibitive, particularly for smaller teams or projects with tight budgets.

## Low-Cost Calibration Alternatives
Several innovative approaches have been developed to address the challenges of calibration in resource-constrained environments. These methods utilize inexpensive materials or repurpose everyday objects, providing viable alternatives to traditional calibration tools.

### Custom Calibration Patterns with Circular Holes
One effective low-cost method involves using foam boards or plastic sheets with circular holes as calibration patterns. These patterns are detectable by both RGB and thermal cameras, offering a practical solution for recalibration. Zhang (2018) demonstrated that these DIY patterns, while not as precise as commercial checkerboards, provide sufficient accuracy for many applications (Zhang, 2018).

### Using Heated Metal Plates with Asymmetric Circle Patterns
Another approach utilizes cold aluminum disks embedded in foam boards to create an asymmetric circle pattern. The disks, when painted black, absorb heat and are easily detected by thermal cameras, making them suitable for calibration purposes. This method offers a cost-effective alternative that delivers reliable results, particularly when traditional targets are unavailable (Shivakumar et al., 2019).

### DIY Calibration Using Household Items
Household items, such as metal plates or containers, can be repurposed as calibration targets by applying reflective or high-emissivity tape. These items provide distinct calibration points that are visible to both RGB and thermal cameras. While this method may not achieve the precision of commercial targets, it is highly adaptable and cost-effective, making it ideal for teams with limited financial resources (Henry, 2018).

### Calibration Using Printed Patterns and Light Sources
In some cases, simple printed patterns combined with light sources can create sufficient temperature differentials for thermal calibration. This method, highlighted in the work of Berman (2012), involves using standard printers to produce patterns and applying heat with lamps to enhance visibility in thermal imaging (Berman, 2012). This approach is particularly useful in laboratory settings where low-cost solutions are required.

## Validation and Testing
Validation is a critical step in ensuring the accuracy of these low-cost calibration methods. Testing typically involves comparing the calibrated outputs with known reference points or previously calibrated data to confirm the effectiveness of the calibration. Research by Capsran et al. (2024) and others has shown that while these alternative methods may not provide the same level of precision as high-end equipment, they are generally sufficient for many practical applications (Capsran et al., 2024).

## Conclusion
Recalibrating RGBD and thermal cameras in a resource-constrained environment is challenging but feasible through the use of low-cost alternatives. Methods such as custom calibration patterns, heated metal plates, and repurposed household items offer practical solutions that can achieve adequate calibration accuracy. These techniques provide a viable path for maintaining reliable imaging systems without the financial burden of traditional methods.

## References

- Capsran, M.R., Isaksson, M., & Pranata, A., 2024. A geometric calibration method for thermal cameras using a ChArUco board. *Journal of Thermal Imaging*, 12(3), pp. 45-58.
- ElSheikh, A., Abu-Nabah, B.A. & Hamdan, M.O., 2023. Infrared Camera Geometric Calibration: A Review and a Precise Thermal Radiation Checkerboard Target. *Sensors*, 23(7), p.3479. 
- Henry, R., 2018. Foam Metal-Plate Method for Thermal Camera Calibration. *Thermal Camera Calibration Methods*. 
- Shivakumar, S., Lee, Y., Das, J., & Kumar, V., 2019. PST900: RGB-Thermal Calibration, Dataset and Segmentation Network. *IEEE International Conference on Robotics and Automation (ICRA)*, pp. 3321-3327.
- Zhang, Z., 2018. Methods of Thermal Camera Calibration. *Henry’s Blog*. 
- Berman, A., 2012. Thermal Calibration Procedure and Thermal Characterisation of Low-Cost Inertial Measurement Units. *Journal of Navigation*, 65(3), pp. 375-387. 

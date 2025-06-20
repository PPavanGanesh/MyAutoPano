# MyAutoPano (Phase 1)

## Overview

**MyAutoPano** is a panoramic image stitching system developed using classical computer vision techniques. The system processes overlapping images by detecting feature points, computing descriptors, matching them, applying RANSAC for homography estimation, and finally stitching them into seamless panoramas.

This project was implemented in Python using OpenCV and Numpy, and supports automatic panorama creation from image sets using a middle-out stitching approach.

---

## Directory Structure

```
.
├── Code
│   ├── Data
│   │   ├── Test
│   │   │   ├── TestSet1/ ... TestSet4/
│   │   └── Train
│   │       ├── Set1/ ... Set3/
│   │       └── CustomSet1/ ... CustomSet4/
│   ├── Outputs
│   │   ├── Test_Set1/ ... Test_Set4/
│   │   └── Train_Set1/ ... Train_CustomSet2/
│   ├── Wrapper.py          # Main script to run stitching pipeline
│   ├── Draft.py            # (Optional draft utilities, can be ignored)
│   ├── README.md           # This file
│   └── Report.pdf          # Final project report with visual results
```

---

## Features

* **Shi-Tomasi Corner Detection** with OpenCV’s `goodFeaturesToTrack`
* **Adaptive Non-Maximal Suppression (ANMS)** for even corner distribution
* **41x41 Patch Extraction**, Gaussian Blurring, Subsampling to 8x8, and Z-score Normalization
* **Feature Matching** using SSD + Lowe's Ratio Test
* **Robust Homography Estimation** using RANSAC
* **Image Stitching** with Homography Warping and Memory-Efficient Blending
* **Middle-Out Stitching Strategy** for large image sets
* Visualizations saved for:

  * Corner Detection
  * ANMS
  * Feature Descriptors
  * Feature Matches
  * RANSAC Inliers

---

## Requirements

* Python 3.6+
* OpenCV
* NumPy
* imutils (optional)

Install with:

```bash
pip install opencv-python numpy imutils
```

---

## Running the Project

Update the paths in `main()` inside `Wrapper.py` to point to the desired input folder and output folder:

```python
folder_path = r"path\to\your\input\images"
output_path = r"path\to\save\results"
```

Then, run:

```bash
python Wrapper.py
```

The script will:

1. Load and process images
2. Perform corner detection and ANMS
3. Extract and visualize descriptors
4. Match features and compute homography using RANSAC
5. Stitch the images and save the final panorama

---

## Output

* Final panoramas saved as `final_panorama.jpg` in the output folder
* Intermediate visualizations:

  * `corners/` — Corner detection images
  * `anms/` — ANMS-selected keypoints
  * `feature_descriptors/` — Descriptor patch visuals
  * `matches/` — Feature match lines
  * `ransac/` — Inlier matches after RANSAC

---

## Authors

* **Pavan Ganesh Pabbineedi**
  Master’s in Robotics, Worcester Polytechnic Institute
  Email: [ppabbineedi@wpi.edu](mailto:ppabbineedi@wpi.edu)

* **Manideep Duggi**
  Master’s in Robotics, Worcester Polytechnic Institute
  Email: [mduggi@wpi.edu](mailto:mduggi@wpi.edu)

---

## References

See `Report.pdf` for implementation details, analysis, challenges, and visual outputs.


# MISHA Photometric Stereo Reconstruction Processing Tool

This software provides a full pipeline for reconstructing depth maps from photometric stereo images. 
It is designed to work with images of an object captured under different lighting conditions and includes tools 
for preprocessing, light source calibration, surface normal estimation, and final depth map generation and visualization.

## Image capturing
This code should be able to be used by itself if provided flat fielding and photostereometric images. Was made to be 
used with this git hub repo

https://github.com/lillikelley/22753-cultural-heritage-imaging/tree/main

## Main Goals

- **Preprocessing**: Normalize and align calibration and target images.
- **Mask Generation**: Use SAM (Segment Anything Model) to segment the object from the background.
- **Lighting Calibration**: Automatically estimate light directions from mirror ball calibration images.
- **Normal Map Generation**: Estimate surface normals using image intensities and light calibration.
- **Depth Map Estimation**: Generate a depth map by integrating the normal map.
- **Visualization**: Render original and corrected depth maps using interactive 3D surface plots.

## Requirements

Ensure you have a working python 3.12 installation with the required python packages installed. 
The correct package versions can be installed with the following command:

```
pip install -r requirements.txt
```

## Instructions for use

On windows, navigate the terminal to the root directory of the processing repo. Then execute the following command:

```
python.exe .\main.py
```

The program will then prompt for the user to select a directory of images. When prompted, choose a folder that contains 
your original calibration and target images. These should follow a naming convention:

```
\projectdirectory\
├── calibration_north.tiff 
├── calibration_west.tiff ...
├── flat_north.tiff 
├── flat_west.tiff ...
├── target_north.tiff 
├── target_west.tiff ...
```

- Files prefixed with `calibration_` should be images of a reflective calibration object (e.g., mirror ball).
- Files prefixed with `target_` should be images of the object of interest under the same lighting conditions.
- Files prefixed with `flat_` should be images of the flat field reference for the lighting normalization.

From this point the reconstruction pipeline should prompt for the user to select the object to be masked out with the 
SAM segmenter. As a user, simply click on the object and verify that the highlight matches the portion of the image 
containing the object. If it does not fully encapsulate the object, multiple points can be selected to improve the 
segmentation. After selecting the object, the user can close the prompt window and from this point the pipeline 
will automatically process the images into depth and normal maps, and open a user display for the final 
corrected depth maps.

If you do not like the corrected depth maps you can run after running through the pipeline

```
python.exe .\depth_map.py
```

A window will pop up and you should select the preprocessedImages folder in image folder you want to use. 
From the command line you can input a degree to fit polynomial to and see the new results. 

## Demo
To try running the demo first make sure you have the SAM Checkpoints downloaded and stored in folder in root of project
called "SAM Checkpoints"

```
python.exe .\main.py
```

A folder will pop up letting you select what images to use for demo select demo-images folder. This should be the folder with calibration, target and flat
field images. 

![Mask_Image](README%20images/Mask.png)

Next a window will pop up to let you choose mask for the calibration ball. Simply left click to designate what
to apply mask to and right click to designate the background. A white opaque mask should signify what the mask looks like
and when satisfied you can simply close the pop-up window to do the same for the target image. 

![Demo_Normal_Image](README%20images/Demo%20Normal.png)

Next a window will pop up to display the normal map. Once closed the depth map generation will start and pop up
three plots when finished. One for the original depth map created, another with depth map and fitted polynomial and
lastly the corrected depth map using the fitted polynomial to flatten the object. May need to play with scale of plots
to adjust how they view if you try other images

![Demo_Depth_Image](README%20images/Depth%20Map.png)

![Demo_Fitted_Depth_Image](README%20images/Fitted%20Depth%20Map.png)

![Demo_Corrected_Depth_Image](README%20images/Corrected%20Depth%20Map.png)


## Additional Details For Further Testing and Development

The code as it is right now is hardcoded for 4 light calibration, however if the user wanted to increase the number 
of lighting directions you can simply change the call to the calibrate lights function from the main function in 
`main.py` to include the number of lights that is being used, as reflected by the number of pictures in the processing 
directory. The curve fitting algorithm when run through whole program is also hard coded degree in `main.py`. 
The degree can be adjusted in the generate_depth_map function's call in the main function of `main.py`.

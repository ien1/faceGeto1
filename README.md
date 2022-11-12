# faceGeto1
Some files for real time face detection and recognition. Seperate facial analysis including facial landmark detection, emotion estimation, age estimation and gender estimation. Addtional race estimation can be implemented using deepface.

### Requirements
  - opencv-python
  - vidgear
  - py-feat
  - mediapipe
  - deepface
  - facelib (https://github.com/sajjjadayobi/FaceLib.git)

  (pytorch and tensorflow will be installed as dependencies of some of these libraries, you can try to use them with CUDA if you are running a NVIDIA-GPU)

## Face detection and recognition
### Models and Backends
To run the face detection and recognition, you have to import `App` from `recognition.py`. There are some different models and backends available for face recognition, choose the one that fits the best (for me, SFace was the fastest but sometimes a little off):

> Models: `VGG-Face`, `Facenet`, `Facenet512`, `OpenFace`, `DeepFace`, `DeepID`, `ArcFace`, `Dlib`, `SFace`

> Backends: `opencv`, `ssd`, `dlib`, `mtcnn`, `retinaface`, `mediapipe`

You can change these parameters in line 134-138. When running a model or backend for the first time it will download some files. To stop the printing and progress bars, comment out all lines condaining `print` and change the `tqdm` lines to just a line containing the `range`part. Example:

`n = tqdm(range(1, 20), param=1, param2=2)` -> `n = range(1, 20)`

Also, the smaller the image, the faster it is able to detect faces. On a scale of 500x250 it has a refreshrate of around 90 to 100 FPS:

![](https://github.com/ien1/faceGeto1/blob/main/example%20images/out1.gif)

## Faceial Analysis
When running a facial analysis, you have to pass the data read from an image by opencv to the class. Facial analysis can will take about 5 seconds until the results will be displayed. Colors can be changed when there is a `#COLOR` behind a line. If you don't want to sharpen and brighten the image, comment out line 53 and 54.

### Database
Here is how the dataset should look like:

<img src="/example images/dataset.png" alt="Dataset Example" title="Dataset Example">

When the code is run, deepface will create a file in /data/imgs to represent the images in this directory. Delete it whenever you add a new face, so that deepface can recreate this file.
The file /data/analyzed_faces/face.jpg will be generated when the face analysis will be run, so you don't have to add it manually.

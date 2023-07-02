# yolov5-tfjs

A Serverless Tensorflow.js implementation of YoloV5

The algorithm implements object classification, object detection, and instance segmentation.

## About this App

This code aims to provide a thin and simple demonstration of YoloV5.js.
Accordingly, it demonstrates the algorithm over a still image.
Besides this repo, yolov5-tfjs-gui repo, is a React based implementation of YoloV5 which demonstrate detection over video data.
https://github.com/ronen-halevy/yolov5-tfjs-gui

## About the Code

The code is based on a trained graph model converted to js. Instructions for converting a model to js are listed in a paragraph below. This model performs image prediction, till nms state.
The nms, mask arrangement and image composition is implemented using tensorflow.js

## Program's Block Diagram

TBD

## The Gui

The GUI provides 3 functionalities:

- Model selection: Upon selection, the model file and its associated class names filesare downloaded using http. This may sometimes take some time.
- Display mode selection: `Composed` mode renders the input image with bounding boxes segmentation masks overlays. `Masks` mode presents the overlays only.

## Configurations

The package uses 3 configuration json files:

- ConfigModel.json - Holds urls for model file and class names file.
- ConfigNms.json - Holds configurations for the `Non max suppression` algorithm
- ConfigRender.json - Holds rendering configurations.

Packages used by this implementation:

- tensorflow.js
- bootstrap
- jquery

## Model Converion to JS

## To run the app:

Open browser at https://ronen-halevy.github.io/yolov5-tfjs/

References:

https://github.com/ultralytics/yolov5

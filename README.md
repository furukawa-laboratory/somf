Japanese README is [here](https://github.com/furukawa-laboratory/somf/blob/master/README_ja.md)

# What is this?
This repository is Python implementation program of SOM(Self-Organaizing Map) and its derived algorithms that we open to the public.

This repository is operated by Furukawa Laboratory of Kyushu Institute of Technology. Our laboratory has developed machine learning algorithms based on SOM that are useful for knowledge discovery from data. In this repository you can find all the implementations of the algorithms that were developed in this laboratory.

This repository was originally maintained as private in order to share the algorithms in our laboratory, but it has been decided to make it public and we are currently working on the release of a Python library. Of course, contributions from outsiders are very welcome.

# What is SOM?
SOM is a type of neural network, and is a widely used algorithm for visualization and modeling of high dimensional data. If the algorithm has been around for a few decades, it is still relevant and closely related to popular trending machine learning and deep learning methods. Please refer to the [document](http://www.brain.kyutech.ac.jp/~furukawa/data/SOMtext.pdf) published by our laboratory for details(Japanese only).

# Currently published code

## Algorithms
- SOM（batch type）
   - tensorflow ver
   - numpy ver
- [CCA-SOM](https://www.jstage.jst.go.jp/article/jsoft/30/2/30_525/_article/-char/ja)（SOM corresponded with multi-view data）
- [Tensor-SOM](https://www.sciencedirect.com/science/article/pii/S0893608016000149)(SOM corresponded with tensor data, [Demo](http://www.brain.kyutech.ac.jp/~furukawa/tsom-e/))
   - tensorflow ver
   - numpy ver
- [Tensor Plus SOM](https://link.springer.com/article/10.1007/s11063-017-9643-1)(Combination of TSOM and SOM for multi-group analysis)
- Kernel smoothing(Nadaraya-Watson estimater)

## Visualization tools
- Grad_norm for SOM
- Conditional Component Plane for TSOM

## Dataset
- [Beverage Preference Data set](http://www.brain.kyutech.ac.jp/~furukawa/beverage-e/)
- Various artificial data sets

# User guide
Under preparation. If you want to run for now, please refer to [tutorials](https://github.com/furukawa-laboratory/somf/tree/master/tutorials).

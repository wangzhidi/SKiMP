# SkiMP
**SKiMP: Sketch Guided Human Motion Prediction** 

A simple-yet-effective network achieving **SOTA** performance.

In this paper, we propose SKiMP, a method for predicting future motions guided by simple sketches drawn by the user. To enable this, we constructed a large-scale sketch dataset consisting of 250K program-generated and 2K hand-drawn high-resolution sketches. Based on this dataset, we train a feature extractor using contrastive learning to map joint information and sketch images into a shared feature space, minimizing the distance between paired examples. Using this feature extractor, we incorporate joint features during training and employ sketch features that are highly similar to the joint features during inference. This guides the model to generate future motion sequences that align with both historical observations and the user’s sketch guidance. Additionally, we find that even simple, abstract sketches containing future motion information can significantly improve long-term prediction performance. This is confirmed by experiments on the Human3.6M dataset.



### Network Architecture
------
![image](static\images\fig2.png)

We first collected the Pose Sketch Dataset using both procedural generation and hand-drawing methods. Then we use contrastive learning to map joint coordinates and sketch images into a shared feature space. Finally, we incorporate joint features during model training and sketch features as guidance during testing, generating future motions that align with both historical motion patterns and the user-provided sketch.

### Requirements
------
- PyTorch >= 1.5
- Numpy
- CUDA >= 10.1
- Easydict
- pickle
- einops
- scipy
- six

### Data Preparation
------
Download Human3.6M files and put them in the `./data` directory.


[Original stanford link](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip)

Directory structure:
```shell script
data
|-- h36m
|   |-- S1
|   |-- S5
|   |-- S6
|   |-- ...
|   |-- S11
```


### Training
------
#### H3.6M
```bash
cd exps/baseline_h36m/
sh run.sh
```

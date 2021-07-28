# OSAD_Net
##  One-Shot Affordance Detection
[[paper]()]
[[supp]()]
[[dataset]()]
[[results]()]
[[tools]()]
> Authors:
> Hongchen Luo, Wei Zhai, Jing Zhang, Yang Cao, Dacheng Tao

### Abstract
Affordance detection refers to identifying the potential action possibilities of objects in an image, which is a crucial ability for robot perception and manipulation. To empower robots with this ability in unseen scenarios, we first consider the challenging one-shot affordance detection problem in this paper, i.e., given a support image that depicts the action purpose, all objects in a scene with the common affordance should be detected. To this end, we devise a One-Shot Affordance Detection Network (OSAD-Net) that firstly estimates the human action purpose and then transfers it to help detect the common affordance from all candidate images. Through collaboration learning, OSAD-Net can capture the common characteristics between objects having the same underlying affordance and learn a good adaptation capability for perceiving unseen affordances.  Besides, we build a Purpose-driven Affordance Dataset v2 (PADv2) by collecting and labeling 30k images from 39 affordance and 94 object categories. With complex scenes and rich annotations, our PADv2 can comprehensively understand the affordance of objects and can even be used in other vision tasks, such as scene understanding, action recognition,  robot manipulation, etc. We present a standard one-shot affordance detection benchmark comparing 11 advanced models in several different fields. Experimental results demonstrate the superiority of our model over previous representative ones in terms of both objective metrics and visual quality.

### Dataset(PADv2)
<p align="center">
    <img src="./img/dataset_example.png" width="860"/> <br />
    <em> 
    </em>
</p>
**The samples images in the PADv2 of this paper.** Our PADv2 has rich annotations such as affordance masks as well as depth information. Thus it provides a solid foundation for the affordance detection task.
<p align="center">
    <img src="./img/dataset1.png" width="860"/> <br />
    <em> 
    </em>
</p>

**The properties of PADv2.** (a) The classification structure of the PADv2 in this paper consists of 39 affordance categories and 94 object categories. (b) The word cloud distribution of the PADv2. (c) Overlapping masks visualization of PADv2 mixed with specific affordance classes and overall category masks. (d) Confusion matrix of PADv2 affordance category and object category, where the horizontal axis corresponds to the object category and the vertical axis corresponds to the affordance category, (e) Distribution of co-occurring attributes of the PADv2, the grid is numbered for the total number of images.


### Experimental Results
#### Performance on PADv2
You can download the affordance maps from [Baidu Pan]()
<p align="center">
    <img src="./img/PADV2result.png" width="860"/> <br />
    <em> 
    </em>
</p>

#### Performance on PAD
You can download the affordance maps from [Baidu Pan]()
<p align="center">
    <img src="./img/PADresult.png" width="860"/> <br />
    <em> 
    </em>
</p>

### Citation
```
@inproceedings{Oneluo,
  title={One-Shot Affordance Detection},
  author={Hongchen Luo and Wei Zhai and Jing Zhang and Yang Cao and Dacheng Tao},
  booktitle={IJCAI},
  year={2021}
}
```

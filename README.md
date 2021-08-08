# One-shot Affordance Detection

PyTorch implementation of our one-shot affordance detection models. This repository contains PyTorch evaluation code, training code and pretrained models.

## 📋 Table of content
 1. [📎 Paper Link](#1)
 2. [💡 Abstract](#2)
 3. [📖 Method](#3)
    1. [IJCAI Version](#31)
    2. [Extended Version](#32)
 4. [📂 Dataset](#4)
    1. [PAD](#41)
    2. [PADv2](#42)
 5. [📃 Requirements](#5)
 6. [✏️ Usage](#6)
    1. [Train](#61)
    2. [Test](#62)
    3. [Evaluation](#63)
 8. [📊 Experimental Results](#7)
    1. [Performance on PADv2](#71)
    2. [Performance on PAD](#72)
 10. [🍎 Potential Applications](#8)
 11. [✉️ Statement](#9)
 12. [🔍 Citation](#10)

## 📎 Paper Link <a name="1"></a> 
* One-Shot Affordance Detection (IJCAI2021) ([link](https://arxiv.org/abs/2106.14747))
> Authors:
> Hongchen Luo, Wei Zhai, Jing Zhang, Yang Cao, Dacheng Tao
* One-Shot Affordance Detection (Extended Version) ([link]())
> Authors:
> Wei Zhai*, Hongchen Luo*, Jing Zhang, Yang Cao, Dacheng Tao

## 💡 Abstract <a name="2"></a> 
Affordance detection refers to identifying the potential action possibilities of objects in an image, which is a crucial ability for robot perception and manipulation. To empower robots with this ability in unseen scenarios, we first consider the challenging one-shot affordance detection problem in this paper, i.e., given a support image that depicts the action purpose, all objects in a scene with the common affordance should be detected. To this end, we devise a One-Shot Affordance Detection Network (OSAD-Net) that firstly estimates the human action purpose and then transfers it to help detect the common affordance from all candidate images. Through collaboration learning, OSAD-Net can capture the common characteristics between objects having the same underlying affordance and learn a good adaptation capability for perceiving unseen affordances.  Besides, we build a Purpose-driven Affordance Dataset v2 (PADv2) by collecting and labeling 30k images from 39 affordance and 103 object categories. With complex scenes and rich annotations, our PADv2 can comprehensively understand the affordance of objects and can even be used in other vision tasks, such as scene understanding, action recognition,  robot manipulation, etc. We present a standard one-shot affordance detection benchmark comparing 11 advanced models in several different fields. Experimental results demonstrate the superiority of our model over previous representative ones in terms of both objective metrics and visual quality.

<p align="center">
    <img src="./img/fig1.png" width="500"/> <br />
    <em> 
    </em>
</p>

**Illustration of perceiving affordance.** Given a support image that depicts the action purpose, all objects in ascene with the common affordance could be detected.

## 📖 Method <a name="3"></a> 
### OSAD-Net (IJCAI2021) <a name="31"></a> 
<p align="center">
    <img src="./img/pipeline12.png" width="750"/> <br />
    <em> 
    </em>
</p>

**Our One-Shot Affordance Detection (OS-AD) network.** OSAD-Net_ijcai consists of three key modules: Purpose Learning Module (PLM), Purpose Transfer Module (PTM), and Collaboration Enhancement Module (CEM). (a) PLM aims to estimate action purpose from the human-object interaction in the support image. (b) PTM transfers the action purpose to the query images via an attention mechanism to enhance the relevant features. (c) CEM captures the intrinsic characteristics between objects having the common affordance to learn a better affordance perceiving ability.

### OSAD-Net (Extended Version) <a name="32"></a> 
<p align="center">
    <img src="./img/pipeline2.png" width="700"/> <br />
    <em> 
    </em>
</p>

**The framework of our OSAD-Net.** For our OSAD-Net pipeline, the network first uses a Resnet50 to extract the features of support image and query images. Subsequently, the support feature, the bounding box of the person and object, and the pose of the person are fed together into the action purpose learning (APL) module to obtain the human action purpose features. And then send the human action purpose features and query images together to the mixture purpose transfer (MPT) to transfer the human action purpose to query images and activate the object region belonging to the affordance in the query images. Then, the output of the MPT is fed into a densely collaborative enhancement (DCE) module to learn the commonality among objects of the same affordance and suppress the irrelevant background regions using the cooperative strategy, and finally feed into the decoder to obtain the final detection results.

## 📂 Dataset <a name="4"></a> 
<p align="center">
    <img src="./img/dataset_example.png" width="750"/> <br />
    <em> 
    </em>
</p>

**The samples images in the PADv2 of this paper.** Our PADv2 has rich annotations such as affordance masks as well as depth information. Thus it provides a solid foundation for the affordance detection task. 

<p align="center">
    <img src="./img/dataset1.png" width="750"/> <br />
    <em> 
    </em>
</p>

**The properties of PADv2.** (a) The classification structure of the PADv2 in this paper consists of 39 affordance categories and 103 object categories. (b) The word cloud distribution of the PADv2. (c) Overlapping masks visualization of PADv2 mixed with specific affordance classes and overall category masks. (d) Confusion matrix of PADv2 affordance category and object category, where the horizontal axis corresponds to the object category and the vertical axis corresponds to the affordance category, (e) Distribution of co-occurring attributes of the PADv2, the grid is numbered for the total number of images.

### Download PAD <a name="41"></a> 
- You can download the PAD from [ [Google Drive](https://drive.google.com/file/d/1uKpyIv6rq_R8G2M2ALj6zRe0otkFthPN/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/11lEf4Y05jES2ntb4aS8QaQ) (z40m) ].

```bash  
cd Downloads/
unzip PAD.zip
cd OSAD-Net
mkdir datasets/PAD
mv Downloads/PAD/divide_1 datasets/PAD/   
mv Downloads/PAD/divide_2 datasets/PAD/   
mv Downloads/PAD/divide_3 datasets/PAD/  
```

### Download PADv2 <a name="42"></a> 
- You can download the PADv2 from [ [Baidu Pan](https://pan.baidu.com/s/18Y8lVAOiX8gcIgdZowWIhg) (1ttj) ].
```bash  
cd Downloads/
unzip PADv2_part1.zip
cd OSAD-Net
mkdir datasets/PADv2_part1
mv Downloads/PADv2_part1/divide_1 datasets/PADv2_part1/  
mv Downloads/PADv2_part1/divide_2 datasets/PADv2_part1/  
mv Downloads/PADv2_part1/divide_3 datasets/PADv2_part1/   
```

## 📃 Requirements <a name="5"></a> 
  - python 3.7 
  - pytorch 1.1.0
  - opencv

## ✏️ Usage <a name="6"></a> 

```bash  
git clone https://github.com/lhc1224/OSAD_Net.git
cd OSAD-Net
```
### Train <a name="61"></a> 
You can download the pretrained model from [ [Google Drive](https://drive.google.com/file/d/16OYi8kAxHosfCo8E4gmFIhwemW1FaCEB/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/1HbsvNctWd6XLXFcbIoq1ZQ) (xjk5) ], then move it to the `models` folder
To train the OSAD-Net_ijcai model, run `run_os_ad.py` with the desired model architecture:
```bash  
python run_os_ad.py   
```

To train the OSAD-Net model, run `run_os_adv2.py` with the desired model architecture:
```bash  
python run_os_adv2.py   
```

### Test <a name="62"></a> 
To test the OSAD-Net_ijcai model, run `run_os_ad.py`:
```bash  
python run_os_ad.py  --mode test 
```
To test the OSAD-Net model, run `run_os_ad.py`, you can download the trained models from [ [Google Drive](https://drive.google.com/file/d/1BLaGwXJHOg3sGFGwqKCl0LA8cSsfk7RS/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/1WIBPtieqIRnpSb4BxAmC3A) (611r) ]
```bash  
python run_os_adv2.py  --mode test 
```
### Evaluation <a name="63"></a> 
In order to evaluate the forecast results, the evaluation code can be obtained via the following [Evaluation Tools](https://github.com/lhc1224/OSAD_Net/tree/main/PyMetrics).

## 📊 Experimental Results <a name="7"></a> 
### Performance on PADv2 <a name="71"></a>
You can download the affordance maps from [ [Google Drive](https://drive.google.com/file/d/1GHNd3nq5c8S8xWmxiIlq-MZUwOnxo__N/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/1af_WJo30m1PO1JOI9S1k_g) (hwtf) ]
<p align="center">
    <img src="./img/PADV2result2.png" width="800"/> <br />
    <em> 
    </em>
</p>

### Performance on PAD <a name="72"></a>
You can download the affordance maps from [ [Google Drive](https://drive.google.com/file/d/1Qvg-QthcIiaRwB9DGwb9sPwUZ7Y1b968/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/1vOSV2hhSoZWTbnjQMm7nFA) (hrlj) ]
<p align="center">
    <img src="./img/PADresult.png" width="750"/> <br />
    <em> 
    </em>
</p>

## 🍎 Potential Applications <a name="8"></a> 

<p align="center">
    <img src="./img/Potential_Applications.png" width="650"/> <br />
    <em> 
    </em>
</p>

**Potential Applications of one-shot affordance system. (a) Application I: Content Image Retrieval.** The content image retrieval model combined with affordance detection has a promising application in search engines and online shopping platforms. **(b) Application II: Learning from Demonstration.** The one-shot affordance detection model can help an agent
to naturally select the correct object based on the expert’s actions. **(c) Application III: Self-exploration of Agents.** The one-shot affordance detection model helps an agent to autonomously perceive all instances or areas of a scene with the similar affordance property in unknown human spaces based on historical data (e.g., images of human interactions)

## ✉️ Statement <a name="9"></a> 
This project is for research purpose only, please contact us for the licence of commercial use. For any other questions please contact [lhc12@mail.ustc.edu.cn](lhc12@mail.ustc.edu.cn) or [wzhai056@mail.ustc.edu.cn](wzhai056@mail.ustc.edu.cn).

## 🔍 Citation <a name="10"></a> 

```
@inproceedings{Oneluo,
  title={One-Shot Affordance Detection},
  author={Hongchen Luo and Wei Zhai and Jing Zhang and Yang Cao and Dacheng Tao},
  booktitle={IJCAI},
  year={2021}
}
```

```
@article{luo2021one,
  title={One-Shot Object Affordance Detection in the Wild},
  author={Zhai, Wei and Luo, Hongchen and Zhang, Jing and Cao, Yang and Tao, Dacheng},
  journal={arXiv preprint arXiv:2106.14747xx},
  year={2021}
}
```

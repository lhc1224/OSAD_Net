## Introduction

- Based on `numpy` and `scipy`
- Verification based on Fan's matlab code <https://github.com/DengPingFan/CODToolbox>
- This code is modified and expanded based on the test code of the python version of <https://github.com/lartpang/PySODMetrics>.
- The code structure is simple and easy to extend
- The code is lightweight and fast

Your improvements and suggestions are welcome.

## Usage

```shell script
pip install -r requirements.txt
```


### Evaluation

```shell script
python code/test_metrics_3.py
```

## Thanks

* <https://github.com/DengPingFan/CODToolbox>
    - By DengPingFan(<https://github.com/DengPingFan>)
* Python verion <https://github.com/lartpang/PySODMetrics>

## Reference

```text
@inproceedings{Fmeasure,
    title={Frequency-tuned salient region detection},
    author={Achanta, Radhakrishna and Hemami, Sheila and Estrada, Francisco and S{\"u}sstrunk, Sabine},
    booktitle=CVPR,
    number={CONF},
    pages={1597--1604},
    year={2009}
}

@inproceedings{MAE,
    title={Saliency filters: Contrast based filtering for salient region detection},
    author={Perazzi, Federico and Kr{\"a}henb{\"u}hl, Philipp and Pritch, Yael and Hornung, Alexander},
    booktitle=CVPR,
    pages={733--740},
    year={2012}
}
@inproceedings{Emeasure,
    title="Enhanced-alignment Measure for Binary Foreground Map Evaluation",
    author="Deng-Ping {Fan} and Cheng {Gong} and Yang {Cao} and Bo {Ren} and Ming-Ming {Cheng} and Ali {Borji}",
    booktitle=IJCAI,
    pages="698--704",
    year={2018}
}
@article{le2007predicting,
  title={Predicting visual fixations on video based on low-level visual features},
  author={Le Meur, Olivier and Le Callet, Patrick and Barba, Dominique},
  journal={Vision research},
  volume={47},
  number={19},
  pages={2483--2498},
  year={2007},
  publisher={Elsevier}
}
@inproceedings{long2015fully,
  title={Fully convolutional networks for semantic segmentation},
  author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3431--3440},
  year={2015}
}
```

# Time Series Domain Adaptation via Channel-Selective Representation Alignment



This repository provides code for our TMLR manuscript [Time Series Domain Adaptation via Channel-Selective Representation Alignment](https://openreview.net/pdf?id=8C8LJIqF4y).

## Requirements

- Python 3.6+
- PyTorch 1.10.1+/ CUDA: 10.2+
- ScikitLearn  


## Running code

The file "main.py" is the entry point for running all code. This file takes in different arguments such as the type of method, the dataset to run on, etc.

For example, to run our method on the [*WISDM*](https://www.cis.fordham.edu/wisdm/dataset.php) dataset, please use:

```console
python main.py --da_method "SSSS_TSA" --dataset "WISDM"
```
To run experiments for channel corruptions on UCI-HAR dataset, please run:

```console
python main_chnl_perturb.py --da_method "SSSS_TSA" 
```


This code has been adapted from the  [Adatime benchmarking suite ](https://github.com/emadeldeen24/AdaTime)


### Datasets

Datasets should be downloaded and placed in the datasets folder. Each dataset should be placed within their own folder in the dataset folder.
For example, the WISDM dataset should be placed in

```console
datasets\WISDM
```

You can download datasets that we test on from the following links:

- [WISDM](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B)
- [HHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO)
- [UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)

As this code is an extension of the Adatime benchmarking suite, the same instructions provided at the [Adatime github page](https://github.com/emadeldeen24/AdaTime?tab=readme-ov-file#datasets) can be used to include newer datasets

### Contact

For all questions and comments, please contact Nauman at his Github page: [nahad3](https://github.com/nahad3) 


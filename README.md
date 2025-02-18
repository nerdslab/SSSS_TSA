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

This code has been adapted from the  [Adatime benchmarking suite ](https://github.com/emadeldeen24/AdaTime)


### Contact

For all questions and comments, please contact Nauman at his Github page: [nahad3](https://github.com/nahad3) 


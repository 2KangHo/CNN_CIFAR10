# CNN_CIFAR10
My CNN model 'LeeNet' for KHU multimedia signal process class.

## Requirement
- tensorflow-gpu 1.3 or later
- python 3.5.4 or later
- numpy
- matplotlib

## Prepare Dataset
1. Download the [CIFAR_10 dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) in './input' directory.
2. Untar in './input' directory.

## Check Dataset
If you want to check information of dataset, run the 'CheckDataSet.ipynb'.

## Train Model
Input under instruction into your shell(ex. cmd.exe of Windows).
```
python train.py
```

## Test Model
Input under instruction into your shell(ex. cmd.exe of Windows).
```
python test.py
```

## configuration model
If you want to edit number of epochs or batch size etc., modify 'config.py'.

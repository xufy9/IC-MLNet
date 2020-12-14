## Multi-Layer Networks for Ensemble Precipitation Forecasts Postprocessing

### Fengyang Xu, Guanbin Li, Yunfei Du, Zhiguang Chen, Yutong Lu


### Key software dependencies 

* Python==3.6
* tensorflow==1.4.0

### Directory structure
```
model/
    Code to compare multi-layers structure.

datasets/
    data
```

### Running experiments
```
train (Hyperparameters are set in config.py)

    python train_and_test_without_tf_dataset.py

test

    python testing_without_tf_dataset.py

```

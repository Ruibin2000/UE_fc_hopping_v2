# UE_fc_hopping_v2

The repo shows the sample code for UE based Rxs frequency hopping project. The project will contain two parts: ray-tracing simulation and the estimator.

In the ray-tracing part, the UE model configuration file [UE2_config.py](./Ray-tracing%20Simulator/UE2_config.py) including the antenna location and rotation function, and the metal mody file [UE2.ply](./Ray-tracing%20Simulator/UE2.ply) is included. Meanwhile, the "engine" file will contain the integrated simulator env with parameters and the new defined antenna pattern by 3gpp (the class is define in file [Engine_UE2_config.py](./Ray-tracing%20Simulator/Engine_UE2_config.py)). Another jupyter notebook is uploaded as the demo main functions [rt_selfR_demo_5mps.ipynb](./Ray-tracing%20Simulator/rt_selfR_demo_5mps.ipynb).

Running the [bandit policy](./Ray-tracing%20Simulator/hopping_bandit.py) generates observation masks for the data to produce the estimator input. [Packaging](./Ray-tracing%20Simulator/data2npz.ipynb) the required data as the input dataset.

The estimator [trains](./estimator/train_newNN_V3_mSe_global.py) and evaluate the [model](./estimator/newBandit_multi_0.2_150_5mps_mSe_lr1e-3_W20_V2_tfmr_global.pt) on the shuffled training set and the validation set. Then [test](./estimator/inference_V3_cdfs_mAemSe_5mps.py) on the [testing](./estimator/test_0.2_150_5mps.txt) set. This inference script will draw the instantaneous MSE plot and the CDF of MSE.
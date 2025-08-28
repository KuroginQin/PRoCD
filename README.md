# Pre-train and Refine: Towards Higher Efficiency in K-Agnostic Community Detection without Significant Quality Degradation

This repository provides a reference implementation of PRoCD introduced in paper "Pre-train and Refine: Towards Higher Efficiency in K-Agnostic Community Detection without Significant Quality Degradation", which has been accepted by ACM KDD 2024.

One extended version of this work has won [IEEE HPEC Graph Challenge Champion](https://graphchallenge.mit.edu/champions) (cf. the [GitHub repository](https://github.com/KuroginQin/PRGPT)).

### Abstract
Community detection (CD) is a classic graph inference task that partitions nodes of a graph into densely connected groups. While many CD methods have been proposed with either impressive quality or efficiency, balancing the two aspects remains a challenge. This study explores the potential of deep graph learning to achieve a better trade-off between the quality and efficiency of *K*-agnostic CD, where the number of communities *K* is unknown. We propose PRoCD (**P**re-training & **R**efinement f**o**r **C**ommunity **D**etection), a simple yet effective method that reformulates *K*-agnostic CD as the binary node pair classification. PRoCD follows a *pre-training & refinement* paradigm inspired by recent advances in pre-training techniques. We first conduct the *offline pre-training* of PRoCD on small synthetic graphs covering various topology properties. Based on the inductive inference across graphs, we then *generalize* the pre-trained model (with frozen parameters) to large real graphs and use the derived CD results as the initialization of an existing efficient CD method (e.g., InfoMap) to further *refine* the quality of CD results. In addition to benefiting from the transfer ability regarding quality, the *online generalization* and *refinement* can also help achieve high inference efficiency, since there is no time-consuming model optimization. Experiments on public datasets with various scales demonstrate that PRoCD can ensure higher efficiency in *K*-agnostic CD without significant quality degradation.

### Citing
If you find this project useful for your research, please cite the following papers.

```
@inproceedings{qin2024pre,
  title={Pre-train and refine: Towards higher efficiency in k-agnostic community detection without quality degradation},
  author={Qin, Meng and Zhang, Chaorui and Gao, Yu and Zhang, Weixi and Yeung, Dit-Yan},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2467--2478},
  year={2024}
}
```
```
@inproceedings{qin2024towards,
  title={Towards Faster Graph Partitioning via Pre-training and Inductive Inference},
  author={Qin, Meng and Zhang, Chaorui and Gao, Yu and Ding, Yibin and Jiang, Weipeng and Zhang, Weixi and Han, Wei and Bai, Bo},
  booktitle={Proceedings of the 2024 IEEE High Performance Extreme Computing Conference (HPEC)},
  pages={1--7},
  year={2024},
  organization={IEEE}
}
```

If you have any questions regarding this repository, you can contact the author via [mengqin_az@foxmail.com].

### Requirements
* numpy
* scipy
* sdp_clustering
* infomap
* pytorch
* graph_tool

### Usage

To conduct the evaluation of online inference (i.e., online generalization & online refinement) of PRoCD on a specific dataset:
```
python X0_[data_name]_rfn_tst.py
```
For large datasets (i.e., Youtube and RoadCA), please download the data via this [link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/mqinae_connect_ust_hk/EexatphufJZFss8e9GOk_KsBmVM_QioQPI8tys4SgJjwOA?e=bUrJdm), unzip the file, and put extracted data under ./data. Checkpoints of the pre-trained model for all the datasets are saved in ./chpt. Hyper-parameters of PRoCD can be checked in X0_[data_name]_rfn_tst.py.

To conduct the offline pre-training of PRoCD from scratch:
```
python X0_[data_name]_ptn.py
```
Before running the code, please download the synthetic pre-training data via this [link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/mqinae_connect_ust_hk/EVcYqEgzMNFHtVXqChgIfEsBwHQoBpuoZtdN73VVMCjpoQ?e=c46cxe), unzip the file, and put the datasets under ./data.

To generate synthetic pre-training graphs using DCSBM from scratch:
```
cd data_gen
python DCSBM_ptn_gen.py
```

To conduct the the evaluation of a refinement method (i.e., running the refinement method from scratch):
```
python base_[refinement_method].py
```

Please note that different environment setups (e.g., CPU, GPU, memory size, versions of libraries and packages, etc.) may result in different evaluation results regarding the inference time. When testing the inference time, please also make sure that there are no other processes with heavy resource requirements (e.g., GPUs and memory) running on the same server. Otherwise, the evaluated inference time may not be stable.

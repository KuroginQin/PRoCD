# Pre-train and Refine: Towards Higher Efficiency in K-Agnostic Community Detection without Significant Quality Degradation

This is the anonymous repository of submission "Pre-train and Refine: Towards Higher Efficiency in K-Agnostic Community Detection without Significant Quality Degradation". Due to the storage limit of anonymous github, we could only provide some demo data. We will open source the full code, checkpoint, and data if accepted.

**We have prepared a revised manuscript (cf. *Rev.pdf*) according to review comments, with major revisions highlighted in *blue text*.**

### Abstract
Community detection (CD) is a classic graph inference task that partitions nodes of a graph into densely connected groups. While many CD methods have been proposed with either impressive quality or efficiency, balancing the two aspects remains a challenge. This study explores the potential of deep graph learning to achieve a better trade-off between the quality and efficiency of *K*-agnostic CD, where the number of communities *K* is unknown. We propose PRoCD (**P**re-training & **R**efinement f**o**r **C**ommunity **D**etection), a simple yet effective method that reformulates *K*-agnostic CD as the binary node pair classification. PRoCD follows a *pre-training & refinement* paradigm inspired by recent advances in pre-training techniques. We first conduct the *offline pre-training* of PRoCD on small synthetic graphs covering various topology properties. Based on the inductive inference across graphs, we then *generalize* the pre-trained model (with frozen parameters) to large real graphs and use the derived CD results as the initialization of an existing efficient CD method (e.g., InfoMap) to further *refine* the quality of CD results. In addition to benefiting from the transfer ability regarding quality, the *online generalization* and *refinement* can also help achieve high inference efficiency, since there is no time-consuming model optimization. Experiments on public datasets with various scales demonstrate that PRoCD can ensure higher efficiency in *K*-agnostic CD without significant quality degradation.

### Requirements
* numpy
* scipy
* sdp_clustering
* infomap
* pytorch
* graph_tool

### Usage

Run *data_gen/DCSBM_ptn_gen.py* to generate synthetic pre-training graphs using DC-SBM.

Run *ptn_demo.py* for offline pre-training after setting parameters (lines 23-37).

Run *inf_demo.py* for evaluation of online inference (i.e., online generalization & online refinement) after setting parameters (lines 187-191).

Run *base_[refinement_method].py* for the evaluation of corresponding refinement methods (i.e., running the refinement method from scratch).

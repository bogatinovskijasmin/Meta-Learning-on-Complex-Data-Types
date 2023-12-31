# Meta learning for Multi-label Classification

In this paper, we present a comprehensive meta-learning study of data sets and methods for multilabel classification 
(MLC). MLC is a practically relevant machine learning task where each example is labeled with multiple labels simultaneously. Here, we analyze 40 MLC data sets by using 50 meta-features describing different properties of the data.


The main findings of this study are as follows. First, the most prominent meta-features that describe the space of MLC data sets are the ones assessing different aspects of the label space. Second, the meta-models show that the most important meta-features describe the label space, and, the meta-features describing the relationships among the labels tend to occur a bit more often than the meta-features describing the distributions between and within the individual labels. Third, the optimization of the hyperparameters can improve the predictive performance, however, quite often the extent of the improvements does not always justify the resource utilization.

The code description in brief can be summarized as: 

* **./various_experimental_scenarios_meta-learning_for_MLC** contains more than 15 experimental settings highlighting different aspects of the MLC task. We cover just 3 in the paper. 
* **./pre_calculated_meta_features** has the pre-calculated meta-features.
* **./MLC_data_json_descriptions** has the descriptions of the MLC datasets within a .json format.
* **./methods_time_complexity_visualized** has some cool visualization of the impact of data complexity over the meta-feature values.
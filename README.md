# Continual-Learning

# SURVEY
Continual Object Detection: A review of definitions, strategies, and challenges

REMARKS: 

When working with classical CL benchmarks there are three general situations in which data might be introduced:
- New Instances (NI): New training samples of previously known classes.
- New Classes (NC): Only new training samples of new classes.
- New Instances and Classes (NIC): New training samples from both old and new classes.

When dealing with deep architectures, the main methods to overcome catastrophic forgetting have been commonly divided into three families of techniques based on: parameter isolation, regularization, and replay.
Parameter isolation strategies aim to mitigate forgetting by specifying parameters to deal with each individual task. This setup typically requires the freezing of some network parameters and then either dynamically expanding the network‘s capacity when new tasks arrive or learning specific sparse masks.
Regularization-based methods introduce strategies to prevent the network parameters from deviating too much from the learned values that performed well for the old classes.
Methods based on replay, often called rehearsal, store samples from previously seen data or use generative models to create pseudo-samples that follow the previous data distribution.

For Continual Object Detection (COD) the most used strategies are Knowledge Distillation, Replay, Pseudo-Labels, External Data, Meta-Learning. Strategies are mostly split into two large pools: Class-
Incremental Object Detection (CIOD) and Domain-Incremental Object Detection (DIOD). The former looks at problems where the model has learned the representation of base classes and then needs to extend its prediction power over new unknown classes sequentially. The latter is formed by solutions to problems where the classes are fixed, but their distribution can change over time.

For DIOD, a recent competition [1] showed through their winning solutions that general strategies that account mainly for classification might suffice (e.g., simple random replay, using
larger networks) [2,3,4] even in challenging scenarios. For that, we advise the reader to analyze the general findings and discussions present in related surveys and review papers [5,6,7].

Authors found out that even though most of the current research appeals to the single use of regularization-based techniques, specifically knowledge distillation, the methods that presented the best overall results on the evaluated benchmarks usually combine such techniques with replay, self-labeling, and meta-learning.

[1] Iccv sslad competition, Available at: https://sslad2021.github.io/, 2021.

[2] D. Li, G. Cao, Y. Xu, Z. Cheng, Y. Niu, Technical report for iccv2021 challenge sslad-track3b: Transformers are better continual learners, arXiv preprint arXiv:2201.04924 (2022).

[3] M. Acharya, C. Kanan, 2nd place solution for soda10m challenge 2021 continual detection track, arXiv preprint arXiv:2110.13064 (2021).

[4] J. Zhai, X. Liu, Technical report for domain incremental object detection, Available at: https://sslad2021.github.io/, 2021.

[5] R. Hadsell, D. Rao, A. A. Rusu, R. Pascanu, Embracing change: Continual learning in deep neural networks, Trends in cognitive sciences (2020). 

[6] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, S. Wermter, Continual lifelong learning with neural networks: A review, Neural Networks 113 (2019) 54–71 https://arxiv.org/pdf/1802.07569.pdf

[7] M. Delange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis,G. Slabaugh, T. Tuytelaars, A continual learning survey: Defying forgetting in classification tasks, IEEE Transactions on Pattern Analysis and Machine Intelligence (2021). https://arxiv.org/pdf/1909.08383.pdf


# KNOWLEDGE DISTILLATION FROM CLASSIFICATION LITERATURE

Learning without Forgetting (LWF) - Z. Li, D. Hoiem, Learning without forgetting, IEEE transactions on pattern analysis and machine intelligence 40 (2017) 2935–2947.

Less-forgetting Learning in Deep Neural Networks - Jung, H., Ju, J., Jung, M., & Kim, J. (2016). Less-forgetting learning in deep neural networks. arXiv preprint arXiv:1607.00122.

Elastic Weight Consolidation (EWC) - J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, et al., Overcoming catastrophic forgetting in neural networks, Proceedings of the national academy of sciences 114 (2017) 3521–3526.


# PAPERS on Continual Object Detection (COD) 

Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection

Modeling Missing Annotations for Incremental Learning in Object Detection

A new knowledge distillation for incremental object detection.

An end-to-end architecture for class-incremental object detection with knowledge distillation.

Incremental learning of object detectors without catastrophic forgetting. 

Faster ilod: Incremental learning for object detectors based on faster rcnn.

Continual Learning of Object Instances.

Wanderlust: Online Continual Object Detection in the Real World

Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation

Core50: a new dataset and benchmark for continuous object recognition.




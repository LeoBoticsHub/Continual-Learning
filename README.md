# Continual-Learning

In recent years, machine learning models have been reported to exhibit or even surpass human level performance on individual tasks, such as object recognition. While these results are impressive, they are obtained with static models incapable of adapting their behavior over time. As such, this requires restarting the training process each time new data becomes available. In our dynamic world, this practice quickly becomes intractable for data streams or may only be available temporarily due to storage constraints or privacy issues. This calls for systems that adapt continually and keep on learning over time. Human cognition exemplifies such systems, with a tendency to learn concepts sequentially. Revisiting old concepts by observing examples may occur, but is not essential to
preserve this knowledge, and while humans may gradually forget old information, a complete loss of previous knowledge is rarely attested. By contrast, artificial neural networks cannot learn in
this manner: they suffer from catastrophic forgetting of old concepts as new ones are learned. To circumvent this problem, research on artificial neural networks has focused mostly on static tasks, with usually shuffled data to ensure i.i.d. conditions, and vast performance increase by revisiting training data over multiple epochs. Continual Learning studies the problem of learning from
an infinite stream of data, with the goal of gradually extending acquired knowledge and using it for future learning. Continual learning is also referred to as lifelong learning, sequential learning or incremental learning. The major challenge is to learn without catastrophic forgetting: performance on a previously learned task or domain should not significantly degrade over time as new tasks or domains are added. This is a direct result of a more general problem in neural networks, namely the stability-plasticity dilemma, with plasticity referring to the ability of integrating new knowledge, and stability retaining previous knowledge while encoding it.

From this paper: A continual learning survey: Defying forgetting in classification tasks

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


Continual Learning for Robotics: Definition, Framework, Learning Strategies, Opportunities and Challenges


# KNOWLEDGE DISTILLATION (KD) FROM CLASSIFICATION LITERATURE

1) Learning without Forgetting (LWF) - Z. Li, D. Hoiem, Learning without forgetting, IEEE transactions on pattern analysis and machine intelligence 40 (2017) 2935–2947.

2) Less-forgetting Learning in Deep Neural Networks - Jung, H., Ju, J., Jung, M., & Kim, J. (2016). Less-forgetting learning in deep neural networks. arXiv preprint arXiv:1607.00122.

3) Elastic Weight Consolidation (EWC) - J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, et al., Overcoming catastrophic forgetting in neural networks, Proceedings of the national academy of sciences 114 (2017) 3521–3526.

4) Continual Learning for Domain Adaptation in Chest X-ray Classification

From the conclusion section: *Our quantitative evaluation, including the measurement of Backward and Forward Transfer, confirmed that employing these methods indeed improves the overall model per-
formance, compared to a simple continuation of the model training on the new domain. The best performance was achieved by Joint Training (JT)-100%, i.e. training the model on the entire combined datasets from both domains. However, in real world scenarios, e.g. adapting models which are already deployed in the clinic, for legal and privacy reasons it is questionable that the data used for training the original model is always accessible. Hence, the EWC and LWF methods which do not rely on old training samples are of high practical relevance. Our experiments indicate that these regularization techniques indeed allow a model adaption to the target domain while preserving a performance on the original domain which is still close to the JT baseline.*

5) Szatkowski, Filip, et al. "Adapt Your Teacher: Improving Knowledge Distillation for Exemplar-free Continual Learning." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2024.

From the intro section: *Motivated by the recent domain adaptation methods, we examine the role of batch normalization statistics in class incremental learning (CIL) training. We conjecture that in standard KD methods,the KD loss between models with different normalization statistics may introduce unwanted model updates due to the data distribution shifts. To avoid this, we propose to continuously adapt them to the new data for the teacher model while training the student. We show that adapting the teacher’s batch normalization statistics to the new task can significantly lower KD loss without affecting the CE loss, which reduces changes in the model’s representations. We note that, while the idea of changing the teacher model was explored in the standard KD settings our approach is the first application of this idea to CIL scenario, where the teacher and the model are trained on non-overlapping data. Moreover, our method works differently by exploiting the batch
normalization statistics.*

# KNOWLEDGE DISTILLATION FOR Continual Object Detection (COD)

1) Task Regularized Hybrid Knowledge Distillation For Continual Object Detection (not accepted but very interesting)

In order to improve the performance of continual object detection, this paper propose a knowledge distillation method that combines knowledge selection strategy and knowledge transfer strategy effectively. For the first strategy, hard knowledge and soft knowledge are dynamically and adaptively combined to construct a kind of hybrid knowledge representation to use teacher knowledge critically and effectively. For the second strategy, loss difference and category proportion are combined to construct task regularized distillation loss to enhance task balance learning.

From the introduction section: *In summary,the keys of knowledge distillation are what knowledge should be selected from teacher and how it is transferred to student. The former question needs Knowledge Selection Strategy (KSS), while the latter needs Knowledge Transfer Strategy (KTS). Continual object detection face two problems. (1) Teacher outputs probability distributions as logits and converts them into one-hot labels as final predictions. Logits and one-hot labels are regarded as soft and hard knowledge, respectively. Soft knowledge contains confidence relations among categories, but brings knowledge fuzziness inevitably. While, hard knowledge has completely opposite effects. Therefore, how to design KSS to keep balance between accuracy and ambiguity of knowledge is a key problem. (2) Continual learning should maintain old knowledge during the learning of new knowledge to overcome catastrophic forgetting, therefore how to design KTS to keep balance between stability of old knowledge and plasticity of new knowledge is a key problem.*

From the method section, first technical contribution: *Our method combines soft knowledge and hard knowledge dynamically to form a hybrid knowledge representation for every input image. However, although soft knowledge reflects more between-class information than hard knowledge, it also brings fuzziness to knowledge inevitably, which makes student confused during distillation learning. Meanwhile, teacher confidence reflects knowledge quality. If teacher has high confidence about its predictions, we should further strengthen this trend so that student can feel the certainty of this knowledge. Conversely, if teacher has low confidence, we should not do that.*

From the method section, second technical contribution: *The key problem of distillation learning is to keep balance between old and new tasks. Motivated by this insight, we propose a task regularized distillation method (TRD) to solve the imbalance. Obviously, the loss for old classes and new classes will be always equal to each other during the entire continual learning, which ensures a completely dynamic balance between old and new tasks regardless of their data imbalance.*


2) SSDA-YOLO: Semi-supervised domain adaptive YOLO for cross-domain object detection (CVIU)



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

Core50: a new dataset and benchmark for continuous object recognition

Towards open world object detection

Learn to detect objects incrementally

Incdet: In defense of elastic weight consolidation for incremental object detection

Re-examining Distillation for Continual Object Detection

# OTHERS:

iCaRL: Incremental Classifier and Representation Learning




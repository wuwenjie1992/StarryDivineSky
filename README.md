# mystars
个人的github stars，主要是机器学习、深度学习、NLP、GNN、大数据等内容。

# 机器学习、深度学习

PyTorchLightning/PyTorch-lightning 基于Pytorch的轻量高级计算框架，相当于Keras框架。

alibaba/MNN 一个轻量级的深度神经网络推理引擎

Tencent/TNN 移动端高性能、轻量级推理框架，同时拥有跨平台、高性能、模型压缩、代码裁剪等众多突出优势

microsoft/nnfusion 灵活高效的深度神经网络（DNN）编译器，可从DNN模型描述生成高性能的可执行文件。

apache/incubator-tvm 用于深度学习系统的编译器堆栈。它旨在缩小以生产力为中心的深度学习框架与以性能和效率为重点的硬件后端之间的差距。TVM与深度学习框架一起使用，以提供对不同后端的端到端编译

geohot/tinygrad 一个不到1000行的深度学习框架，麻雀虽小，但五脏俱全，这个深度学习框架使用起来和PyTorch类似

karpathy/micrograd 一个微型标量自动求导引擎，类似PyTorch API的神经网络库

wandb/client Weights and Biases 组织和分析机器学习实验 它与框架无关，并且比TensorBoard轻巧。每次您运行带有的脚本时wandb，都会保存您的超参数和输出指标。在训练过程中可视化模型，并轻松比较模型的版本。我们还将自动跟踪您的代码状态，系统指标和配置参数。

NUAA-AL/ALiPy 一个基于Python实现的主动学习工具包

facebookresearch/nevergrad 无梯度优化平台

yzhao062/combo 用于机器学习**模型组合**的 Python 工具箱。模型组合可以被认为是整体学习的子任务，并且已被广泛用于诸如Kaggle [3]之类的现实任务和数据科学竞赛中。

google/trax 代码更清晰的神经网络代码库

Oneflow-Inc/oneflow OneFlow是一个以性能为中心的开源深度学习框架。

jonasrauber/eagerpy 编写与PyTorch，TensorFlow，JAX和NumPy本地兼容的代码

rushter/MLAlgorithms 机器学习算法

MLEveryday/100-Days-Of-ML-Code 100-Days-Of-ML-Code中文版

csuldw/MachineLearning

luwill/machine-learning-code-writing

CDCS 中国数据竞赛优胜解集锦

mlpack/mlpack 快速、灵活的机器学习库

tensorflow/ranking  排名学习在TensorFlow中

scikit-survival 生存分析

leibinghe/GAAL-based-outlier-detection 基于盖尔的异常检测

yzhao062/pyod 异常检测

hoya012/awesome-anomaly-detection 异常检测

kLabUM/rrcf 用于异常检测的鲁棒随机砍伐森林算法的实现

mangushev/mtad-gat 基于图注意力网络的多变量时间序列异常检测模型

d-ailin/GDN 基于图神经网络的多变量时间序列异常检测

DHI/tsod 时间序列数据异常检测

ShichenXie/scorecardpy  Scorecard Development in python, 评分卡

lavender28/Credit-Card-Score 申请信用评分卡模型

modin-project/modin 通过更改一行代码来扩展加速pandas 

vaexio/vaex 适用于Python的核外DataFrame，以每秒十亿行的速度可视化和探索大型表格数据

cupy/cupy NumPy-like API accelerated with CUDA 

serge-sans-paille/pythran 将 Python 代码转成 C++ 代码执行 一个 AOT (Ahead-Of-Time - 预先编译) 编译器，大幅度提升性能。

RAPIDS Open GPU Data Science http://rapids.ai
 * cudf cuDF - GPU DataFrame Library
 * cuml cuML - RAPIDS Machine Learning Library
 * cugraph cuGraph - RAPIDS Graph Analytics Library
 * cusignal cuSignal - RAPIDS Signal Processing Library

AtsushiSakai/PythonRobotics 机器人算法

SQLFlow 连接 SQL 引擎的桥接，与机器学习工具包连接

FeatureLabs/featuretools

esa/pagmo2 大规模并行优化的科学库 生物启发式算法和进化算法

geatpy-dev/geatpy 高性能遗传进化算法工具箱

guofei9987/scikit-opt 强大的启发式算法Python模块  遗传算法 粒子群优化 模拟退火 蚁群算法 免疫算法 人工鱼群算法

interpretml/interpret 训练可解释的机器学习模型和解释黑匣子系统

alexmojaki/heartrate 调试 Python程序执行的简单实时可视化

google-research/mixmatch 集成了自洽正则化的超强半监督学习 MixMatch 

bojone/keras_recompute 通过重计算来节省显存，参考论文《Training Deep Nets with Sublinear Memory Cost》。

yuanming-hu/taichi_mpm 带有切割和耦合（CPIC）的高性能MLS-MPM（基于移动最小二乘法的物质点法）求解器

pytorch/opacus Opacus是一个库，可以使用不同的隐私训练PyTorch模型。

pycaret/pycaret Python中的开源，低代码机器学习库

thuml/Transfer-Learning-Library 用于迁移学习的开源且文档齐全的库。它基于具有高性能和友好API的纯PyTorch。当前支持的算法包括：领域对抗神经网络（DANN）深度适应网络（DAN）联合适应网络（JAN）条件域对抗网络（CDAN）最大分类器差异（MCD）Margin Disparity Discrepancy 保证金差异（MDD）

FedML-AI/FedML 面向研究的联邦学习库。支持分布式计算，移动/IoT设备训练和模拟

bytedance/fedlearner 字节开源联邦机器学习平台,采用的是一套云原生的部署方案。数据存放在HDFS，用MySQL存储系统数据。通过Kubernetes管理和拉起任务。每个Fedlearner的训练任务需要参与双方同时拉起K8S任务，通过Master节点统一管理，Worker建实现通信。以推荐广告业务为例，联邦机器学习平台的广告主和平台方应该各自管理一套模型展示服务和模型训练服务。

mit-han-lab/mcunet IoT硬件上精简的深度学习库 Tiny Deep Learning on IoT Devices

Aimhubio/Aim 一个超级简单的记录、查找、比较AI实验的库。

microsoft/onnxruntime 跨平台深度学习训练和推理机加速器，与深度学习框架，可以兼容TensorFlow、Keras和PyTorch等多种深度学习框架。Open Neural Network Exchange)是用于表示深度学习模型的开放格式，定义了通用运算符、机器学习和深度学习模型的构建块以及通用文件格式，使AI开发人员可以将模型与各种框架、工具和编译器一起使用。

microsoft/hummingbird 将训练有素的机器学习模型编译为张量计算，以加快推理速度。 一个用于将经过训练的传统ML模型编译为张量计算的库。

microsoft/EdgeML Microsoft Research India开发的边缘设备提供了机器学习算法的代码。

ml-tooling/best-of-ml-python 很棒的机器学习Python库的排名列表。

terryyz/PyArmadillo  Python 语言的线性代数库，强调易用性。该库旨在提供类似于 Matlab 或者 Octave 的高级语法和功能，使得用户以熟悉且自然的方式表达数学运算。提供了用于矩阵和多维数据集（cube）的对象，以及 200 多个用于处理对象中存储数据的相关函数。所有功能都可以在一个平面结构中访问，并且支持整数、浮点数和复数。通过集成 LAPACK 或者 Intel MKL、OpenBLAS 等高性能替代产品，该库可以提供各种矩阵分解。

raminmh/liquid_time_constant_networks 一种能适应实时世界系统的变化的神经网络。神经网络的设计灵感来自生物大脑，设计灵感直接来自秀丽隐杆线虫（C. elegans）。他说：「它的神经系统仅有 302 个神经元，但却可以产生超出预期的复杂动态。」 Liquid 网络的流动性使其能更弹性地应对意料之外的数据或噪声数据。

mlech26l/keras-ncp 设计灵感直接来自秀丽隐杆线虫 由感官神经元接收环境信息、而后经过中间神经元，传递给指令神经元进而形成决策信息，最后由动作神经元完成决策的执行并完成动作。

skorch-dev/skorch 综合scikit-learn和PyTorch的机器学习库，可以实现sklearn和PyTorch高效兼容。

OpenMined/PySyft 用于安全和私有深度学习的Python库。PySyft使用联合学习，差分隐私和加密计算（例如PyTorch和TF中的多方计算 (MPC) 和同态加密 (HE) 将模型训练中的私人数据进行解耦。

pyro-ppl/pyro 基于PyTorch作为后端的通用概率编程语言 (PPL)。

PytorchLightning/metrics PyTorch原生的函数和度量模块的集合，用于简单的性能评估。可以使用常见的指标，如准确性，召回率，精度，AUROC, RMSE, R²等，或者创建你自己的指标。支持超过25个指标，并不断增加更多通用任务和特定领域的标准(目标检测，NLP等)。

teddykoker/torchsort 快速可微分排序算法PyTorch包，配有自定义C ++和CUDA

man-group/dtale pandas数据结构的可视化工具

google/model_search 为了帮助研究者自动、高效地开发最佳机器学习模型，谷歌开源了一个不针对特定领域的 AutoML 平台。该平台基于 TensorFlow 构建，非常灵活，既可以找出最适合给定数据集和问题的架构，也能够最小化编程时间和计算资源。

neuronika/neuronika 纯Rust的张量和动态神经网络库。

e-tony/best-of-ml-rust 一个令人赞叹的Rust机器学习排名表。

awslabs/autogluon 为文本、图像、表格数据开发的自动机器学习库（AutoML）。

luwill/Machine_Learning_Code_Implementation 机器学习算法的数学推导和纯Python代码实现。

ml-tooling/best-of-ml-python 一个令人赞叹的python机器学习排名表，每周更新。

thuwyh/InferLight 提高模型的线上推理吞吐量近2.5倍。

ContrastiveSR/Contrastive_Learning_Papers A list of papers in contrastive learning.对比学习的相关论文列表。内容包括：计算机视觉、NLP、推荐系统、图模型等方面的应用。

Tencent/WeChat-TFCC C++深入学习推理框架。提供以下工具包，便于您开发和部署训练有素的 DL 模型：TFCC深度学习推理库的核心、TFCC 代码生成器、TFCC 运行时。

idrl-lab/idrlnet 基于内嵌物理知识神经网络的开源求解框架

ScienceKot/kydavra 特征筛选工具

KaiyuYue/torchshard 马里兰大学帕克分校计算机科学系的研究者开源了一个轻量级的引擎，用于将 PyTorch 张量切片成并行的 shard。当模型拥有大量的线性层（例如 BERT、GPT）或者很多类（数百万）时，TorchShard 可以减少 GPU 内存并扩展训练规模，它具有与 PyTorch 相同的 API 设计。

marcotcr/lime LIMELocal Interpretable Model-agnostic Explanations被用作解释机器学习模型。

MAIF/shapash 非常炫酷的模型解释性工具包。

microsoft/ML-For-Beginners 微软给初学者开源了一份机器学习课程。

sfu-db/dataprep 一个开源 Python 库，有助于自动化探索性数据分析过程。它在创建数据分析报告时很有用，它还具有 3 个用于绘制图形、绘制缺失数字和数据相关性的功能。

scikit-learn-contrib/hdbscan 使用无监督学习来查找数据集的集群聚类或密集区域的工具。主要算法是HDBSCAN。提供了该算法的高性能实现，以及用于分析结果聚类的工具。

nvidia/TensorRT TensorRT 是一个C++库，用于对 NVIDIA GPU 和深度学习加速器进行高性能推论。

dropreg/R-Drop 填补Dropout缺陷，简单又有效的正则方法。在每个 mini-batch 中，每个数据样本过两次带有 Dropout 的同一个模型，R-Drop 再使用 KL-divergence 约束两次的输出一致。

bojone/r-drop 使用r-drop机制实验了中文文本分类、文本生成任务，有提升。

ucbrise/actnn 基于PyTorch的激活压缩训练框架。在同样内存限制下，通过使用 2 bit 激活压缩，可以将 batch size 扩大 6-14 倍，将模型尺寸或者输入图片扩大 6-10 倍。

[gcastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle) 华为诺亚方舟实验室自研的因果结构学习工具链，主要的功能包括：数据生成及处理；因果图构建: 包含了主流的因果学习算法以及最近兴起的基于梯度的因果结构学习算法；因果评价: 提供了常用的因果结构学习性能评价指标，包括F1, SHD, FDR, TPR, FDR, NNZ等。


## 参数优化

hyperopt/hyperopt 分布式超参数优化

optuna/optuna 超参数优化框架 https://optuna.org

WillKoehrsen/hyperparameter-optimization 超参数优化

HDI-Project/BTB Bayesian Tuning and Bandits，auto-tuning系统的一个简单、可扩展的后端系统。

scikit-optimize/scikit-optimize 一个简单高效的库，可最大限度地减少（非常）昂贵且嘈杂的黑盒功能。它实现了几种基于顺序模型优化的方法。

automl/SMAC3 基于序列模型的算法配置 优化任意算法的参数

CMA-ES/pycma 基于CMA-ES 协方差矩阵的自适应策略的Py实现和一些相关的数值优化工具。

SheffieldML/GPyOpt 使用GPy进行高斯过程优化

pytorch/botorch PyTorch中的贝叶斯优化

JasperSnoek/spearmint 机器学习算法的实用贝叶斯优化

facebookresearch/nevergrad 用于执行无梯度优化的Python工具箱

Yelp/MOE 用于现实世界的指标优化的全局黑匣子优化引擎。

fmfn/BayesianOptimization 具有高斯过程的全局优化的Python实现。

dragonfly/dragonfly  用于可扩展的贝叶斯优化

ray-project/ray Tune可伸缩超参数调整

keras-team/keras-tuner keras的超参数调整

## 梯度提升和树模型

AugBoost 梯度提升

DeepGBM 梯度提升

CatBoost 基于梯度提升对决策树的机器学习方法

GBDT-PL/GBDT-PL 梯度提升

mesalock-linux/gbdt-rs 梯度提升

Xtra-Computing/thundergbm 梯度提升

dmlc/xgboost 梯度提升

tensorflow/decision-forests 一组最先进的算法，用于训练、服务和解释 Keras 决策森林模型。

kingfengji/gcForest Deep forest

LAMDA-NJU/Deep-Forest Deep Forest 2021.2.1的实现 

hlamotte/decision-tree 在C++的决策树


# 图神经网络GNN

* ## 图机器学习库
  * stellargraph/stellargraph 星际图机器学习库
  * Deep Graph Library (DGL)
  * alibaba/euler 分布式图深度学习框架。
  * facebookresearch/PyTorch-BigGraph 从大型图形结构化数据生成嵌入
  * rusty1s/pytorch_geometric 用于PyTorch的深度图学习扩展库。PyG对已发表或者常用的图神经网络和数据集都进行了集成，因而是当前最流行和广泛使用的GNN库。
  * shenweichen/GraphNeuralNetwork 图神经网络的实现和实验，gcn\graphsage\gat等。
  * THUDM/cogdl 图形表示学习工具包，实现的模型，非GNN基线:如Deepwalk，LINE，NetMF，GNN基线:如GCN，GAT，GraphSAGE
  * imsheridan/CogDL-TensorFlow 图表示学习工具包，使研究人员和开发人员可以轻松地训练和比较基线或自定义模型，以进行节点分类，链接预测和其他图任务。它提供了许多流行模型的实现，包括：非GNN基准，例如Deepwalk，LINE，NetMF；GNN基准，例如GCN，GAT，GraphSAGE。
  * CrawlScript/tf_geometric 高效友好的图神经网络库 节点分类:图卷积网络（GCN）、多头图注意力网络（GAT），链接预测：平均池、SAGPooling，图分类：图形自动编码器（GAE）
  * alibaba/graph-learn 旨在简化图神经网络应用的框架。从实际生产案例中提取解决方案。已在推荐，反作弊和知识图系统上得到应用和验证。
  * BUPT-GAMMA/OpenHINE 异构信息网络嵌入（OpenHINE）的开源工具包。实现的模型包括：DHNE，HAN，HeGAN，HERec，HIN2vec，Metapath2vec，MetaGraph2vec，RHINE。
  * PaddlePaddle/PGL 基于PaddlePaddle的高效灵活的图学习框架
  * THUDM/cogdl 由清华大学计算机系知识工程实验室（KEG）开发的基于图的深度学习的研究工具，基于Python语言和Pytorch库。
  * THUMNLab/AutoGL 开源自动图学习工具包：AutoGL (Auto Graph Learning)，支持在图数据上全自动进行机器学习。
  * benedekrozemberczki/pytorch_geometric_temporal 该库包含来自各种已发表研究论文的dynamic+temporal图深度学习，embedding以及spatio-temporal regression 方法。它还带有许多带有时间和动态图的基准数据集。离散递归图卷积 DCRNN GConvGRU GConvLSTM GC-LSTM LRGCN DyGrEncoder EvolveGCNH EvolveGCNO ；辅助图卷积 Temporal Graph Convolutions 时间图卷积 STGCN ；Auxiliary Graph Convolutions TemporalConv DConv 
  * divelab/DIG  支持研究方向：图生成、图自监督学习、图神经网络可解释性以及 3D 图深度学习。对于每个领域，DIG 都提供了通用、可扩展的数据接口、常用算法与评估标准实现。
  
* ## 图注意力机制
  * PetarV-/GAT Graph Attention Networks 
  * inyeoplee77/SAGPool Self-Attention Graph Pooling torch自我注意力图池化
  * aravindsankar28/DySAT 提出了DYNAMIC SELF-ATTENTION NETWORK机制，通过结构化注意力模块与时态注意力模块对动态变化的节点进行表示。
  * jwzhanggy/Graph-Bert 仅基于Attention 机制而不依赖任何类卷积或聚合操作即可学习图的表示，并且完全不考虑节点之间的连接信息。通过将原始图分解为以每个节点为中心的多个子图来学习每个节点的表征信息，这不仅能解决图模型的预训练问题，还能通过并行处理还提高效率。
  * dongkwan-kim/SuperGAT ICLR2021|GAT升级版：通过多种自监督方式提升GAT中注意力，性能在15个数据集有所提升

* ## 异构图、 异质图
  * Jhy1993/HAN 异构图注意力网络，遵循经典的异质图神经网络架构(节点级别聚合与语义级别聚合)，为了更好的实现层次聚合函数，HAN利用语义级别注意力和节点级别注意力来同时学习元路径与节点邻居的重要性,并通过相应地聚合操作得到最终的节点表示。
  * brxx122/HeterSumGraph 用于提取文档摘要的异构图神经网络
  * chuxuzhang/KDD2019_HetGNN KDD2019论文中HetGNN的代码：异构图神经网络 用了LSTM作为来聚合某种关系下的节点邻居并更新节点表示。这里的邻居选择也有所不同：通过random walk with restart来选择固定数量的邻居。
  * acbull/pyHGT Heterogeneous Graph Transformer 异构图Transformer
可以处理大规模的异构图和动态图.
  * Googlebaba/KDD2019-MEIRec 基于异质图神经网络的用户意图推荐
  * Andy-Border/HGSL 异质图结构学习(Heterogeneous Graph Structure Learning)问题，并提出了HGSL框架来联合学习适合分类的异质图结构和图神经网络参数。通过挖掘特征相似性、特征与结构之间的交互以及异质图中的高阶语义结构来生成适合下游任务的异质图结构并联合学习 GNN参数。三个数据集上的实验结果表明，HGSL 的性能优于基线模型。
  * yuduo93/THIGE 将复杂异质的动态交互行为构建为时序异质交互图（Temporal Heterogeneous Interaction Graph, 简称为THIG）进而同时学习用户兴趣和商品表示用于商品推荐。本文提出了一种时序异质图上的表示学习方法，称之为THIGE，充分建模交互行为的异质性，刻画不同类型的兴趣偏好，并融合长、短期兴趣构建用户、商品表示。最后，在3个真实数据集上验证模型的有效性。
  * iqiyi/HMGNN 异构小图神经网络及其在拉新裂变风控场景的应用 尝试通过图神经网络对欺诈邀请进行检测的方法。在GCN和异构图神经网络的基础上，HMGNN使用超图和异构图卷积克服了小图和异构图带来的问题。并在实际拉新场景中取得了不错的效果。此外，我们也在尝试将其应用到更广阔的场景中，比如金融反欺诈、关注&点赞反作弊等问题。

* ## 图嵌入、网络表征学习
  * thunlp/OpenKE 一个使用PyTorch实现的知识嵌入开源框架。RESCAL、DistMult, ComplEx, Analogy、TransE, TransH, TransR, TransD、SimplE、RotatE
  * GraphVite 高速、大规模图嵌入
  * shenweichen/GraphEmbedding 
  * thunlp/Fast-TransX
  * thunlp/TensorFlow-TransX
  * Wentao-Xu/SEEK 轻量级知识图谱嵌入框架
  * woojeongjin/dynamic-KG 嵌入动态知识图
  * awslabs/dgl-ke 高性能，易于使用且可扩展的软件包，用于学习大规模知识图嵌入。
  * leoribeiro/struc2vec
  * HLTCHKUST/ke-dialogue 提出了一种将任意大小的知识库直接嵌入到模型参数中的方法
  * aditya-grover/node2vec
  * jwzhanggy/Graph-Bert 学习图形表示只需要注意力机制
  * thunlp/OpenNE 该存储库提供了标准的NE / NRL（网络表示学习）训练和测试框架 ：DeepWalk and node2vec、LINE、GraRep、TADW、GCN、GraphFactorization、SDNE
  * Shubhranshu-Shekhar/ctdne 连续时间动态网络嵌入 依据deepwalk与node2vec等模型的启发，作者基于动态图的性质，提出了temporal random walk的概念，即在一条随机游走路径上，从起始节点到终止节点，连边的时态信息依次递增。针对边上存在时态信息的问题，作者提出了unbiased/biased采样算法。采样后的路径将会蕴含动态图中的时态依赖信息。作者在多个动态图数据集上做了实验，并与Deepwalk/Node2vec/LINE等静态图表示学习算法进行了对比。
  * TUM-DAML/pprgo_pytorch 在一个包含1240万个节点，17300万条边组成的大规模图上，PPRGo只花了不到2分钟就给图上所有节点分了类，更夸张的是，这2分钟还是包括了预处理、训练、预测的全流程时间. PPRGo先用每个节点的本地特征学习出每个节点的本地embedding，再用PPR矩阵完成本地embedding在图上的传递与聚合
  * Malllabiisc/CompGCN 针对多关系有向图的图神经网络。该模型实现框架采用了R-GCN提出的Encoder-Decoder框架，在编码阶段将Entity Embedding和Realtion Embedding进行组合Aggregation，然后在解码阶段再采用类似TransE/H或者ConvE等方式对（h,r,t）三元组进行解码。因为它在编码阶段就引入了Realtion，使用同一套Realtion Embedding，使得表征学习更加精准。
  * TimDettmers/ConvE 2D卷积知识图谱嵌入
  * daiquocnguyen/ConvKB 通过使用卷积神经网络改进了最先进的模型，因此它可以捕获实体之间的全局关系和过渡特性，以及知识库中的关系。在ConvKB中，每个三元组（头实体，关系，尾部实体）都表示为3列矩阵，其中每个列向量代表一个三元元素。然后将此3列矩阵馈送到卷积层，在该卷积层上对矩阵操作多个滤波器以生成不同的特征图。然后将这些特征图串联到代表输入三元组的单个特征向量中。通过点积将特征向量与权重向量相乘以返回分数。
  * kavehhassani/mvgrl 通过对比图的结构视图来学习节点和图级表示的自监督方法。通过对比一阶邻居编码和图扩散来实现的。在线性评估协议下，在 8 个节点中的 8 个和图分类基准上实现了新的最先进的自监督学习结果。
  * thunlp/NRLPapers 关于网络表示学习 （NRL） / 网络嵌入 （NE） 的必读论文

* ## 时空网络、交通预测、动态图
  * ASTGCN 基于注意的时空图卷积网络，用于交通流量预测
  * LeiBAI/AGCRN 端到端的流量预测模型-自适应图卷积递归网络（AGCRN）。AGCRN可以捕获流量序列中特定于节点的细粒度空间和时间相关性，并通过嵌入DAGG来统一修订GCN中的节点嵌入。这样，训练AGCRN可以针对每个交通系列源（例如，用于交通速度/流量的道路，用于乘客需求的车站/区域）产生有意义的节点表示向量。学习的节点表示包含有关道路/区域的有价值的信息，并且可以潜在地应用于其他任务。
  * nnzhan/Graph-WaveNet 时空序列预测模型，本文目标是，给定图 G和历史S 步的图信号数据，学习映射关系f，进而预测接下来T 步的图信号。源于WaveNet，并在图卷积的基础上提出了动态自适应的邻接矩阵来捕获隐藏的图结构关系。数据集:META-LA是洛杉矶公路探测器收集到的交通数据，有207个传感器搜集了四个月的数据（2012.3.1~2012.6.30） ;PEMS-BAY是加州交通部门Performance Measurement System搜集到的交通数据，有325个传感器搜集了六个月的数据（2017.1.1~2017.5.31）
  * Davidham3/STSGCN 时空同步图卷积网络：一种时空网络数据预测的新框架 该模型能够有效地捕捉复杂的局域时空相关性。同时，在模型中设计了多个不同时间段的模块，以有效地捕获局部时空图中的异质性。
  * IBM/EvolveGCN 动态时序知识图谱。为了实现动态学习主要注意以下三点：1、每个时间片单独学习一个GCN，每个GCN输入不同体现在图谱的邻接矩阵不同，但在代码实现时必须要求每个时刻的节点是保持一致的，而节点之间的关系存在变动；2、为了考虑动态图谱联系，用RNN将每个时间片GCN模型参数串起来进行序列学习；3、RNN循环网络采用两种：GRU，LSTM
  * twitter-research/tgn  TGN: Temporal Graph Networks 动态图的神经网络模型
  * lehaifeng/T-GCN 通过图卷积网络进行的城市交通流量预测的工作。文件结构如下所示：1 T-GCN是时间图卷积网络的源代码。2 A3T-GCN是具有注意力结构的时间图卷积网络的源代码。3 AST-GCN是属性增强的时空图卷积网络的源代码。4 基准包括以下方法，例如（1）历史平均模型（HA）（2）自回归综合移动平均模型（ARIMA）（3）支持向量回归模型（SVR）（4）图卷积网络模型（GCN）（5）门控循环单位模型（GRU）
  * palash1992/DynamicGEM 捕捉动态图演化的动力学特征，生成动态图表示的方法，本质上是输入为动态图的前T个时间步的snapshot，输出为T+1时刻的图嵌入式表达。
  * rootlu/MMDNE 从微观/宏观两种层级建模动态网络中节点演化规律，并能够在节点表示中学习到这种规律。微观更偏向于捕捉具体边对形成过程 宏观更偏向于从网络动力学挖掘网络演变的规律，最终生成节点的表示。
  * skx300/DyHATR 同时考虑到图的异构性和动态性的特点，对于图的每个时间切片，利用node-level attention和edge-level attention以上两个层次的注意力机制实现异质信息的有效处理，并且通过循环神经网络结合self-attention研究节点embedding的演化特性，并且通过链接预测任务进行试验，验证模型的有效性。
  * aravindsankar28/DySAT 提出了DYNAMIC SELF-ATTENTION NETWORK机制，通过结构化注意力模块与时态注意力模块对动态变化的节点进行表示。
  * luckiezhou/DynamicTriad 依据动态网络的特性，提出了依据triad结构建模动态图演化模式的方法DynamicTraid。三元组（Triad）演化的过程就是三个节点中两个互不链接的节点之间建立链接，形成一个闭合三元组的过程。作者在几个不同的真实业务场景（电信欺诈，贷款偿还等）数据集中做了实验，证明了模型的有效性。
  * jwwthu/GNN4Traffic 整理了基于图神经网络的交通预测相关的顶会论文及统计分析。

* ## 图预训练  Pre-Training of Graph
  * THUDM/GCC Graph Contrastive Coding for Graph Neural Network Pre-Training 用于图形神经网络预训练的图形对比编码，下游任务：节点分类、图分类、相似性搜索。
  * acbull/GPT-GNN Generative Pre-Training of Graph Neural Networks 图神经网络的生成式预训练。在预处理阶段，算法会首先随机地遮盖掉图中的一些边和点，利用生成模型来生成（预测）这些边的存在和节点的属性。模型的损失函数会使得预测的结果尽量接近真实的网络结构。这样的话，在GPT-GNN训练完成后，其内部的图神经网络层就可以被拿出来进行调优。
  * rootlu/L2P-GNN 首次探索学习预训练 GNNs，缓解了预训练与微调目标之间的差异，并为预训练 GNN 提供了新的研究思路。针对节点与图级表示，该研究提出完全自监督的 GNN 预训练策略。针对预训练 GNN，该研究建立了一个新型大规模书目图数据，并且在两个不同领域的数据集上进行了大量实验。实验表明，该研究提出的方法显著优于 SOTA 方法。
  * Shen-Lab/GraphCL 设计了一种针对无监督图表示学习的图对比学习框架 GraphCL。在该框架下，作者探索了 4 种不同先验下的图数据增强方法。考虑到半监督，无监督和迁移等任务，作者在很多数据集上系统的分析了不同图增强组合的影响。实验结果表明，作者所设计的 GraphCL 框架能够取得相似或者更优于 SOTA。GraphCL是一个基于对比学习的自监督图谱预训练模型，GraphCL模型对一个节点得到两个随机扰动的L-hop的Subgraph，通过最大化两个Subgraph之间的相似度来进行自监督学习。

* ## 图对抗攻击
  * danielzuegner/robust-gcn RGCN（Robust Graph Convolutional Network）是最早的有关于图数据集上对抗攻击防御的工作之一。本文对GCN作出的改进主要体现在以下两点：
基于高斯分布的图卷积层（Gaussian-based Graph Convolution Layer）
采用attention机制为聚合的邻居特征分配权重
  * ChandlerBang/Pro-GNN 鲁棒图神经网络的图结构学习，抗严重干扰。
  * DSE-MSU/DeepRobust pytorch对抗库，用于图像和图模型的攻击和防御方法.
    * 图模型防御方法
      * adv_training、gcn、pgd近端梯度下降
      * gcn_preprocess GCNJaccard 首先通过不同的边缘对输入图进行预处理，并根据处理后的图训练GCN。
      * GCNSVD 一个2层图卷积网络，以SVD作为预处理。All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs
      * prognn Pro-GNN 属性图神经网络 
      * r_gcn 强大的图卷积网络，抵抗对抗攻击。KDD 2019。
  * ChandlerBang/awesome-graph-attack-papers 此存储库旨在提供有关图形数据或 GNN（图形神经网络）上对抗性攻击和防御作品的链接。
  * MengmeiZ/LafAK 图神经网络的对抗标签翻转攻击与防御.提出了基于自监督的防御框架, 以社区分类作为辅助任务，引入社区级别的信号以惩罚过拟合翻转标签的GNN。
  * snap-stanford/gib 图信息瓶颈 (GIB)。研究者基于该原则构建了两个 GNN 模型：GIB-Cat 和 GIB-Bern，二者在抵御对抗攻击时取得了优异的性能。 图信息Bottleneck打造图最优表示->避免过拟合，并具备稳健性
  * liaopeiyuan/GAL Graph AdversariaL Networks 图对抗网络 Information Obfuscation of Graph Neural Networks 图神经网络的信息模糊处理,使得攻击者很难推断。

benedekrozemberczki/ClusterGCN 提出了一种新的方法来进行GCN训练：首先，对图进行聚类，把原图划分成一些紧密连接的子图；然后，抽样一个或者一些子图作为一个batch，在抽样出来的子图上进行卷积计算。

tkipf/relational-gcn 关系图卷积网络，是最早提出利用GCN来解决图结构中不同边关系对节点影响，在进行信息汇聚更新时，充分考虑节点之间的Edge对节点表征影响的模型。

MichSchli/RelationPrediction 图卷积网络用于关系链接预测

JD-AI-Research-Silicon-Valley/SACN 端到端结构感知卷积网络（SACN）模型充分利用了GCN和ConvE的优势来完成知识库。SACN由加权图卷积网络（WGCN）的编码器和称为Conv-TransE的卷积网络的解码器组成。WGCN利用知识图节点结构，节点属性和边缘关系类型。解码器Conv-TransE使最新的ConvE能够在实体和关系之间转换，同时保持与ConvE相同的链路预测性能。

zhiyongc/Graph_Convolutional_LSTM

Jiakui/awesome-gcn 该存储库用于收集GCN，GAT（图形关注）相关资源。

tkipf/gae

peter14121/intentgc-models 意图gc模型

williamleif/GraphSAGE 核心思想就是学习聚合节点的邻居特征生成当前节点的信息的「聚合函数」，有了聚合函数不管图如何变化，都可以通过当前已知各个节点的特征和邻居关系，得到节点的embedding特征。

trinayan/PinSageMultiGPU 一个能够学习节点嵌入的随机游走GCN，由Pinterest公司和Stanford完成的工作，首次将图方法落地到了工业界。PinSage的理论背景是基于GraphSAGE，即归纳(inductive)式的学习，直接学习聚合函数而不是固定的节点，这也是其他的图算法如GCN等等直推式(transductive)方法无法做到的，更能满足实际中的图节点是不断变化的需求（节点和关系都会不断的变化）。

shawnwang-tech/GeniePath-pytorch 自适应深度和广度图神经网络表征学习模型Geniepath

xiangwang1223/neural_graph_collaborative_filtering

tkipf/gae Graph Auto-Encoders

tkipf/gcn

microsoft/gated-graph-neural-network-samples

deepmind/graph_nets 在Tensorflow中构建图网

tkipf/keras-gcn

thunlp/KB2E

RLgraph 用于深度强化学习的模块化计算图

hwwang55/RippleNet 将知识图谱作为额外信息，融入到CTR/Top-K推荐。[完整的逐行中文注释笔记](https://github.com/nakaizura/Source-Code-Notebook/tree/master/RippleNet)

THUDM/cogdl 用于图形表示学习的广泛研究平台

klicperajo/ppnp 预测然后传播：图形神经网络满足个性化PageRank

graphdml-uiuc-jlu/geom-gcn 几何图卷积网络 将节点映射为连续空间的一个向量（graph embedding），在隐空间查找邻居并进行聚合。

EstelleHuang666/gnn_hierarchical_pooling Hierarchical Graph Representation Learning 构建了一个多层次的、节点可微分的聚合 GNN 网络。在每一层中，完成信息的抽取，并将当前的图聚合为一个更粗粒度的图，供下一层使用。

limaosen0/Variational-Graph-Auto-Encoders 可变图自动编码器 链接预测

animutomo/gcmc Graph Convolution Matrix Completion 解决推荐系统中 矩阵补全 matrix completion 问题，并引入 side information（节点的额外信息）提升预测效果。

Ruiqi-Hu/ARGA 对抗正则化图自动编码器 Adversarially Regularized Graph Autoencoder，可用于图卷积的链路预测。进化路线 GAE -> VGAE -> ARGA 

TAMU-VITA/L2-GCN GCN高效分层训练框架

safe-graph/DGFraud 基于深度图的工具箱，用于欺诈检测

safe-graph/graph-fraud-detection-papers 基于图的欺诈检测论文和资源

aister2020/KDDCUP_2020_AutoGraph_1st_Place KDD KDD CUP 2020自动图形表示学习：第一名解决方案。实现了四种不同的模型GCN、GAT、GraphSage、TAGConv

snap-stanford/distance-encoding 距离编码-为结构表示学习设计更强大的GNN，提出了一类与结构相关的特征，称为距离编码(Distance Encoding，DE)，以帮助 GNN 以比 1-WL test 更严格的表达能力来表示任意大小的节点集。

megvii-research/DPGN DPGN: Distribution Propagation Graph Network for Few-shot Learning 分布传播图网络的小样本学习

THUDM/GRAND Graph Random Neural Network (GRAND)，一种用于图半监督学习的新型图神经网络框架。在模型架构上，GRAND 提出了一种简单有效的图数据增强方法 Random Propagation，用来增强模型鲁棒性及减轻过平滑。基于 Random Propagation，GRAND 在优化过程中使用一致性正则（Consistency Regularization）来增强模型的泛化性，即除了优化标签节点的 cross-entropy loss 之外，还会优化模型在无标签节点的多次数据增强的预测一致性。节点预测 state of the Art.

CUAI/CorrectAndSmooth 标签信息 + 简单模型 直接使用标签进行预测。与 其他方案相比，本文中的 C&S 模型需要的参数量往往要少得多。在很多标准直推式节点分类（transductive node classification）基准上，超过或媲美当前最优 GNN 的性能。

YimiAChack/GraphSTONE Graph Structural-topic Neural Network 图结构主题神经网络 本文类比自然语言处理中的相关概念，借助主题模型学习图的结构信息。

YuGuangWang/PAN 借鉴了物理中的一些概念，设计了一种 path integral based graph neural networks (PAN)。 PAN 将图拉普拉斯泛化到一种新的转移矩阵 maximal entropy transition (MET) matrix。重要的是，MET 矩阵的对角线元素直接和子图中心性相关，因此提供了一种自然的自适应池化机制。

lukecavabarrett/pna 作者提出了 Principal Neighbourhood Aggregation (PNA)，一种考虑了 degree 的全新的 GNN 聚合器（泛化了现有的求和聚合器）。作者通过一些图例形象的解释了现有的各种聚合器的表示能力及其缺陷。

benedekrozemberczki/SimGNN A Neural Network Approach to Fast Graph Similarity Computation  图相似度计算

snap-stanford/GraphGym Identity-aware Graph Neural Networks一种身份感知图神经网络对现有的消息传递 GNN 进行了扩展，将其性能提升到了高于 1-WL 测试的水平。实验结果表明，将现有的 GNN 转变为 ID-GNN 可以在难以分类的节点预测、边预测、图属性预测任务中获得平均 40% 的准确率提升；在节点和图分类对比基准任务中可以获得 3%的准确率提升；链接预测任务重可以获得 15% 的 ROC UC 提升。

YuweiCao-UIC/KPGNN 图神经网络增量学习在事件检测中的应用 

divelab/DeeperGNN 解耦Transformation和Propagation的深度图神经网络 1、Transformation操作：MLP操作，torch.nn.Linear线性映射操作；2、Propagation操作：图中的邻居节点往中心节点汇聚的操作，最简单的实现方式是AH，A是图的邻接矩阵，H是图的特征矩阵。

BUPT-GAMMA/CPF 提出了一个有效的知识蒸馏框架，以将任意预训练的GNN教师模型的知识注入精心设计的学生模型中。学生模型是通过两个简单的预测机制构建的，即标签传播和特征转换，它们自然分别保留了基于结构和基于特征的先验知识。

WangXuhongCN/APAN Asynchronous Propagation Attention Network for Real-time Temporal Graph Embedding 实时时间图嵌入的异步传播注意网络

flyingdoog/PGExplainer GNN 的参数化解释器 PGExplainer。PGExplainer 利用深度神经网络对解释的生成过程进行参数化处理，能够实现同时对多个实例进行解释。

lsj2408/GraphNorm 图归一化:一种加速图神经网络训练的原则性方法,通过一个可学习的移位来归一化每个单独图的所有节点上的特征值。收敛速度要快得多。它还改进了GNN的泛化，在图分类上实现更好的性能。

YuGuangWang/UFG 基于小波变换（framelet transforms）的图神经网络。

LirongWu/awesome-graph-self-supervised-learning 图自监督学习（Graph Self-supervised Learning）最新综述+Github代码汇总

karenlatong/AGC-master Attributed Graph Clustering via Adaptive Graph Convolution 通过自适应图卷积的属性图聚类

maxiaoba/GRAPE 基于GNN的缺失特征填充和标签预测模型。将特征填充问题转为边级别的预测任务，将标签预测问题转为节点级别的预测任务。

PKU-DAIR/DGMLP 评估深度图神经网络，对图形结构数据使用深度汇总的实验评估。GNN模型普遍较浅的真正原因 - 模型退化与过平滑。

snap-stanford/CAW 基于因果匿名游走的时序网络归纳表示学习模型

BUPT-GAMMA/Graph-Structure-Estimation-Neural-Networks 用于估计适配于GNN的图结构，以提高下游任务性能。具体地，GEN引入结构模型考虑图生成过程中的潜在社团结构，并提出观察模型将多方面信息（例如，多阶邻域相似性）作为图结构的观测。基于这些模型，GEN利用贝叶斯推断框架得到最终估计图。大量实验结果验证了GEN的有效性及其估计图的合理性。

huawei-noah/trustworthyAI 基于图自编码器的因果结构学习模型

wanyu-lin/ICML2021-Gem 针对图神经网络的通用因果解释方法

thunlp/GNNPapers 图神经网络上的必读论文 （GNN）

# NLP自然语言处理

## Transformer库与优化
  * huggingface/transformers Transformers: State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0. 
  * pytorch/fairseq Python编写的Facebook AI Research Sequence-to-Sequence工具包。
  * dbiir/UER-py 一个用于对通用语料进行预训练并对下游任务进行微调的工具包。提供了非常丰富的模型库。
  * ml-jku/hopfield-layers NLP 领域里大热的 Transformer，其网络更新规则其实是和 Hopfield 网络在连续状态下是相同的。Transformer 中的这种注意力机制其实等价于扩展到连续状态的 modern Hopfield 网络中的更新规则。论文作者来自奥地利林茨大学、挪威奥斯陆大学等机构，与 Jürgen Schmidhuber 合著 LSTM 的 Sepp Hochreiter 也是作者之一。
  * laiguokun/Funnel-Transformer Transformer优化，一种新的自我注意模型，可以将隐藏状态的序列逐渐压缩为较短的状态，从而降低了计算成本。
  * mit-han-lab/hardware-aware-transformers 用于高效自然语言处理的硬件感知型Transformers.实现高达3倍的加速和3.7倍的较小模型尺寸，而不会降低性能。
  * mit-han-lab/lite-transformer 具有长距离短距离注意的Lite transformer
  * microsoft/DeBERTa：注意力分散的增强解码的BERT，使用两种新颖的技术改进了BERT和RoBERTa模型，显着提高了预训练的效率和下游任务的性能。
  * allenai/longformer 用于长文档的类似BERT的模型
  * Tencent/TurboTransformers a fast and user-friendly runtime for transformer inference on CPU and GPU
  * idiap/fast-transformers Pytorch library for fast transformer implementations
  * bytedance/lightseq 高效的序列处理与生成库，提供 Bert, GPT, Transformer，beam search, diverse beam search, topp/topk sampling
  * Big Bird 稀疏注意力机 随机注意力机制+局部注意力机制+全局注意力机制 PurdueCAM2Project/TensorFlowModelGardeners/official/nlp/projects/bigbird/
  * lucidrains/performer-pytorch 使用一个高效的线性广义注意力框架（generalized attention framework），允许基于不同相似性度量（核）的一类广泛的注意力机制。该框架通过谷歌的新算法 FAVOR+（ Fast Attention Via Positive Orthogonal Random Features）来实现，后者能够提供注意力机制的可扩展低方差、无偏估计，这可以通过随机特征图分解（常规 softmax-attention）来表达。该方法在保持线性空间和时间复杂度的同时准确率也很有保证，也可以应用到独立的 softmax 运算。此外，该方法还可以和可逆层等其他技术进行互操作。google-research/google-research/tree/master/performer
  * microsoft/fastformers 实现Transformers在CPU上223倍的推理加速 它能对基于Transformer的模型在各种NLU任务上实现高效的推理时间性能。论文FastFormers的作者表明，利用知识蒸馏、结构化剪枝和数值优化可以大幅提高推理效率。我们表明，这种改进可以达到200倍的加速，并在22倍的能耗下节省超过200倍的推理成本。
  * mit-han-lab/lite-transformer 轻量级Transformer，注意力长短搭配 长依赖和短依赖的剥离，并引入卷积来捕捉短依赖，总体思想和Transformer之自适应宽度注意力有点类似。这篇文章中发现低层次上的注意力都比较短，层次越高，注意力的所关注的依赖越长。
  * ThilinaRajapakse/simpletransformers Transformers for Classification, NER, QA, Language Modelling, Language Generation, T5, Multi-Modal, and Conversational AI 
  * mit-han-lab/lite-transformer  Lite Transformer with Long-Short Range Attention 
  * cloneofsimo/RealFormer-pytorch 通过在Transformer架构上进行改造来提升BERT训练效果，具体为：使用attention残差机制改造Transformer。1、realformer在标签数量较少的分类任务上有一定的提升效果，提升的幅度与数据集和任务难度有关，一般越难的任务提升的幅度越大。2、realformer在标签数量达到一定的数值时，其效果便会大打折扣，在某些数据集上甚至会无法学习。
  * openai/sparse_attention 稀疏Attention
  * sacmehta/delight 提出了一个更深更轻的Transformer，DeLighT，它的性能与Transformer相似，甚至更好，平均少了2到3倍的参数。
  * BSlience/transformer-all-in-one 记录了学习Transformer过程中的一些疑问和解答，并且实现Transformer的全过程。
  * mlpen/Nystromformer 利用了 Nyström 方法来近似标准的Attention。
  * xuanqing94/FLOATER 基于连续动态系统学习更加灵活的位置编码

## 文本分类
  * tcxdgit/cnn_multilabel_classification 基于TextCNN和Attention的多标签分类
  * ilivans/tf-rnn-attention Tensorflow实现文本分类任务的关注机制。
  * skdjfla/toutiao-text-classfication-dataset 中文文本分类数据集 共382688条，分布于15类中。
  * xiaoqian19940510/text-classification-surveys 文本分类资源汇总，包括深度学习文本分类模型，如SpanBERT、ALBERT、RoBerta、Xlnet、MT-DNN、BERT、TextGCN、MGAN、TextCapsule、SGNN、SGM、LEAM、ULMFiT、DGCNN、ELMo、RAM、DeepMoji、IAN、DPCNN、TopicRNN、LSTMN 、Multi-Task、HAN、CharCNN、Tree-LSTM、DAN、TextRCNN、Paragraph-Vec、TextCNN、DCNN、RNTN、MV-RNN、RAE等，浅层学习模型，如LightGBM 、SVM、XGboost、Random Forest、C4.5、CART、KNN、NB、HMM等。介绍文本分类数据集，如MR、SST、MPQA、IMDB、Ye…
  * 649453932/Chinese-Text-Classification-Pytorch 中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention，DPCNN，Transformer，基于pytorch，开箱即用。 
  * 649453932/Bert-Chinese-Text-Classification-Pytorch  使用Bert，ERNIE，进行中文文本分类
  * SanghunYun/UDA_pytorch Unsupervised Data Augmentation  with BERT 一种半监督学习方法，可在多种语言和视觉任务上实现SOTA结果。仅用20个标记的示例，UDA的性能就优于之前在25,000个标记的示例上训练的IMDb上的SOTA。
  * GT-SALT/MixText 文本半监督方法MixText 提出一种全新文本增强方式——TMix，在隐式空间插值，生成全新样本。对未标注样本进行低熵预测，并与标注样本混合进行TMix。MixText可以挖掘句子之间的隐式关系，并在学习标注样本的同时利用无标注样本的信息。超越预训练模型和其他半监督方法
  * beyondguo/label_confusion_learning 利用标签之间的混淆关系，提升文本分类效果。利用标签信息时能够充分考虑标签之间的重叠或者依赖关系。
  * AIRobotZhang/STCKA 基于知识图谱的文本分类 将每个短文本与其在KB中的相关概念相关联，之后，将概念信息作为先验知识整合到深度神经网络中。
  * ShannonAI/Neural-Semi-Supervised-Learning-for-Text-Classification 在大规模通用领域预训练的前提下，更好地利用大规模领域内无标注语料与标注语料，从而最大限度地提升模型效果.足量的领域内语料U使模型不需要再在通用领域语料上预训练；无论是采用预训练的方式还是自训练的方式，都可以显著提升模型效果，二者结合可以得到最佳结果；当领域内标注数据D较小的时候，在伪平行数据D'上训练、再在D上微调的方法可以提升更多的效果；当D更大的时候，在D和D'上联合训练取得的效果更好。
  * xmu-xiaoma666/External-Attention-pytorch 17 篇注意力机制 PyTorch 实现
  * DunZhang/LM-MLC 基于完型填空(模板)的多标签分类算法.

## 文本摘要 文本生成
  * abisee/pointer-generator 使用指针生成器网络进行汇总
  * AIKevin/Pointer_Generator_Summarizer 指针生成器网络：具有关注，指向和覆盖机制的Seq2Seq，用于抽象性摘要。 tensorflow 2.0
  * kjc6723/seq2seq_Pointer_Generator_Summarizer 中文会话中生成摘要总结的项目  tensorflow 2.0
  * steph1793/Pointer_Transformer_Generator tensorflow 2.0
  * magic282/NeuSum 通过共同学习评分和选择句子进行神经文本摘要
  * dmmiller612/bert-extractive-summarizer BERT易于使用的提取文本摘要
  * nju-websoft/NEST 输入知识图谱的基于联合编码的弱监督神经实体摘要方法
  * xcfcode/Summarization-Papers 文本摘要论文总结
  * liucongg/GPT2-NewsTitle GPT2.带有超级详细注释的中文GPT2新闻标题生成项目。
  * bojone/SPACES 端到端的长本文摘要模型（法研杯2020司法摘要赛道） 
  * RUCAIBox/TextBox 基于Python和PyTorch开发的，用于在一个统一的、全面的、高效的框架中复现和开发文本生成算法，主要面向研究者使用。我们的库包括16种文本生成算法，涵盖了两个主要任务：无条件（无输入）生成、序列到序列（Seq2Seq）生成，包括机器翻译和摘要生成。模型 无条件：LSTMVAE	(Bowman et al., 2016)、CNNVAE (Yang et al., 2017)、HybridVAE	(Semeniuta et al., 2017)、SeqGAN	(Yu et al., 2017)、TextGAN (Zhang et al., 2017)、RankGAN	(Lin et al., 2017)、MaliGAN (Che et al., 2017)、LeakGAN (Guo et al., 2018)、MaskGAN	(Fedus et al., 2018)。序列到序列 RNN (Sutskever et al., 2014)、Transformer	(Vaswani et al., 2017b)、GPT-2 (Radford et al.)、XLNet (Yang et al., 2019)、BERT2BERT (Rothe et al., 2020)、BART（Lewis et al。，2020）
  * google-research/text-to-text-transfer-transformer Text-To-Text Transfer Transformer T5的理念就是“万事皆可 Seq2Seq”，它使用了标准的 Encoder-Decoder 模型，并且构建了无监督/有监督的文本生成预训练任务，最终将效果推向了一个新高度。
  * google-research/multilingual-t5 T5 的多国语言版
  * bojone/t5_in_bert4keras 在keras中使用T5模型 , 用 mT5 small 版本 finetune 出来的 CSL 标题生成模型，BLEU 指标能持平基于 WoBERT 的 UniLM 模型，并且解码速度快 130%；而用 mT5  base 版本 finetune 出来的 CSL 标题生成模型，指标能超过基于 WoBERT 的 UniLM 模型 1% 以上，并且解码速度也能快 60%。
  * ZhuiyiTechnology/t5-pegasus 中文生成式预训练模型，以mT5为基础架构和初始权重，通过类似PEGASUS的方式进行预训练。
  * yym6472/ms_pointer_network 用多来源Pointer Network的产品标题摘要方法.从两个信息来源：原始商品标题和知识信息knowledge中抽取信息，然后将二者进行综合得到最后的结果。
  * FeiSun/ProductTitleSummarizationCorpus Dataset for CIKM 2018 paper "Multi-Source Pointer Network for Product Title Summarization" 
  * xcfcode/Summarization-Papers This repo contains a list of summarization papers including various topics. If any error, please open an issue.摘要论文列表，包括各种主题。
  * MaartenGr/keyBERT 一种最小且易于使用的关键字提取技术，它利用BERT嵌入来创建与文档最相似的关键字和关键字短语。
  * Morizeyao/GPT2-Chinese GPT2中文文生模型，包括散文、诗词、对联、通用中文、中文歌词、文言文
  * jiacheng-ye/kg_one2set 解决关键词生成任务，给一篇源文档（比如论文的摘要），关键词预测任务就是希望能预测出一些表达文档重点信息的关键词，或者更准确的说是关键短语。提出了模型SetTrans，其特点是能够预测更多、更准确而且重复率更低的关键词集合。并行预测，在 inference 效率上是Transfomer的6.44倍。
  * LLluoling/PENS-Personalized-News-Headline-Generation 个性化新闻头条生成的数据集和通用框架
  * YunwenTechnology/QueryGeneration 智能扩充机器人的“标准问”库之Query生成
  * imcaspar/gpt2-ml GPT2 多语言支持, 15亿参数中文预训练模型 

## 文本匹配 文本相似度
  * UKPLab/sentence-transformers 句子转换器：使用BERT / RoBERTa / XLM-RoBERTa＆Co.和PyTorch的多语言句子嵌入
  * thunlp/OpenMatch 总体架构包括两大部分：一是相关文档检索，即根据用户检索词，从大规模文档集合中返回最相关的Top-K(K通常为100或1000)文档。二是文档重排序，即将各神经网络模型和非神经网络模型的排序特征整合，对Top-K文档重排序，进一步提升排序效果。OpenMatch提供了融合外部知识图谱信息的知识增强模型，和筛选大规模数据的数据增强模型。
  * terrifyzhao/text_matching 常用文本匹配模型tf版本，数据集为QA_corpus 模型:DSSM\ConvNet\ESIM\ABCNN\BiMPM\DIIN\DRCN
  * Brokenwind/BertSimilarity 基于Google的BERT模型来进行语义相似度计算。
  * bohanli/BERT-flow 基于流式生成模型，将BERT的表示可逆地映射到一个均匀的空间，文本表示、语义文本相似性任务的SOTA。
  * DataTerminatorX/Keyword-BERT  带关键词的BERT语义匹配
  * bojone/BERT-whitening 简单的向量白化改善句向量质量，可以媲美甚至超过BERT-flow的效果。
  * autoliuweijie/BERT-whitening-pytorch Pytorch version of BERT-whitening
  * princeton-nlp/SimCSE SimCSE：句子嵌入的简单对比学习 。提供无监督或有监督的对比学习。是目前文本相似度更好的方法。
  * nilboy/gaic_track3_pair_sim  短文本语义匹配，2021年全球人工智能技术创新大赛-赛道三-冠军方案
  * yym6472/ConSERT 基于对比学习的句子语义表示迁移框架。包含三部分，数据增强，BERT 编码层，对比损失层。
  * amazon-research/sccl 利用对比学习促进更好地基于距离的短文本聚类实现。
 * ZhuiyiTechnology/roformer-sim 融合检索和生成的RoFormer-Sim模型.可应用于相似句生成、相似句扩增、语义相似度问题。

## BERT优化
  * google-research/bert Bidirectional Encoder Representations from Transformers 来自Transformers的双向编码器表示法
  * google-research/ALBERT 用于语言表达自我监督学习的Lite BERT
  * bojone/bert-of-theseus BERT 模型压缩方法 ,theseus(忒修斯之船 如果忒修斯的船上的木头被  逐渐替换，直到所有的木头都不是原来的木头，那这艘船还是原来的那艘船吗？),将原始大模型切分为多个大模块，固定大模型权重，训练时随机替换为小模块,充分训练后，将小模型继续微调。
  * brightmart/albert_zh 使用TensorFlow 进行自我监督学习语言表示的Lite Bert的实现预训练的汉语模型
  * bert4keras 更清晰、更轻量级的keras版bert
  * huawei-noah/Pretrained-Language-Model 华为诺亚方舟实验室开发的预训练语言模型及其相关优化技术NEZHA是一种经过预训练的中文语言模型，可以在多项中文NLP任务上实现最先进的性能TinyBERT是一种压缩的BERT模型，推理时可缩小7.5倍，加快9.4倍
  * Lisennlp/TinyBert 基于华为的TinyBert进行修改的，简化了数据读取的过程，方便我们利用自己的数据进行读取操作。
  * epfml/collaborative-attention 整合多头注意力,任何经过预训练的注意力层重新配置为协作注意力层。
  * thunlp/ERNIE 用知识图谱增强 BERT 的预训练效果 
    * 1) 对于抽取并编码的知识信息，研究者首先识别文本中的命名实体，然后将这些提到的实体与知识图谱中的实体进行匹配。研究者并不直接使用 KG 中基于图的事实，相反他们通过知识嵌入算法（例如 TransE）编码 KG 的图结构，并将多信息实体嵌入作为 ERNIE 的输入。基于文本和知识图谱的对齐，ERNIE 将知识模块的实体表征整合到语义模块的隐藏层中。
    * 2) 与 BERT 类似，研究者采用了带 Mask 的语言模型，以及预测下一句文本作为预训练目标。除此之外，为了更好地融合文本和知识特征，研究者设计了一种新型预训练目标，即随机 Mask 掉一些对齐了输入文本的命名实体，并要求模型从知识图谱中选择合适的实体以完成对齐。
 * ZhuiyiTechnology/WoBERT 以词为基本单位的中文BERT（Word-based BERT）
 * autoliuweijie/FastBERT FastBERT：具有自适应推断时间的自蒸馏BERT pip install fastbert
 * alexa/bort 论文 Optimal Subarchitecture Extraction for BERT. “ BERT的最佳子体系结构提取”的代码。Bort是用于BERT架构的最佳子集，它是通过对神经架构搜索应用完全多项式时间近似方案（FPTAS）提取的。 Bort的有效（即不计算嵌入层）大小是原始BERT大型体系结构的5.5％，是净大小的16％。它在CPU上也比基于BERT的速度快7.9倍，并且比体系结构的其他压缩变体和某些非压缩变体性能更好。与多个公共自然语言理解（NLU）基准上的BERT-large相比，它的平均性能提高了0.3％至31％。
 * ymcui/MacBERT MacBERT是经过改进的BERT，具有新颖的MLM作为校正预训练任务，从而减轻了预训练和微调的差异。
 * valuesimplex/FinBERT 基于 BERT 架构的金融领域预训练语言模型
 * yitu-opensource/ConvBert ConvBERT，通过全新的注意力模块，仅用 1/10 的训练时间和 1/6 的参数就获得了跟 BERT 模型一样的精度。依图研发团队从模型结构本身的冗余出发，提出了一种基于跨度的动态卷积操作，并基于此提出了 ConvBERT 模型。
 * wtma/CharBERT 字符敏感的预训练语言模型 通过结合字符级别和词级别的信息实现了更为全面的输入编码，同时，结合 RNN 和 CNN 的优势，基本上 CNN，RNN，Transformer 都使用上了，体现了新老研究成果的结合在一定程度上能进一步提升方法的性能。
 * Sleepychord/CogLTX 将BERT应用于长文本 CogLTX 遵循一种特别简单直观的范式，即 抽取关键的句子 => 通过 BERT 得到答案 这样的两步流程。
 * ShannonAI/service-streamer 服务流媒体BERT服务,每秒处理1400个句子的BERT服务.
 * DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding 双重 BERT 模型的解耦上下文编码框架 shawroad/NLP_pytorch_project/Text_Ranking/DC_Bert_Ranking/
 * Sleepychord/CogLTX 可将当前类似BERT的预训练语言模型应用于长文本。使用动态规划算法将长文本划分为文本块集合；使用MemRecall对原长句中的子句进行打分：从而选择出分数最高的子句组成  再进行训练，这样一来的话，COGLTX相当于使用了了两个bert，MemRecall中bert就是负责打分，另一个bert执行原本的NLP任务。
 * bojone/BERT-whitening  简单的线性变换（白化）操作，就可以达到BERT-flow的效果。自定义全局池化
 * alibaba/AliceMind/tree/main/LatticeBERT Leveraging Multi-Granularity Representations in Chinese Pre-trained Language Models  利用多粒度的词格信息（word lattice），相对字级别的模型取得了性能提升。
 * ShannonAI/ChineseBert 融合字形与拼音信息的中文Bert预训练模型

* ## 机器阅读理解
  * basketballandlearn/MRC_Competition_Dureader 基于大规模MRC数据再训练的机器阅读理解预训练模型（包括roberta-wwm-large、macbert-large），可以使用[transformers库](https://huggingface.co/luhua/chinese_pretrain_mrc_roberta_wwm_ext_large)。
  * wptoux/albert-chinese-large-webqa 基于百度webqa与dureader数据集训练的Albert Large QA模型
  * bojone/dgcnn_for_reading_comprehension 基于膨胀门卷积的阅读理解式问答模型（Keras实现）
  * cooelf/AwesomeMRC 对MRC的研究摘要和参考资料
  * nlpdata/c3 中文机器阅读理解数据集 multiple-Choice Chinese machine reading Comprehension dataset.
  * qiufengyuyi/event_extraction 百度aistudio事件抽取比赛 使用机器阅读理解来尝试解决。
  * liuhuanyong/MiningZhiDaoQACorpus 百度知道问答语料库，包括超过580万的问题，938万的答案，5800个分类标签。基于该问答语料库，可支持多种应用，如闲聊问答，逻辑挖掘。 
  * xv44586/ccf_2020_qa_match CCF2020问答匹配比赛 任务是：给定IM交流片段，片段包含一个客户问题以及随后的经纪人若干IM消息，从随后的经纪人消息中找出一个是对客户问题的回答。
  * lgw863/LogiQA-dataset 数据集包含8,678个QA实例
  * HIT-SCIR/Molweni 提出了构建于多人对话的英文机器阅读理解（MRC）数据集——Molweni，并覆盖了对话语篇结构。Molweni源自于Ubuntu聊天语料库，包括10,000个对话，共计88,303条话语（utterance）。我们共标注了30,066个问题，包括可回答和不可回答的问题。Molweni独特地为其多人对话提供了语篇结构信息，共标注了78,245个语篇关系实例，为多人对话语篇结构分析（Discourse  parsing）贡献了大规模数据。

* ## 知识图谱问答KBQA、多跳推理
  * RUCAIBox/KBQAPapers 知识图谱问答KBQA论文集
  * shijx12/TransferNet An Effective and Transparent Framework for Multi-hop Question Answering over Relation Graph 多跳问题解答关系图的有效透明框架，通过每一跳都预测当前关系得分，并更新实体得分，直到最大跳数。预测该问题的跳数，按跳数的概率加权每一跳得分作为实体的最终得分。
  * malllabiisc/EmbedKGQA 基于知识图谱嵌入的链路预测处理多跳问答。首先训练实体嵌入，随后利用实体嵌入学习问题嵌入，预测时对所有实体，构建(head entity, question)并评分，并选择评分最高的头实体作为答案。能很好地处理知识图谱中的不完整和稀疏的问题
  * cdjhz/multigen Language Generation with Multi-hop Reasoning on Commonsense Knowledge Graph 基于常识知识图的多跳推理语言生成 本研究关注一类条件文本生成任务，即给定输入源文本X，目标是生成一段目标文本 Y。研究员们额外增加了一个知识图谱 G=(V,E) 的输入为模型在生成时提供常识知识的信息。
  * INK-USC/MHGRN 基于知识库的多跳关系推理 本篇文章提出了multi-hop relational reasoning module（多跳关系推理模型）叫做MHGRN多跳推理网络。该模型在额外的多跳知识图谱中抽取的子网络中进行推理。本文提出的方法将已有的基于路径的常识推理以及GCN融合在了一起，并在CommonsenseQA和OpenbookQA上取得了良好的效果。
  * lanyunshi/Multi-hopComplexKBQA 查询图生成，用于回答知识库中的多跳复杂问题.提出了一种改进的分阶段查询图生成方法，该方法具有更灵活的生成查询图的方式。在查询图生成的每一步，包含三种预定义的操作：扩展、连接、聚合。
  * nju-websoft/SPARQA 基于知识库的问题解答,提出了一种新颖的骨架语法来表示一个复杂问题的高级结构。骨架语法本质上是依赖语法的一个选定子集，用于专门表示复杂问题的高级结构。这种专用的粗粒度表示形式由于其简单性而可能具有准确的解析算法，有助于提高下游细粒度语义解析的准确性。
  * mori97/JKNet-dgl 跳跃知识网络的dgl实现
  * THUDM/CogQA 基于认知图谱实现多跳阅读.从人类的认知过程中受到启发。双过程理论认为，我们的大脑思考过程由两套系统构成: System1 和 System 2。System 1: 我们的大脑首先通过System 1隐式的、无意识的和凭借直觉的过程来检索相关信息。System 2: 在System 1过程的基础上，再进行一个显式的、有意识的、可控的推理过程，即System 2。作者使用BERT模型构建System 1，使用GNN模型构建System 2。
  * michiyasunaga/qagnn GNN 在融合 QA 上下文与 KG 的一个尝试，在问答任务上相比现有的预训练语言模型、以及预训练 +KG 模型，都有不小的提升。同时，使用 attention-base GNN，能够可视化知识图谱中节点之间的注意力关系，有助于提高 QA 可解释性和结构化推理的能力。
  * BDBC-KG-NLP/QA-Survey 北航大数据高精尖中心研究张日崇团队对问答系统的总结。包括基于知识图谱的问答（KBQA），基于文本的问答系统（TextQA），基于表格的问答系统（TabletQA）和基于视觉的问答系统（VisualQA），每类系统分别对学术界和工业界进行总结。
  * WenRichard/KBQA-BERT 基于知识图谱的问答系统，BERT做命名实体识别和句子相似度，分为online和outline模式
  * RichardHGL/WSDM2021_NSM KBQA 的神经状态机器 ComplexWebQuestions # 2 
  * UKPLab/coling2018-graph-neural-networks-question-answering 用门图形神经网络建模语义，用于知识库问题解答

* ## 知识图谱
  * autoliuweijie/K-BERT Enabling Language Representation with Knowledge Graph ，已被AAAI2020所录取，是较早的考虑将知识图谱中的边关系引入预训练模型的论文。论文链接：arxiv.org/pdf/1909.07606v1.pdf 主要通过修改Transformer中的attention机制，通过特殊的mask方法将知识图谱中的相关边考虑到编码过程中，进而增强预训练模型的效果。
  * THU-KEG/KEPLER 主要通过添加类似于TransE的预训练机制来增强对应文本的表示，进而增强预训练模型在一些知识图谱有关任务的效果。
  * txsun1997/CoLAKE 使用知识图谱以增强预训练模型的效果 首先将上下文看作全连接图，并根据句子中的实体在KG上抽取子图，通过两个图中共现的实体将全连接图和KG子图融合起来；最终本文将文本上下文和知识上下文一起用MLM进行预训练，将mask的范围推广到word、entity和relation；为训练该模型，本文采用cpu-gpu混合训练策略结合负采样机制减少训练时间；最终本文提出的方法在知识图谱补全和若干NLP任务上均带来了增益。然后本文将该图转化为序列，使用Transformer进行预训练，并在训练时采用特殊的type embedding来表示实体、词语与其他子图信息
  * JanKalo/KnowlyBERT 提出了一种混合的语言知识模型查询系统，该系统使用语言模型来应对现实世界中知识图谱的不完整性问题。作为KnowlyBERT的输入，用户可以向系统提出以实体为中心的SPARQL查询。首先，查询语言模型（a）；然后，对不完整的知识图谱进行查询，并获得结果（b）；另外SPARQL查询被翻译成多种自然语言语句，这些语言语句在“关系模板生成”步骤中由语言模型完成；语言模型返回多个单词列表以及每个单词（c）的置信度值；然后将这些列表合并为一个列表（d），并根据知识图谱类型信息（e）使用我们的语义过滤步骤进行过滤。此外，执行阈值处理，削减不相关的结果（f）；最后，将语言模型和知识图谱的结果合并（g）并返回给用户。
  * yeliu918/KG-BART 知识图谱增强的预训练模型的生成式常识推理.KG-BART可以利用图上的注意力来聚集丰富的概念语义，从而增强对看不见的概念集的模型泛化。
  * bernhard2202/intkb 一种交互式知识图谱补全框架
  * husthuke/awesome-knowledge-graph 整理知识图谱相关学习资料
  * wangbo9719/StAR_KGC Structure-Augmented Text Representation Learning for Efficient Knowledge Graph Completion 结构增强文本表示学习，实现高效知识图完成.知识图谱补全 
  * Everglow123/MAKG  移动app知识图谱 
  * (openconcept)[http://openkg.cn/dataset/openconcept] 基于自动化知识抽取算法的大规模中文概念图谱。440万概念核心实体，以及5万概念和1200万实体-概念三元组。数据包括了常见的人物、地点等通用实体。
  * openkg-org/OpenEA 基于知识图谱嵌入的开源实体融合工具。本体匹配、实体对齐、真值验证、冲突消解。

* ## NLP语料和数据集
  * thu-coai/CrossWOZ 大规模的中文跨域任务导向对话数据集.它包含5个领域的6K对话会话和102K语音，包括酒店，餐厅，景点，地铁和出租车。
  * goto456/stopwords 中文常用停用词表
  * chatopera/Synonyms 用于自然语言处理和理解的中文同义词。
  * RUCAIBox/TG-ReDial 一个电影领域的对话推荐数据集TG-ReDial (Recommendation through Topic-Guided Dialog)。它包含1万个完整的对话和近13万条语句，加入了话题线索以实现将用户引导至推荐场景这一语义的自然转移，并且采用半自动的方式构建，保留了用户真实的个性化信息（如交互历史，偏好主题），使得人工标注过程更加合理可控。
  * fighting41love/funNLP NLP民工的乐园: 几乎最全的中文NLP资源库 中英文敏感词、语言检测、中外手机/电话归属地/运营商查询、名字推断性别、手机号抽取、身份证抽取、邮箱抽取、中日文人名库、中文缩写库、拆字词典、词汇情感值、停用词、反动词表、暴恐词表、繁简体转换、英文模拟中文发音、汪峰歌词生成器、职业名称词库、同义词库、反义词库、否定词库、汽车品牌词库、汽车零件词库、连续英文切割、各种中文词向量、公司名字大全、古诗词库、IT词库、财经词库、成语词库、地名词库、历史名人词库、诗词词库、医学词库、饮食词库、法律词库、汽车词库、动物词库、中文聊天语料、中文谣言数据、百度中文问答数据集、句子相似度匹配算法集合、bert资源、文本生成&摘要相关工具、cocoNLP信息抽取工具、国内电话号码正则匹配、清华大学XLORE:中英文跨语言百科知识图谱
  * brightmart/nlp_chinese_corpus 大规模中文自然语言处理语料 维基百科json版(wiki2019zh) 新闻语料json版(news2016zh) 百科类问答json版(baike2018qa) 社区问答json版(webtext2019zh) ：大规模高质量数据集 翻译语料(translation2019zh)
  * msra-nlc/ChineseKBQA NLPCC-ICCPOL 2016 Shared Task: Open Domain Chinese Question Answering [开放域中文问答数据集](http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html)
  * jkszw2014/bert-kbqa-NLPCC2017 A trial of kbqa based on bert for NLPCC2016/2017 Task 5 (基于BERT的中文知识库问答实践） 
  * wavewangyue/NLPCC-MH  中文多跳问答数据集 基于 NLPCC 所包含的单跳问题，通过扩充问句内容的方式，构建了一个专注多跳问题的中文 KBQA 数据集
  * (BERT-CCPoem)[https://thunlp.oss-cn-qingdao.aliyuncs.com/BERT_CCPoem_v1.zip] 是完全基于一个囊括了几乎所有中国古典诗词的语料库CCPC-Full v1.0训练而成的，该语料库共计926,024首诗词及8,933,162个诗词句子。THUNLP-AIPoet/BERT-CCPoem 中国古典诗词预训练模型 
  * liucongg/NLPDataSet 数据集包括：DRCD、cmrc2018、chinese-squad、中医数据集、法研杯2019、莱斯杯机器阅读理解、疫情QA、WebQA、Dureader等9个数据集。
  * thunlp/Few-NERD 一个大规模的人工标注的用于少样本命名实体识别任务的数据集。该数据集包含8种粗粒度和66种细粒度实体类型，每个实体标签均为粗粒度+细粒度的层级结构，共有18万维基百科句子，460万个词，每个词都被注释为上下文（context）或一个实体类型的一部分。

* ## 关系抽取、信息抽取
  * weizhepei/CasRel 一种用于关系三重提取的新颖级联二进制标记关系抽取框架.
  * loujie0822/DeepIE DeepIE： 基于深度学习的信息抽取技术,实体抽取\实体关系联合抽取\属性抽取\实体链接/标准化\事件抽取\摘要抽取
  * 131250208/TPlinker-joint-extraction 联合抽取模型 实体关系联合抽取标注关系抽取方案
  * TanyaZhao/MRC4ERE_plus 提出有效和多样的问题：基于机器阅读理解的联合实体关系提取框架
  * cuhksz-nlp/RE-TaMM 于词依存信息类型映射记忆神经网络的关系抽取
  * xiaoqian19940510/Event-Extraction 近年来事件抽取方法总结，包括中文事件抽取、开放域事件抽取、事件数据生成、跨语言事件抽取、小样本事件抽取、零样本事件抽取等类型，DMCNN、FramNet、DLRNN、DBRNN、GCN、DAG-GRU、JMEE、PLMEE等方法
  * 231sm/Reasoning_In_EE 利用本体表示学习实现低资源的事件抽取
  * zjunlp/openue 开源的通用文本信息抽取工具 三元组抽取 事件抽取 槽填充和意图检测
  * thunlp/OpenNRE 开源的神经网络关系抽取工具包，包括了多款常用的关系抽取模型，CNN、BERT、bag-level PCNN-ATT。
  * thunlp/NREPapers 神经网络关系抽取必读论文列表，覆盖了较为经典的神经网络关系抽取领域的已发表论文、综述等。
  * zjunlp/DocED 跨句事件抽取旨在研究如何同时识别篇章内多个事件。提出多层双向网络Multi-Layer Bidirectional Network融合跨句语义和关联事件信息，从而增强内各事件提及的判别。

* ## 实体识别NER、意图识别、槽位填充
  * LeeSureman/Flat-Lattice-Transformer 中文NER 基于Transformer设计了一种巧妙position encoding来融合Lattice结构，可以无损的引入词汇信息。基于Transformer融合了词汇信息的动态结构，支持并行化计算，可以大幅提升推断速度。
  * MiuLab/SlotGated-SLU 意图识别和槽位填充（slot filling）联合模型，提出一个槽位门控机制（slot-gated mechanism）来解决没有明确地建立槽位和意图之间联系的缺陷，达到较好的效果。
  * monologg/JointBERT 意图识别和槽位填充（slot filling）联合训练模型，使用了BERT来进行语义编码，然后做序列标注任务和多分类任务的联合训练。
  * z814081807/DeepNER 天池中药说明书实体识别挑战冠军方案；中文命名实体识别；NER; BERT-CRF & BERT-SPAN & BERT-MRC；Pytorch 
  * liuwei1206/LEBERT Lexicon Enhanced BERT模型来解决中文序列标注NER任务。相比于 FLAT，Lattice LSTM 等方法，它把词汇信息融入到了 BERT 底层的编码过程中。相比于 Lex-BERT，它无需包含词汇类型信息的词典，只需要普通的词向量即可。
  * kangbrilliant/DCA-Net 用于插槽填充和意图检测的协同互感器。数据集ATIS上，意向Acc 97.7 插槽填充F1 95.9 。
  * yizhen20133868/Awesome-SLU-Survey 口语语言理解（Spoken Language Understanding，SLU）作为任务型对话系统的核心组件，目的是为了获取用户询问语句的框架语义表示（semantics frame）信息，进而将这些信息为对话状态追踪模块（DST）以及自然语言生成模块（NLG）所使用。SLU任务通常包含以下两个任务：意图识别（intent detection）和槽位填充（slot filling）。
  * * wuba/qa_match 58同城推出的一款基于深度学习的轻量级问答匹配工具，它融合领域识别与意图识别，对问答意图进行精确理解。

EleutherAI/gpt-neo 模型并行GPT2和类似GPT3的模型的实现，能够使用mesh-tensorflow库扩展到完整的GPT3尺寸（甚至可能更多！）。

huseinzol05/NLP-Models-Tensorflow 抽象总结 聊天机器人依赖解析器 实体标记 提取摘要 发电机 语言检测 神经机器翻译 光学字符识别 POS标签 问题答案 句子对 语音转文字 拼写校正 小队问题答案 抽干 文字扩充 文字分类 文字相似度 文字转语音 主题生成器 主题建模 无监督提取摘要 矢量化器 老少少的声码器 可视化 注意Attention

CyberZHG/keras-xlnet XLNet的非官方实现。

ymcui/Chinese-XLNet 面向中文的XLNet预训练模型

bojone/attention  Attention机制的实现tensorflow/keras

425776024/nlpcda 中文数据增强工具,：1.随机实体替换 2.近义词 3.近义近音字替换 4.随机字删除 5.NER类 BIO 数据增强 6.随机置换邻近的字  7.百度中英翻译互转实现的增强  8.中文等价字替换

wac81/textda Python3中文文本的数据增强

zhanlaoban/EDA_NLP_for_Chinese 适合中文语料的数据增强EDA的实现

ShomyLiu/Neu-Review-Rec Pytorch的基于评论文本的深度推荐系统模型库。DeepCoNN(WSDM'17)、D-Attn(RecSys'17)、ANR(CIKM'18)、NARRE(WWW'18)、MPCN(KDD'18)、TARMF(WWW'18)、CARL(TOIS'19)、CARP(SIGIR'19)、DAML(KDD'19)

squareRoot3/Target-Guided-Conversation 目标指导的开放域对话,在开放域的聊天中目标引导.

JasonForJoy/MPC-BERT MPC-BERT：一种预训练的多方会话理解语言模型.多方会话（MPC）的各种神经模型在收件人识别、说话人识别和反应预测等方面取得了显著的进展。

qiufengyuyi/sequence_tagging 使用bilstm-crf，bert等方法进行序列标记任务

microsoft/unilm UniLM-NLP及更高版本的统一语言模型预训练
 * layoutlm 多模态文档理解预训练模型LayoutLM 2.0，模型首先将文本、图像、布局三种模态的输入转换成向量表示，然后再交给编码器网络，最终输出的表示向量可以供下游任务使用。下游任务：表单理解、票据理解、复杂布局长文档理解、文档图像分类、视觉问答。

YunwenTechnology/Unilm UniLM模型既可以应用于自然语言理解（NLU）任务，又可以应用于自然语言生成（NLG）任务。论文来自微软研究院。模型虽然强大，但微软并没有开源中文的预训练模型。因此云问本着开源之前，将我们预训练好的中文unilm_base模型进行开源。

airaria/TextBrewer 基于PyTorch的NLP任务知识蒸馏工具包，适用于多种模型结构，支持自由组合各种蒸馏策略，并且在文本分类、阅读理解、序列标注等典型NLP任务上均能获得满意的效果。 

czhang99/SynonymNet 基于多个上下文双向匹配的同义实体发现

PRADO 用于文档分类的投影注意网络 性能媲美BERT，但参数量仅为1/300 tensorflow/models/tree/master/research/sequence_projection

rikdz/GraphWriter 基于图Transformer从知识图谱中生成文本

stanford-futuredata/ColBERT ColBERT: 基于上下文（contextualized）的后期交互的排序模型 Efficient and Effective Passage Search via Contextualized Late Interaction over BERT 兼顾匹配的效率和doc中的上下文信息

ymcui/Chinese-ELECTRA 中文ELECTRA预训练模型 其中ELECTRA-small模型可与BERT-base甚至其他同等规模的模型相媲美，而参数量仅为BERT-base的1/10

salesforce/pytorch-qrnn 准循环神经网络Quasi-Recurrent Neural Network,基于使用实例可以比高度优化的 NVIDIA cuDNN LSTM 实现2到17倍快

ChenghaoMou/pytorch-pQRNN pQRNN 结合一个简单的映射和一个quasi-RNN编码器来进行快速并行处理。pQRNN模型表明这种新的体系结构几乎可以达到BERT级的性能，尽管只使用1/300的参数量和有监督的数据。

alibaba/EasyTransfer 自然语言处理的迁移学习工具。主要特性：预训练语言模型工具，丰富且高质量的预训练模型库 BERT, ALBERT, RoBERTa, T5, etc,丰富且易用的NLP应用 如文本匹配、分本分类、机器阅读理解MRC，自动化的知识蒸馏，易用且高效的分布式训练。

RUCAIBox/TG_CRS_Code TG-ReDial相应的推荐、回复生成、主题预测功能实现。

Qznan/QizNLP 快速运行分类、序列标注、匹配、生成等NLP任务的Tensorflow框架 (中文 NLP 支持分布式）

toizzy/tilt-transfer 运行TILT迁移学习实验的代码 让语言模型先在乐谱上进行训练，再在自然语言上训练可以有效的提升语言模型的性能。

XiaoMi/MiNLP/tree/main/minlp-tokenizer 小米 AI 实验室 NLP 团队开发的小米自然语言处理平台 MiNLP 现已开源了中文分词功能

explosion/spaCy 工业级强度的NLP工具包，被称为最快的工业级自然语言处理工具。支持多种自然语言处理的基本功能，主要功能包括分词、词性标注、词干化、命名实体识别、名词短语提取等。

microsoft/unilm/tree/master/layoutlm 多模态预训练模型 LayoutLM 2.0，不仅考虑了文本和页面布局信息，还将图像信息融合到了多模态框架内。下游任务微调：表单理解 票据理解 复杂布局长文档理解 文档图像分类 视觉问答 

RUCAIBox/CRSLab 用于构建会话推荐系统（Conversational Recommender System CRS）的开源工具包。 对话推荐任务主要拆分成三个任务：推荐任务（生成推荐的商品），对话任务（生成对话的回复）和策略任务（规划对话推荐的策略）。模型 CRS 模型 ReDial、KBRD、KGSF、TG-ReDial、推荐模型 Popularity、GRU4Rec、SASRec、TextCNN、R-GCN、BERT、对话模型	HERD、Transformer、GPT-2 策略模型	PMI、
MGCG、Conv-BERT、Topic-BERT、Profile-BERT

RUCAIBox/CRSPapers 选取了近年来基于深度学习的对话推荐系统相关论文（共 62 篇），并根据工作的类型进行分类，以供参考。

nlp-uoregon/trankit 用于多语言自然语言处理的基于轻型变压器的Python工具包 支持以下任务：句子分割。标记化。多字令牌扩展。词性标记。形态特征标记。依赖性解析。命名实体识别。

yizhen20133868/NLP-Conferences-Code 记录NLP相关顶会(如ACL、EMNLP、NAACL、COLING、AAAI、IJCAI)的论文开源项目合集

cuhksz-nlp/DGSA 基于方向建模图卷积网络的联合方面提取和情感分析.输入:由句子生成的依存句法分析树得到的图;句子（词序列）.输出表示为一个标签序列.可用于序列标注、ER 和情感分析。

FedML-AI/FedNLP FedNLP：自然语言处理中的联合学习研究平台

Graph4nlp是一个易于使用的NLP图形神经网络库。应用：文本分类、神经机器翻译、摘要、KG补全：预测konwledge图中两个现有实体之间的缺失关系。数学问题解决：自动解决数学习题，用易懂的语言提供问题的背景信息。名称实体识别、问题生成。

pytorch/fairseq/tree/master/examples/bart BART Bidirectional and Auto-Regressive Transformers 是以去噪为预训练目标训练的序列间模型， 一种符合生成任务的预训练方法。我们证明了这种预训练目标更为通用，并且证明了我们可以在SQuAD和GLUE上匹配RoBERTa的结果，并在摘要（XSum，CNN数据集）、长形式生成性问答（ELI5）和对话-反应生成（ConvAI2）上获得最新的结果。在生成任务上显著高于BERT, UniLM, XLNet, RoBERTa等模型

huybery/r2sql Dynamic Hybrid Relation Network for Cross-Domain Context-Dependent Semantic Parsing 跨域上下文相关语义分析的动态混合关系网络 应用于：多轮text-to-SQL 任务（通过多轮对话的方式生成最终的查询语句， Text-to-SQL 任务：给定一个自然语言查询和数据库的作为输入，产生一个SQL语句作为输出。）

facebookresearch/GENRE 首创生成式实体检索，通过seq2seq方法(BART)生成有意义的实体名称从而实现实体链接，而且还可以取得SOTA结果。

sebastian-hofstaetter/intra-document-cascade IDCM模型: 文档内部级联选择段落服务于文档排序。采用文档内部级联策略，在运行复杂并且高效果的排序模型（ETM，Effective Teacher Model）之前，使用高效率的模型（ESM，Efficient Student Model）进行候选文档中多余段落的删除。相比bert，具有基本相同的效果，而且查询延迟降低400%以上。

jingtaozhan/DRhard 通过难负例优化稠密向量文档检索模型训练，利用动态难负例抽样提高模型效果，以及将随机抽样结合静态难负例抽样提高模型稳定性。

yechens/NL2SQL Text2SQL 语义解析数据集、解决方案、paper资源整合项。Text to SQL( 以下简称Text2SQL)，是将自然语言文本（Text）转换成结构化查询语言SQL的过程，属于自然语言处理-语义分析（Semantic Parsing）领域中的子任务。

destwang/CTCResources 中文文本纠错（Chinese Text Correction, CTC）相关论文、数据集。

fushengwuyu/chinese_spelling_correction 中文文本纠错模型：bert语言模型+字音字形相似度 、MLM、seq2seq

grammarly/gector ”GECToR – Grammatical Error Correction: Tag, Not Rewrite”，使用给序列打标签来替代主流的Seq2Seq模型。本文采取了一种迭代的方法，也就是通过多次(其实最多也就两三次)序列打标签。

destwang/CTC2021 本赛题主要选择互联网上中文母语写作者撰写的网络文本作为校对评测数据，从拼写错误、语法错误、语病错误等多个方面考察机器的认知智能能力。

Jingjing-NLP/VOLT 借鉴边际效用通过最优转移学习词表。

thunlp/OpenAttack 文本对抗攻击工具包，可以用于文本对抗攻击的全过程，包括文本预处理、受害模型访问、对抗样本生成、对抗攻击评测以及对抗训练等。

thunlp/TAADpapers 文本对抗攻击和防御必读论文列表。

lupantech/InterGPS 基于符号推理的几何数学题求解器。建立了一个新的大规模基准数据集，称为 Geometry3K。这些数据从两本中学教材收集，涵盖了北美 6 到 12 年级的几何知识。每道题收集了 LaTeX 格式的问题文本、几何图形、四个选项和正确答案。为了模型的精细评估，每个数据标注了问题目标和几何图形的类型。Inter-GPS 将几何关系集 R 和定理集 KB 作为输入，应用定理预测器预测适用的定理序列，逐步对关系集进行符号推理，从而输出问题目标的答案。

Helsinki-NLP/Tatoeba-Challenge 这是一个机器翻译的挑战集，包含 29G 翻译单元在 3，708 位ext 覆盖 557 种语言。该包包括从涵盖 134 种语言的 Tatoeba.org 衍生的 631 套测试集的版本。此包提供以多种语言进行机器翻译的数据集，并提供从 Tatoeba 获取的测试数据。

princeton-nlp/LM-BFF 更好的Few-shot小样本微调语言模型.包括：1.基于提示（prompt）进行微调，关键是如何自动化生成提示模板；
2.将样本示例以上下文的形式添加到每个输入中，关键是如何对示例进行采样.

thunlp/PromptPapers 关于基于提示的预先训练语言模型的必读论文。

# 推荐系统

shenweichen/DeepCTR Easy-to-use,Modular and Extendible package of deep-learning based CTR models for search and recommendation. 

ChenglongChen/tensorflow-DeepFM 

cheungdaven/DeepRec  An Open-source Toolkit for Deep Learning based Recommendation with Tensorflow. 

lyst/lightfm  A Python implementation of LightFM, a hybrid recommendation algorithm. 

tensorflow/recommenders TensorFlow Recommenders is a library for building recommender system models using TensorFlow.

RUCAIBox/RecBole 统一，全面，高效的推荐库，包括： 
 * AFM,AutoInt,DCN,DeepFM,DSSM,FFM,FM,FNN,FwFM,LR,NFM,PNN,WideDeep,xDeepFM,BPR,ConvNCF,DGCF,DMF,FISM,GCMC,ItemKNN,LightGCN,NAIS,NeuMF,NGCF,Pop,SpectralCF,CFKG,
 * CKE（Collaborative Knowledge base Embedding 发自16年KDD，将KG与CF融合做联合训练）
 * KGAT Knowledge Graph Attention Network for Recommendation 用KG做增强，捕捉这种高阶交互式特征，做推荐预测。
 * KGCN,KGNNLS,
 * KTUP Unifying Knowledge Graph Learning and Recommendation:Towards a Better Understanding of User Preferences 一方面利用KG可以帮助更好的理解用户偏好。另一方面，用户-物品的交互可以补全KG，增强KG中缺少的事实。最终使两个部分都得到加强。
 * MKR(Multi-task Learning for KG enhanced Recommendation 融合KG和RC) 左边是推荐任务。用户和物品的特征表示作为输入，预测点击率y 右边是知识图谱任务。三元组的头结点h和关系r表示作为输入，预测的尾节点t 两者的交互由一个cross-feature-sharing units完成。由于物品向量和实体向量实际上是对同一个对象的两种描述，他们之间的信息交叉共享可以让两者都获得来自对方的额外信息，从而弥补了自身的信息稀疏性的不足。
 * ippleNet,BERT4Rec,Caser,DIN,FDSA,FPMC,GCSAN,GRU4Rec,GRU4RecF,GRU4RecKG,KSR,NARM,NextItNet,S3Rec,SASRec,SASRecF,SRGNN,STAMP,TransRec

oywtece/dstn

shenweichen/DSIN

facebookresearch/dlrm 深度学习推荐模型（DLRM）的实现

vze92/DMR Deep Match to Rank Model for Personalized Click-Through Rate Prediction DMR：Matching和Ranking相结合的点击率预估模型

kang205/SASRec 源于Transformer的基于自注意力的序列推荐模型

shichence/AutoInt 使用Multi-Head self-Attention进行自动的特征提取

xiangwang1223/neural_graph_collaborative_filtering 神经图协同过滤

UIC-Paper/MIMN 点击率预测的长序列用户行为建模的实践

motefly/DeepGBM 结合了GBDT 和神经网络的优点，在有效保留在线更新能力的同时，还能充分利用类别特征和数值特征。DeepGBM 由两大块组成，CatNN 主要侧重于利用 Embedding 技术将高维稀疏特征转为低维稠密特征，而 GBDT2NN 则利用树模型筛选出的特征作为神经网络的输入，并通过逼近树结构来进行知识蒸馏。

shenweichen/DeepMatch 用于推荐和广告的深度匹配模型库。训练模型和导出用户和项目的表示向量非常容易，可用于ANN搜索。

LeeeeoLiu/ESRM-KG 关键词生成的基于电商会话的推荐模型

zhuchenxv/AutoFIS 自动特征交互选择的点击率预测模型

pangolulu/exact-k-recommendation 解决推荐中带约束的Top-K优化问题

Scagin/NeuralLogicReasoning 神经协同推理,提出了一种新的神经逻辑推荐（NLR）框架，能够将逻辑结构和神经网络相结合，将推荐任务转化为一个逻辑推理任务。

ZiyaoGeng/Recommender-System-with-TF2.0 CTR预言论文进行复现，包括传统模型（MF，FM，FFM等），神经网络模型（WDL，DCN等）以及序列模型（DIN）。

allenjack/HGN 用矩阵分解的形式捕捉用户的长期兴趣，同时将短期兴趣进行拆分，分为group-level以及instance-level的，通过Hierarchical Gating来处理group-level的信息,item-item的乘积来捕捉商品之间的关系。

THUwangcy/ReChorus 用于Top-K推荐的通用PyTorch框架，具有隐式反馈，尤其是用于研究目的。BPR\NCF\Tensor\GRU4Rec\NARM\SASRec\TiSASRec\CFKG\SLRC\Chorus

RUCAIBox/CIKM2020-S3Rec 自我推荐学习，用于具有互信息最大化的顺序推荐

chenchongthu/SAMN 社交注意力记忆网络在推荐系统中的应用

Lancelot39/KGSF 基于知识图谱语义融合改进会话推荐系统
Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion

DeepGraphLearning/RecommenderSystems 顺序推荐 基于维度的推荐 社交推荐

FeiSun/BERT4Rec 基于BERT的顺序推荐

ChuanyuXue/CIKM-2019-AnalytiCup 2019-CIKM挑战赛，超大规模推荐之用户兴趣高效检索赛道 冠军解决方案 ,召回阶段基于 Item CF 相似性做召回( item-item 相似性),排序阶段,最终使用了 Catboost 和 Lightgbm 建模。

zyli93/InterHAt 通过分层注意力预测可解释的点击率。

SSE-PT/SSE-PT 基于Transformer的模型,但是和SASRec类似, 效果不错,但是缺少个性化,而且没有加入基于个性化的用户embedding。为了克服这种问题,本文提出来一种个性化的Transformer(SSE-PT),该方法相较于之前的方案提升了5%。

NVIDIA/NVTabular 为特征工程、前处理提供了更快的迭代速度，同时利用异步批量加载的方法有效提高了GPU的利用率，提供更快的加载速率。Merlin推荐系统框架的模块。

NVIDIA/HugeCTR a high efficiency GPU framework designed for Click-Through-Rate (CTR) estimating training ，在Embedding lookup上做了很多优化，可以轻易的通过数据和模型并行的方式将模型扩展到TB级别，在大规模参数的背景下，这给挖掘模型能力提供了更多的想象力。同时更快的训练速度也让算法工程师能够尝试更多的网络结构，挖掘最适合所研究问题的模型。

openai/triton OpenAI 正式的Triton是一种类 Python 的开源编程语言。能够高效编写 GPU 代码。例如，它可以用不到 25 行代码写出与 cuBLAS 性能相匹配的 FP16 矩阵乘法内核。此外，使用 Triton 成功生成比 PyTorch 同类实现效率高 2 倍的内核。

triton-inference-server/server 面向高吞吐低延时的生产环境的框架，通过Triton做线上推理，将TensorRT作为执行后端，能够有效降低Latency，并最大化地利用GPU资源。相比于一个纯CPU的方案，两者的结合使用能够使Latency达到原先的1/18，数据吞吐量达到原先的17.6倍。

lqfarmer/GraphTR 采用了GraphSAGE+FM+Transformer多种手段，粒度上从粗到细，交叉、聚合来自不同领域的异构消息，相比于mean/max pooling、浅层FC等传统聚合方式，极大提升了模型的表达能力

guyulongcs/CIKM2020_DMT 将兴趣建模、多任务学习、偏置学习等几部分进行融合，提出了DMT模型（Deep Multifaceted Transformers）

hwwang55/DKN DKN，将知识图表示融入到新闻推荐中。DKN是一种基于内容的用于点击率预估的深度推荐框架。DKN的主要部分是一个多通道、单词实体对齐的知识感知卷积神经网络，KCNN，其中融入了新闻在语意层面和知识层面的表示。KCNN将单词和实体作为多通道，在卷积过程中明确保留他们之间的对齐关系。

yusanshi/NewsRecommendation NRMS NAML LSTUR DKN Hi-Fi Ark TANR

johnny12150/GCE-GNN 提出了一种全局上下文增强(global-context enhanced)的GNN网络，称为GCE-GNN。能够从两种层次来学习物品的表征，包括global-level：从所有session构成的图上进行全局的表征；以及session-level：从单个session局部item转移图上进行局部的表征；最后融合二者，并通过注意力机制形成最终的序列表征，用于序列推荐任务。

BinbinJin/SD-GAR 第一篇将生成式对抗网络（GAN）框架应用于信息检索（包括推荐系统）的研究工作。在该工作中，IRGAN 训练了一个生成器和一个判别器，其中生成器用来自适应地生成合适的负样本以帮助判别器训练；而判别器则是用来判断样本是来自用户真实的反馈还是生成器生成的样本。通过两者交替式对抗性地训练达到互相提升效果的目的。

twchen/lessr 将会话记录构建成图来建模商品之间的跳转关系的图神经网络

NLPWM-WHU/AGNN 区分了推荐系统中的一般冷启动和严格冷启动，并提出了属性图神经网络方法有效应对严格冷启动的场景。

CRIPAC-DIG/SR-GNN 会话序列推荐的图应用 直接将会话序列建模为图结构数据，并使用图神经网络捕获复杂的项目物品item间转换，每一个会话利用注意力机制将整体偏好与当前偏好结合进行表示。同时这种方式也就不依赖用户的表示了，完全只基于会话内部的潜在向量获得Embedding，然后预测下一个点击。

uctoronto/SHAN Sequential Recommender System based on Hierarchical Attention Network 分层注意力网络SHAN用于序列推荐 。提出新颖的两层分层注意力网络，将上述特性考虑进来，用于推荐可能感兴趣的下一个商品。第一层注意力网络基于用户的历史购买商品的表示来学习用户的长期偏好，第二层通过将用户的长期和短期偏好结合起来，输出最终的用户表示。

chenghuige/mind MIND新闻推荐冠军分享细节揭秘

WayneDW/DeepLight_Deep-Lightweight-Feature-Interactions 轻量级特征交互算法deeplight 大幅加速ctr预估在线服务。 一，通过在浅层结构中精确搜索信息量更大的特征交互来加速模型推理，二，在深层结构中，从层内和层间对冗余的层和冗余的参数进行剪枝，三，促使embedding层的稀疏性，进而保持最有判别性的信息。为了解决预测延迟问题，我们通过结构修剪来加速预测，最终以46倍的速度提高而不会牺牲Criteo数据集上的最新性能。

JiachengLi1995/TiSASRec Time Interval Aware Self-Attention for Sequential Recommendation 时间间隔自注意力模型用于序列推荐。 基于序列模型框架对行为的时间戳进行建模，在下一个商品预测中探索不同时间间隔的影响。

wuch15/IJCAI2019-NAML 多视图学习新闻推荐系统 Neural News Recommendation with Attentive Multi-View Learning 可以通过利用不同种类的新闻信息来学习用户和新闻的特征表示。

guoday/Tencent2020_Rank1st  广告受众基础属性预估 2020 Tencent College Algorithm Contest, and the online result ranks 1st.

yuduo93/THIGE 基于时序异质交互图表示学习的商品推荐 将复杂异质的动态交互行为构建为时序异质交互图（Temporal Heterogeneous Interaction Graph, 简称为THIG）进而同时学习用户兴趣和商品表示用于商品推荐。本文提出了一种时序异质图上的表示学习方法，称之为THIGE，充分建模交互行为的异质性，刻画不同类型的兴趣偏好，并融合长、短期兴趣构建用户、商品表示。最后，在3个真实数据集上验证模型的有效性。

imsheridan/DeepRec 推荐、广告工业界经典以及最前沿的论文、资料集合

guyulongcs/CIKM2020_DMT 大型电子商务推荐系统中多目标排名的深层多面Transformers模型

weiyinwei/MMGCN 多模态图神经网络解决短视频推荐难题

microsoft/recommenders 推荐系统上的最佳实践。包括多个模型：ALS A2SVD BPR Caser DKN xDeepFM FAST LightFM/Hybrid Matrix Factorization LightGBM/Gradient Boosting Tree* LightGCN GeoIMC GRU4Rec Multinomial VAE LSTUR NAML NCF NPA NRMS NextItNet RBM RLRMC SAR SLi-Rec SUM Standard VAE SVD TF-IDF Vowpal Wabbit (VW)* Wide and Deep  FM&FFM

wujcan/SGL 基于图自监督学习的推荐系统。应用于「用户-物品二分图推荐系统」的「图自监督学习」框架。

wangjiachun0426/StackRec 通过迭代堆叠实现推荐系统的高效训练。采用对一个浅层序列推荐模型进行多次层堆叠（Layer Stacking），从而得到一个深层序列推荐模型。具体来说，训练过程包含以下步骤：1）预训练一个浅层序列推荐模型；2）对该模型进行层堆叠，得到一个两倍深度的模型；3）微调这个深层模型；4）将深层模型作为一个新的浅层模型，重复1）至3）直到满足业务需求。

xiangwang1223/neural_graph_collaborative_filtering 神经图协同过滤（NGCF）是一种基于图神经网络的新推荐框架，通过执行嵌入传播，在用户项二部图中以高阶连通性的形式对协同信号进行显式编码。

johnnyjana730/MVIN 提出multi-view item network (MVIN) ，从user和item来学习多个视角下的商品表示，进而进行商品推荐。在实体视图中，项目表示由KG中连接到它的实体来定义的。

weberrr/CKAN Collaborative Knowledge-aware Attentive Network for Recommender Systems 协作知识感知的注意力网络推荐系统 

danyang-liu/KRED KRED：基于知识感知的文档表示应用于新闻推荐。首先是用KGAT来表示每个实体，然后使用用实体的位置 实体出现频率 实体的类别等信息。再用Transformer来优化表征。最后做多任务：包括个性化推荐，项目到项目推荐、新闻流行预测、新类别预测和本地新闻检测等等。

CRIPAC-DIG/DGCF 动态图协同过滤算法DGCF 利用动态图来同时捕捉用户和商品之间的协同和序列关系的框架。提出三种更新机制： 零阶继承，一阶传播，二阶聚合，来表示新的交互发生时，该交互对用户或者商品的影响。基于这三种机制，交互发生时同时更新用户和商品的embedding，并且利用最新的embedding来给出推荐。

QYQ-bot/CLEA 运用对比学习解决购物篮推荐场景。（下一个购物篮推荐，也就是根据用户的历史购物篮序列，来推荐用户在下一次可能购买的商品集合。）

huangtinglin/MixGCF 基于多层嵌入合成负例用于推荐，相对NGCF 提高 26%, LightGCN 提高 22%

DyGRec/ASReP  反向预训练Transformer 增广序列推荐系统.解决序列推荐系统中的冷启动（cold-start）问题。为了解决该问题，我们提出需要对冷启动对应的短序列（short sequence）进行增广（Augmentation），从而能够补全信息而避免冷启动的问题。

NLPWM-WHU/EDUA 多样性推荐的 EDUA 模型。其采用双边分支网络作为双目标优化的主要架构，该架构既保持传统学习分支的准确性，又提高自适应学习分支的多样性。


# 金融股票 时间序列

jdb78/pytorch-forecasting pytorch的时间系列预测库，模型包括：RecurrentNetwork、DecoderMLP、NBeats 、DeepAR 、TemporalFusionTransformer。

QUANTAXIS/QUANTAXIS 量化金融策略框架

ricequant/rqalpha 从数据获取、算法交易、回测引擎，实盘模拟，实盘交易到数据分析，为程序化交易者提供了全套解决方案

cedricporter/funcat 将同花顺、通达信、文华财经麦语言等的公式写法移植到了 Python

georgezouq/awesome-deep-reinforcement-learning-in-finance 金融市场上使用的那些AI（RL/DL/SL/进化/遗传算法）的集合

wangshub/RL-Stock 如何用深度强化学习自动炒股。

tensortrade-org/tensortrade 一个开源强化学习框架，用于训练，评估和部署强大的交易程序。

bsolomon1124/pyfinance 为投资管理和证券收益分析而构建的Python分析包。主要是对面向定量金融的现有包进行补充，如pyfolio和pandas-datareader等。pyfinance包含六个模块，它们分别是：datasets.py ：金融数据下载，基于request进行数据爬虫；general.py：通用财务计算，例如主动份额计算，收益分配近似值和跟踪误差优化；ols.py：回归分析，支持pandas滚动窗口回归；options.py：期权衍生品计算和策略分析；returns.py：通过CAPM框架对财务时间序列进行统计分析，旨在模拟FactSet Research Systems和Zephyr等软件的功能，并提高了速度和灵活性；utils.py：基础架构。

arrigonialberto86/deepar Amazon于2017年提出的基于深度学习的时间序列预测方法

fjxmlzn/DoppelGANger 使用GAN共享网络时间序列数据：挑战，初步承诺和未解决的问题，IMC 2020（最佳论文入围）

AIStream-Peelout/flow-forecast 一个开源的深度学习时间序列预测库。包括模型：Vanilla LSTM、Full transformer、Simple Multi-Head Attention、Transformer w/a linear decoder、DA-RNN (CPU only for now)。

microsoft/qlib Qlib是一个面向AI的量化投资平台，旨在实现潜力，增强研究能力并创造AI技术在量化投资中的价值。包括多个模型。

tslearn-team/tslearn 时间序列机器学习python工具包，其中包括了一些基本的时间序列预测或者分类模型，如多层感知机，SVR，KNN以及基本的数据预处理工具和数据集的生成与加载模块。

blue-yonder/tsfresh 时间序列特征提取python工具包，它会自动计算出大量的时间序列特征。此外，该工具包还包含了一些方法，用于评估回归或分类任务中这些特征的解释能力和重要性。

johannfaouzi/pyts 时间序列分类Python工具包。提供预处理工具及若干种时间序列分类算法。

quantopian/alphalens Python量化分析库，量化网站quantopian开发维护的量化三件套之一，用于股票因子(alpha)的性能分析。alphalens与zipline以及pyfolio常常一同使用，其中，pyfolio提供财务组合的性能和风险分析，zipline用于量化策略回测。alphalens的主要功能包括对一个alpha因子进行统计和绘图，包括：因子收益分析、因子信息系数分析、换手率分析以及分组分析。

quantopian/pyfolio 用于金融投资组合的性能和风险分析。它可以很好地与Zipline回测库一起工作。

quantopian/zipline 美国著名的量化策略平台quantopian开发和维护的量化交易库，并且quantopian量化平台的回测引擎也是基于zipline的，除此之外，像国内比较有名的三大矿聚宽(JointQuant)、米筐(RiceQuant)、优矿的回测引擎也是基于此。另外，由于quantopian平台多年的使用，zipline的专业性是可以保证的，并且zipline在github中的代码也在保持不断更新和改进。zipline是一种事件驱动（event-driven）的回测框架，有完整的文档和社区，如果你是对国外美股交易感兴趣，那么zipline将比较合适；但是对于国内像A股的数据则无法支持，只能通过本地化的数据进行回测。

gbeced/pyalgotrade 一个事件驱动的回测框架，虽然不如zipline的名气大，但是同样也具有完善的社区和详细的文档。据说pyalgotrade的运行速度和灵活度要比zipline强，但是缺点是不支持pandas。

mementum/backtrader 一个功能强大的量化策略回测平台。backtrader允许你专注于编写可重用的交易策略、指标和分析工具，而不是花时间构建基础设施。

enigmampc/catalyst 对于虚拟货币交易的量化回测平台。Catalyst是一个底层基于zipline的算法交易框架，目前比较成熟，并且可以支持策略的回测与实盘（ 目前支持四家交易所 Binance, Bitfinex, Bittrex, Poloniex) 。

vnpy/vnpy 国内由陈晓优团队开发量化交易框架，它目前在github上star和fork的数量已经超过了zipline，目前是全球开源量化框架的首位。vn.py主要侧重于实盘交易，同样支持通过历史数据进行回测，包括数据的可视化、收益结果、参数调优等，除此之外，它还具备一些常用的CTA策略、SpreadTrading价差交易、行情录制等功能，并且它还具备完善的社区以及教程。新手在使用时，可以通过它的GUI环境VN Station进行使用，同时也可以基于它的策略模版进行自定义的策略开发。

waditu/tushare 拥有丰富的数据内容，如股票、基金、期货、数字货币等行情数据，公司财务、基金经理等基本面数据。其SDK开发包支持语言，同时提供HTTP Restful接口，最大程度方便不同人群的使用。并且，它提供多种数据储存方式，如Oracle、MySQL，MongoDB、HDF5、CSV等，为数据获取提供了性能保证。

jindaxiang/akshare 基于 Python 的财经数据接口库, 目的是实现对股票、期货、期权、基金、外汇、债券、指数、加密货币等金融产品的基本面数据、实时和历史行情数据、衍生数据从数据采集、数据清洗到数据落地的一套工具, 主要用于学术研究目的。特点是获取的是相对权威的财经数据网站公布的原始数据, 通过利用原始数据进行各数据源之间的交叉验证, 进而再加工, 从而得出科学的结论。

zhouhaoyi/Informer2020 效果远超Transformer的长序列预测，提出了ProbSparse self-attention机制来高效的替换常规的self-attention并且获得了的O（LlogL)时间复杂度以及O(LlogL)的内存使用率,提出了self-attention distilling操作，它大幅降低了所需的总空间复杂度O((2-e)LlogL)；我们提出了生成式的Decoder来获取长序列的输出，这只需要一步，避免了在inference阶段的累计误差传播；

deeptime-ml/deeptime 用于分析时间序列数据，包括降维，聚类和马尔可夫模型估计

AI4Finance-LLC/FinRL-Library 哥大开源“FinRL”: 一个用于量化金融自动交易的深度强化学习库

nnzhan/MTGNN 通用的图神经网络框架 MTGNN，通过图学习模块融合外部知识和变量之间的单向关系，再使用 mix-hop 传播层和膨胀 inception 捕获空间和时序依赖。

VachelHU/EvoNet Time-Series Event Prediction with Evolutionary State Graph 将时间序列转化为动态图进行表示的方法。该方法成功在阿里云 ·SLS 商业化，作为一项智能巡检服务，可以对大规模时间序列进行异常检测与分析。

microsoft/StemGNN 基于图谱分解的时间序列预测。进一步提高多元时间序列预测的准确性。StemGNN 在spectral domain中捕获系列间(inter-series)相关性和时间依赖性(temporal dependencies)。它结合了图傅立叶变换 (GFT) 和离散傅立叶变换 (DFT)，GFT对序列间(inter-series)相关性进行建模，而离散傅立叶变换 (DFT) 则对端到端框架中的时间依赖性(temporal dependencies)进行建模。通过 GFT 和 DFT 后，谱表示具有清晰的模式，可以通过卷积和序列学习模块进行有效预测。

fulifeng/Temporal_Relational_Stock_Ranking 基于图神经网络、图谱型数据的收益预测模型

Heerozh/spectre GPU 加速的因子分析库和回测工具。

emadeldeen24/TS-TCC 一个无监督的时间序列表示学习框架，通过时间和上下文对比。

nnzhan/MTGNN 基于图神经网络的多变量时间序列预测模型 

[adarnn](https://github.com/jindongwang/transferlearning/tree/master/code/deep/adarnn) 提出自适应的RNN模型，使得其可以更好地泛化。由时序相似性量化和时序分布匹配算法组成，前者用于表征时序中的分布信息，后者通过分布匹配构建广义RNN模型。

facebookresearch/Kats 用于分析时间系列数据的工具包，一个轻量级、易于使用、通用和可扩展的框架，用于执行时间系列分析，从了解关键统计数据和特征、检测变化点和异常，到预测未来趋势。

slaypni/fastdtw 近似动态时间规整算法，提供与 O（N） 时间和内存复杂性的最佳或接近最佳对齐。

ourownstory/neural_prophet 基于神经网络的时间系列模型，灵感来自 Facebook Prophet 和 AR-Net，建立在 PyTorch 之上。

## 强化学习 Reinforcement Learning

ray-project/ray 构建分布式机器学习应用提供简单和通用式的API。Ray打包了Tune、RLlib、RaySGD和Ray Serve等多款机器学习库。

astooke/rlpyt

Generative Adversarial User Model

dennybritz/reinforcement-learning

keiohta/tf2rl

rlgraph/rlgraph

deepmind/trfl

Ceruleanacg/Personae

dgriff777/a3c_continuous

google/dopamine

keras-rl/keras-rl

openai/gym

georgezouq/awesome-deep-reinforcement-learning-in-finance 金融市场上使用的那些AI（RL/DL/SL/进化/遗传算法）的集合

google/brain-tokyo-workshop 世界模型 prettyNEAT

google-research/football

tensortrade-org/tensortrade 一个开源强化学习框架，用于训练，评估和部署强大的交易程序。

Baekalfen/PyBoy Game Boy emulator written in Python

google-research/batch_rl 离线强化学习

tensorflow/agents TF-Agents是TensorFlow中的强化学习库

YingtongDou/Nash-Detect 通过Nash强化学习进行鲁棒的垃圾邮件发送者检测

deepmind/acme 强化学习的研究框架，强化学习组件和代理库

XinJingHao/TD3 TD3强化算法的实现

huawei-noah/xingtian 刑天（XingTian）是一个组件化的库，用于开发和验证强化学习算法。它支持多种算法，包括DQN，DDPG，PPO和IMPALA等，可以在多个环境中训练代理，例如Gym，Atari，Torcs，StarCraft等。

thu-ml/tianshou 天授是基于纯PyTorch强化学习的平台。与现有的强化学习库主要基于TensorFlow，具有许多嵌套类，不友好的API或速度较慢的现有学习库不同，天守提供了快速的模块化框架和pythonic API，用于以最少的行数构建深度强化学习代理代码。

Jingliang-Duan/Distributional-Soft-Actor-Critic 一种用于连续控制任务的强化学习算法—DSAC，其优势在于减少Q值的过估计并显著改进策略的性能。证明了强化学习中引入分布式回报可显著降低Q值的过估计误差，并定量表明此误差与分布的方差呈反比关系。与主流RL算法相比，策略性能提升20% 以上。

tencent-ailab/TLeague 一种基于竞争性自我驱动的多智能体强化学习框架。

minerllabs/minerl Minecraft 游戏环境

mwydmuch/ViZDoom 基于ZDoom末日的AI研究平台，可从原始视觉信息进行强化学习。

openai/retro 复古游戏 

google-research/football 基于开源游戏Game Football的RL环境

TorchCraft/TorchCraftAI 可让您建立机器人以学习玩《星际争霸：巢穴之战》。

deepmind/pysc2 星际争霸II强化学习环境

datamllab/rlcard 纸牌（扑克）游戏中的强化学习/ AI机器人-大酒杯，勒杜克，德克萨斯州，窦滴竹，麻将，UNO。

sourceforge.net/projects/torcs ORCS，开放式赛车模拟器是一种高度便携式的多平台赛车模拟。

Microsoft/AirSim 基于Unreal Engine / Unity的自动驾驶汽车开源模拟器

carla-simulator/carla 用于自动驾驶研究的开源模拟器。

aitorzip/DeepGTAV GTAV的插件，可将其转变为基于视觉的自动驾驶汽车研究环境。

deepdrive/deepdrive Deepdrive是一个模拟器，它使拥有PC的任何人都能推动最新的自动驾驶

robotology/gym-ignition 使用Ignition Gazebo模拟开发OpenAI Gym机器人环境的框架

stanfordnmbl/osim-rl 具有肌肉骨骼模型的强化学习环境

lsw9021/MASS 全身肌肉骨骼系统实现了基本的仿真和控制。骨骼运动由肌肉的驱动力来驱动，并与激活水平相协调。通过与python和pytorch的接口，可以使用深度强化学习（DRL）算法，例如近端策略优化（PPO）。

deepmind/lab DeepMind Lab为学习代理提供了一套具有挑战性的3D导航和解谜任务。它的主要目的是充当人工智能（尤其是深度强化学习）研究的测试平台。

maximecb/gym-minigrid OpenAI Gym的简约gridworld软件包

maximecb/gym-miniworld 用于RL和机器人研究的简单3D室内模拟器

minosworld/minos MINOS：多模式室内模拟器 旨在支持在复杂的室内环境中为目标定向导航开发多传感器模型。MINOS利用复杂3D环境的大型数据集，并支持多模式传感器套件的灵活配置。

facebookresearch/habitat-sim 灵活，高性能的3D仿真器，适用于嵌入式AI研究。

facebookresearch/habitat-lab 一个模块化的高级库，可在各种任务，环境和模拟器中训练嵌入式AI代理。

facebookresearch/house3d 逼真的丰富3D环境 由数以千计的室内场景组成，这些场景配有从SUNCG数据集中获取的各种场景类型，布局和对象。它包含超过4.5万个室内3D场景，从工作室到带有游泳池和健身室的两层房屋。

staghuntrpg/RPG 通过奖励随机化发现多智能体游戏中多样性策略行为。通过奖励随机化对原始游戏（StagHunt）的奖励（reward）进行扰动，将问题转化为在扰动后的游戏中寻找合作策略，然后再回到原始游戏中进行微调（fine-tune），进而找到最优策略。

daochenzha/rapid 一种为每个回合的探索动作打分和排序的机制，以选出好的探索行为。不同于以往基于内部奖励的方法，回合排序算法将好的探索行为记录下来，然后通过模仿学习鼓励智能体探索。初步结果表明，该方法具有非常好的效果，特别是在具有随机性的环境中。

AI4Finance-LLC/ElegantRL 基于PyTorch的轻量-高效-稳定的深度强化学习框架

datawhalechina/easy-rl  强化学习中文教程

kwai/DouZero 斗地主AI 

opendilab/DI-engine 通用的决策智能引擎。它支持最基本的深度强化学习 （DRL） 算法，如 DQN、PPO、SAC 和域特定算法，如多代理 RL 中的 QMIX、逆RL 中的 GAIL 和探索问题的 RND。还支持各种培训管道和定制决策 AI 应用程序。

kzl/decision-transformer UC 伯克利、FAIR 和谷歌大脑的研究者提出了 Decision Transformer，通过序列建模进行强化学习的架构。

instadeepai/Mava 用于构建多智能体强化学习 (MARL) 系统的库。Mava 为 MARL 提供了有用的组件、抽象、实用程序和工具，并允许对多进程系统训练和执行进行简单的扩展，同时提供高度的灵活性和可组合性。 

google/brax 物理模拟引擎Brax，只需一个TPU/GPU，就能和数千个CPU或GPU的计算集群的速度一样快，直接将所需时间缩短到几分钟

sjtu-marl/malib 专门面向 基于种群的多智能体深度强化学习 PB-MARL 的开源大规模并行训练框架。MALib 支持丰富的种群训练方式（例如，self-play, PSRO, league training)，并且实现和优化了常见多智能体深度强化学习算法，为研究人员降低并行化工作量的同时，大幅提升了训练效率。此外，MALib 基于 Ray 的底层分布式框架，实现了全新的中心化任务分发模型，相较于常见的多智能体强化学习训练框架（RLlib，PyMARL，OpenSpiel），相同硬件条件下吞吐量和训练速度有着数倍的提升。现阶段，MALib 已对接常见多智能体环境（星际争霸、谷歌足球、棋牌类、多人 Atari 等），后续将进一步提供对自动驾驶、智能电网等场景的支持。

octavio-santiago/Super-Mario-Land-AI 机器学习和 AI 算法玩超级马里奥。

# 语音
JasonWei512/Tacotron-2-Chinese 中文语音合成

TensorSpeech/TensorflowTTS Tensorflow 2的实时最新语音合成

audier/DeepSpeechRecognition 基于深度学习的中文语音识别系统

athena-team/athena 基于序列到序列的语音处理引擎的开源实现

espnet/espnet  End-to-End Speech Processing Toolkit 端到端的语音处理工具箱，主要特性：kaldi风格的处理模式、ASR、TTS、语音翻译、机器翻译、语音转换、DNN框架

kan-bayashi/ParallelWaveGAN Parallel WaveGAN (+ MelGAN & Multi-band MelGAN) implementation with Pytorch 

KuangDD/zhrtvc  好用的中文语音克隆兼中文语音合成系统，包含语音编码器、语音合成器、声码器和可视化模块。 

JasonWei512/Tacotron-2-Chinese 中文语音合成

lturing/tacotronv2_wavernn_chinese tacotronV2 + wavernn 实现中文语音合成(Tensorflow + pytorch) 

JasonWei512/wavenet_vocoder  WaveNet 声码器 

deezer/spleeter 人声分离模型

ZhengkunTian/OpenTransformer 语音识别的无重复序列到序列模型，实现 aishell 6.7％的CER。

mobvoi/wenet 生产优先和生产就绪的端到端语音识别工具包 在aishell测试上已经做到5以内的CER 

alphacep/vosk-api Offline speech recognition API for Android, iOS, Raspberry Pi and servers with Python, Java, C# and Node 支持十七种语言，提供中文语言模型。

tencent-ailab/pika 基于Pytorch和Kaldi的轻量级语音处理工具包 PIKA 具备以下特征：即时数据增强和特征加载器；TDNN Transformer编码器，以及基于卷积和 Transformer 的解码器结构；RNNT训练和批解码；利用 Ngram FST 的 RNNT 解码；RNNT最小贝叶斯风险MBR训练；用于 RNNT 的 LAS 前向与后向重评分器；基于高效 BMUF的分布式训练。

tulasiram58827/TTS_TFLite 提供了TFLite中广泛流行的文本语音转换（TTS）模型的集合。

speechbrain/speechbrain 基于 PyTorch 的开源一体化语音工具包，SpeechBrain 可用于开发最新的语音技术，包括语音识别、说话者识别、语音增强、多麦克风信号处理和语音识别系统等，且拥有相当出色的性能。团队将其特征概况为「易于使用」、「易于定制」、「灵活」、「模块化」等。

Snowdar/asv-subtools 基于Kaldi和PyTorch推出了一套高效、易于开发扩展的声纹识别开源工具—ASV-Subtools。

Rudrabha/Wav2Lip 唇语识别 唇语同步 ，用来生成准确的唇语同步视频。

[wav2vec](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec/unsupervised) 无监督语音识别 性能直逼监督模型,只需要从录制的语音音频和未配对的文本中学习，无需进行任何转录。

facebookresearch/voxpopuli 目前世界上最大的多语言语音数据集，涵盖了23种语言，时长超过40万小时。

speechio/leaderboard SpeechIO 排行榜：用于自动语音识别的大型、强大、全面的基准测试平台。


# 生物医药

* ## 蛋白质结构
  * deepmind/alphafold 此包提供了 AlphaFold v2.0 的推理流线的实现。AlphaFold是Google旗下DeepMind开发的一款人工智能程序，它使用深度学习算法通过蛋白质序列来预测蛋白质结构。蛋白质通过卷曲折叠会构成三维结构，蛋白质的功能正由其结构决定。了解蛋白质结构有助于开发治疗疾病的药物。
  * RosettaCommons/RoseTTAFold 结合AlphaFold相关思想的网络架构，并通过三轨网络获得了最佳性能，其中 1D 序列、2D 距离图和3D坐标的信息依次转换和集成。三轨网络精度接近AlphaFold2，能够快速解决具有挑战性的 X-ray晶体学和冷冻电镜结构建模问题，并提供对当前未知结构蛋白质功能的见解。该网络还能够仅从序列信息中快速生成准确的蛋白质-蛋白质复合物模型。
  * salesforce/provis BERTology Meets Biology: Interpreting Attention in Protein Language Models 注意力机制在蛋白质语言模型的应用
  * ElwynWang/DeepFragLib  基于深度神经网络和改进的片段测序方法从头预测蛋白质结构
  * bowman-lab/diffnets 采用DiffNets通过比较结构集来深度学习蛋白质生化特性的结构决定因素
  * nadavbra/protein_bert ProteinBERT：专为蛋白质序列和功能设计的通用深度学习模型

* ## 药物发现
  * DeepGraphLearning/torchdrug 药物发现强大而灵活的机器学习平台
  * [jdurrant/deepfrag](https://git.durrantlab.pitt.edu/jdurrant/deepfrag) 药物发现是一个成本高昂且耗时的过程。在药物发现前期，研究人员试图找到能够初步抑制某些疾病关联蛋白的苗头化合物。但这些化合物必须经过先导优化，包括添加或交换某些化学部分，旨在提高化合物的结合亲合力或其他与吸收、分布、代谢、排泄和毒性有关的化学性质（ADMET）。而计算机辅助药物设计（CADD）能够加速前期的这些研究。例如，作者团队最近开发了基于3D卷积神经网络的模型DeepFrag来进行更进一步的先导优化，不幸的是，基于深度学习的模型对于非计算机专业研究者并不友好。为了追求更高的易用性，作者开发了名为DeepFrag的网页应用，为对编程不太熟悉的研究人员提供了图形化的界面，利用本地资源即可运行DeepFrag进行CADD的研究。
  * Mariewelt/OpenChem 用于计算化学和药物设计研究的深度学习工具包

* ## 药物-靶标 药物-药物 化合物-蛋白质 相互作用
  * kexinhuang12345/DeepPurpose 一个基于PyTorch的工具包来解锁50多个用于药物-靶标相互作用（Drug-Target Interaction）预测的模型。DTI预测是新药研发中的一项基本任务。DeepPurpose的操作模式是像scikit-learn一样。只需几行代码，就可以利用最前沿的深度学习和药物研发模型。DeepPurpose还有一个简单的界面来做DTI预测的两个重要应用：虚拟筛选（Virtual Screening）和旧药新用（Drug Repurposing）。
  * ETHmodlab/molgrad 通过将积分梯度可解释人工智能（XAI）方法应用于图神经网络模型，提高了理性分子设计的建模透明度，并基于四个药理学相关ADME终点的实验，验证了所提出的方法能够突出与已知药效团基序一致的分子特征和结构元素，正确识别性质断崖，并提供了对非特异性配体-靶标相互作用的见解。
  * thinng/GraphDTA 使用图神经网络预测药物-靶标的结合亲和力
  * isjakewong/MIRACLE 多视图图对比表示学习用于药物药物相互作用预测
  * FangpingWan/DeepCPI 基于深度学习的化合物和蛋白质相互作用预测框架
  * yueyu1030/SumGNN multi-typed drug interaction prediction via efficientknowledge graph summarization 基于高效知识图谱汇总的多类型药物关联预测。 整合了DDI信息（药物-药物相互作用）以及生物医学KG数据，并提出了有效的聚合机制以进行DDI预测。实验结果表明，该模型具有良好的预测性能。
  * kanz76/SSI-DDI 作者提出了SSI-DDI，一种预测药物之间不良DDI的深度学习框架。该方法首次将药物间相互作用预测的任务转化为子结构间相互作用预测的任务。实验结果表明，该方法有着目前最好的性能。并在transductive和inductive (冷启动场景)设置方面都取得了良好的效果。
  * jacklin18/KGNN IJCAI'20 "KGNN: Knowledge Graph Neural Network for Drug-Drug Interaction Prediction" 基于知识图谱的图神经网络（KGNN），以解决DDI预测问题。该框架可通过在KG中挖掘相关联的关系，来有效地捕获药物及其潜在的邻域实体信息。

* ## 分子
  * futianfan/CORE 利用复制和改进策略自动优化分子
  * zhang-xuan1314/Molecular-graph-BERT 面向药物分子性质预测的大规模原子表征预训练模型
  * microsoft/Graphormer 图结构数据上的Transformer变种，应用于2D 分子化学结构图预测分子性质，还可以应用于主流图预测任务、社交网络的推荐和广告、知识图谱、自动驾驶的雷达点云数据、对交通物流运输等的时空预测和优化、程序理解和生成等等，还包括分子性质预测所涉及的行业，比如药物发掘、材料发现、分子动力学模拟、蛋白质结构预测等等。
  * HIPS/neural-fingerprint 图卷积网络用于学习分子指纹。使用神经网络在数据样本中归纳总结，然后来预测新型分子的属性或者性质。
  * binghong-ml/MolEvol 通过可解释进化进行分子优化
  * MinkaiXu/ConfVAE-ICML21 基于双层规划的端到端分子构象生成框架
  * mohimanilab/molDiscovery 使用质谱数据预测分子的身份
  * binghong-ml/retro_star 自提升策略规划真实且可执行的分子逆合成路线
  * zhang-xuan1314/Molecular-graph-BERT 利用无监督原子表示学习来预测分子性质
  * marcopodda/fragment-based-dgm 基于片段的分子深度生成模型.作者在ZINC数据集上进行了实验，该数据由250K类药物化合物组成。为了进一步评估LFM的影响，作者还使用了Pub Chem Bio Assay(PCBA)数据集测试了模型变体，该数据集包括约440k小分子。
  * torchmd/torchmd 一个混合经典和机器学习势的分子模拟（molecular simulations）的框架。通过将MD（经典分子动力学）中的键合和非键合力术语扩展到任意复杂的DNN上，实现了机器学习势的快速成型和集成。TorchMD关键点:一，PyTorch编写，容易集成其他ML模型；二，提供执行端到端可微模拟能力，在参数上都是可微的。
  * MolecularAI/GraphINVENT 基于GNN的分子生成平台
  * shenwanxiang/bidd-molmap MolMapNet 可预测药物特性，通过广泛学习的基于知识的分子表示对药物特性进行开箱即用的深度学习预测
  * DeepGraphLearning/GraphAF 基于Flow的自回归模型，以生成真实多样的分子图。由于标准化Flow的灵活性，GraphAF能够模拟复杂的分子分布，并在实验中生成新的和100%有效的分子。
  * anny0316/Drug3D-Net 提出了一种新的基于分子空间几何结构的深度神经网络结构Drug3D-Net，用于预测分子性质。它是基于网格的三维卷积神经网络，具有时空门注意模块，可以提取卷积过程中分子预测任务的几何特征。

* ## 抗菌肽
  * vail-uvm/amp-gan 一种基于双向条件生成对抗网络的抗菌肽（AMPs）设计方法AMPGAN v2。AMPGAN v2使用生成器和鉴别器来学习数据驱动的先验知识，并使用条件变量控制生成。
  * reymond-group/MLpeptide 机器学习设计非溶血性抗菌肽。使用来自DBAASP的数据训练RNN来设计非溶血性抗菌肽(Antimicrobial peptides, AMP)，合成并测试了28个生成肽，鉴定出针对绿脓杆菌、鲍曼不动杆菌和耐甲氧西林金黄色葡萄球菌 (MRSA) 的8种新的非溶血性 AMP。结果表明机器学习可以用来设计非溶血性AMP。
  * IBM/controlled-peptide-generation  IBM利用深度生成模型和分子动力学模拟加速抗菌肽发现


KailiWang1/DeepDTAF 预测蛋白质与配体结合亲和力的深度学习方法

cansyl/MDeePred 通过一种多通道蛋白质的特征化来解决深度学习下药物发现中亲和力预测问题

mims-harvard/TDC Therapeutics Data Commons (TDC)，第一个机器学习在生物医药的大规模数据集。TDC目前包含了20+有意义的任务，和70多个高质量数据集，从靶蛋白的发现，药物动力学，安全性，药物生产都有包含到。而且不仅仅是小分子，还有抗体，疫苗，miRNA等。之后也会加入CRISPR，Clinical Trials等等。

lrsoenksen/CL_RNA_SynthBio  RNA合成生物学的深度学习

lanagarmire/DeepImpute 一种基于深度神经网络来插补单细胞RNA测序数据的方法

emreg00/toolbox 疾病和药物相关的生物学数据集时所使用的各种脚本。它包含用于数据处理的通用实用程序（例如，解析，基于网络的分析，邻近性等）。

ruoqi-liu/DeepIPW 基于真实世界患者数据的药物重定位的深度学习框架

CutillasLab/DRUMLR 利用机器学习预测抗癌药物疗效。提出Drug Ranking Using ML方法，使用omics数据，根据药物抗肿瘤细胞增殖疗效对超过400种药物进行排序。

kaist-amsg/Synthesizability-PU-CGCNN  基于半监督学习的晶体结构的合成预测

WLYLab/PepFormer 基于Transformer的对比学习框架实现多肽可检测性预测

NYSCF/monoqlo_release 提出了模块化的深度学习框架Monoqlo来自动识别细胞集落，并从细胞成像中识别克隆性。

deepmodeling/deepks-kit DeePKS: A Comprehensive Data-Driven Approach toward Chemically Accurate Density Functional Theory 提出了构建准确且高效的密度泛函模型的通用机器学习框架，并且利用这一框架训练了具有化学精度的密度泛函模型，应用于电子结构性质的计算。

juexinwang/scGNN 新型的用于单细胞RNA测序分析的图神经网络框架

liulizhi1996/HPOFiller 一种基于图卷积网络(GCN)的方法，用于预测缺失的HPO注释。 人类表型本体(HPO)是描述人类疾病中遇到的表型异常的标准化词汇（疾病的术语）。探索人类蛋白质和异常表型之间的关系在疾病的预防、诊断和治疗中具有重要意义。

zty2009/GCN-DNN 基于图卷积网络和深度神经网络的药物靶点相互作用识别

WebyGit/CGINet 大规模药物信息网络构建及图卷积预测模型

ziyujia/SalientSleepNet 用于睡眠分期的多模态凸波检测网络

ziyujia/Physiological-Signal-Classification-Papers 500余篇基于机器学习/深度学习的生理信号分类论文列表

ziyujia/Sleep-Stages-Classification-Papers 基于机器学习/深度学习的睡眠阶段分类论文列表

ziyujia/Motor-Imagery-Papers 基于机器学习/深度学习的运动想象分类论文列表

BojarLab/SweetNet 图卷积神经网络分析复杂碳水化合物。

kekegg/DLEPS 利用深度学习从基因转录数据中预测药物疗效

jaswindersingh2/SPOT-RNA2 利用进化概况、突变耦合和二维迁移学习改进了RNA二级结构和三级碱基配对预测

QSong-github/scGCN 单细胞图卷积网络模型(single-cell Graph Convolutional Network)可以实现跨越不同数据集的知识转移(knowledge transfer)。通过在30个单细胞组学数据集上进行基准测试实验，结果表明scGCN在利用来自不同组织、平台和物种以及分子层的细胞方面展现了优于其他方法的准确性。

/mauragarofalo/LICTOR 抗体体细胞突变的机器学习分析预测免疫球蛋白轻链毒性

JieZheng-ShanghaiTech/KG4SL 用于人类癌症合成致死预测的知识图神经网络


# 机器视觉

ouyanghuiyu/chineseocr_lite 超轻量级中文ocr

minivision-ai/photo2cartoon 人像卡通化探索项目

hugozanini/realtime-semantic-segmentation 使用TensorFlow.js实施RefineNet以在浏览器中执行实时实例分割

iPERDance/iPERCore 处理人体图像合成任务。其中包括人体运动模仿、外观转换和新视角合成等。并且，该项目的代码、数据集已开源。

facebookresearch/pifuhd 使用AI从2D图像生成人的3D高分辨率重建

LeonLok/Multi-Camera-Live-Object-Tracking 多摄像头实时目标跟踪和计数，使用YOLOv4，Deep SORT和Flask

cfzd/Ultra-Fast-Lane-Detection 论文“超快速结构感知深度车道检测”的实现

RangiLyu/nanodet NanoDet：轻量级（1.8MB）、超快速（移动端97fps）目标检测项目

kornia/kornia 基于 PyTorch 的可微分的计算机视觉 （differentiable computer vision） 开源库， 实现了：可微的基础计算机视觉算子。可微的数据增广（differentiable data augmentation）。OpenCV 和 PIL 都是不可微的，所以这些处理都只可以作为图像的预处理而无法通过观察梯度的变化来对这些算子进行优化 （gradient-based optimization）。因此，Kornia 便应运而生。

microsoft/Bringing-Old-Photos-Back-to-Life 旧照片修复

architras/Advanced_Lane_Lines 基于阈值的车道标记

open-mmlab/mmskeleton 用于人体姿势估计，基于骨骼的动作识别和动作合成。

facebookresearch/pytorch3d 基于PyTorch将深度学习与3D进行结合的研究框架。

orpatashnik/StyleCLIP 文本驱动的StyleGAN风格生成图像处理

thepowerfuldeez/facemesh.pytorch 单目实时人脸表面3D点云提取

facebookresearch/pytorchvideo 为视频理解研究打造的深度学习库。

rwightman/pytorch-image-models PyTorch图像类模型库，包括：ResNet, ResNeXT, EfficientNet, EfficientNetV2, NFNet, Vision Transformer, MixNet, MobileNet-V3/V2, RegNet, DPN, CSPNet, and more 

google-research/vision_transformer Vision Transformer and MLP-Mixer Architectures 视觉Transformer和 MLP-混合器架构，Transformer应用于视觉，纯多层感知机视觉架构。

xinntao/Real-ESRGAN 旨在开发通用图像恢复的实用算法。

China-UK-ZSL/ZS-F-VQA 一种适用于零样本视觉问答（ZS-VQA）的基于知识图谱的掩码机制，更好结合外部知识的同时，缓解了误差传播对于模型性能的影响。

anibali/margipose 基于2D边缘热图的3D人体姿态估计

wmcnally/evopose2d 神经架构搜索推动2D姿态识别边界

hellojialee/OffsetGuided Bottom-up人体姿态估计最优网络,多人关键点坐标的编解码方法.

ziwei-zh/CorrPM 关联人体边缘，人体姿态与人体解析.研究了人的语义边界和关键点位置如何共同改善人的部件解析性能。

Megvii-BaseDetection/YOLOX 高性能目标检测器YOLOX。并将YOLO检测器切换到anchor-free的方式，并结合其他先进的检测技术，如decouple head和标签分配策略SimOTA，实现了当前目标检测最优性能。

luost26/diffusion-point-cloud 基于非平衡态热力学的全新三维点云生成模型

PeterWang512/GANSketching 绘制您自己的 GAN：用手绘草图自定义 GAN 模型。

microsoft/AutoML/tree/main/iRPE 视觉位置编码，在ImageNet和COCO上，与原始版本相比，分别获得了1.5%（top-1 Acc）和1.3%（mAP）的性能提升（无需任何调参）。

yuhuan-wu/P2T 基于金字塔池化的视觉Transformer，可用于各类下游场景理解任务。

yangxy/GPEN 用于脸部高清增强,还能将黑白人物照转成彩色照片。GPEN模型明显优于其他的修复人脸的GAN模型。

hzwer/arXiv2020-RIFE 视频帧插值的实时中级流量估计.旷视和北大提出的一种实时中间流估计算法。用于视频帧插值，能够改善伪影、让视频更丝滑。

Justin62628/Squirrel-RIFE 基于RIFE算法的中文补帧软件.

nihui/rife-ncnn-vulkan RIFE，视频帧插值的实时中级流量估计与 ncnn 库一起实现

jantic/DeOldify 基于NoGAN技术，保证视频着色的稳定性，例如，视频中的同一件衣服，不至于转换成多种颜色。

zhangmozhe/Deep-Exemplar-based-Video-Colorization 基于深层范例的视频着色,着色时间的连贯性与稳定性

AliaksandrSiarohin/first-order-model 图像动画的一阶运动模型,实现静态图像到动态图像的转换.

junyanz/pytorch-CycleGAN-and-pix2pix 图像到图像的转换

## 网络爬虫 下载
soimort/you-get youtube下载

shengqiangzhang/examples-of-web-crawlers python爬虫例子

itgoyo/Aria2  突破百度云限速合集

PanDownloadServer/Server 百度云PanDownload的个人维护版本

## 神经网络结构搜索 Neural Architecture Search

huawei-noah/CARS 华为提出基于进化算法和权值共享的神经网络结构搜索

microsoft/nni 用于自动化机器学习生命周期的开源AutoML工具包，包括功能工程，神经体系结构搜索，模型压缩和超参数调整。

awslabs/autogluon 用于深度学习的AutoML工具包 https://autogluon.mxnet.io

researchmm/CDARTS 循环可微架构搜索

xiaomi-automl/FairDARTS 消除差异化架构搜索中的不公平优势

## 虚拟化
jesseduffield/lazydocker docker 简单终端 UI

KubeOperator/KubeOperator 

rancher/k3s Lightweight Kubernetes. 5 less than k8s. https://k3s.io

docker-slim/docker-slim 请勿更改Docker容器映像中的任何内容并将其最小化30倍

silenceshell/docker_mirror 发现国内加速的docker源。

AliyunContainerService/GPUshare-scheduler-extender GPU Sharing Scheduler Extender in Kubernetes Kubernetes 中的 GPU 共享调度程序扩展器

# 分布式机器学习

horovod/horovod 分布式训练框架

dask/dask  提供大规模性能 高级并行性

Qihoo360/XLearning

sql-machine-learning/elasticdl

kubeflow/kubeflow

alibaba/euler

Angel-ML/angel

ray-project/ray 快速简单的框架，用于构建和运行分布式应用程序。

Alink 基于Flink的通用算法平台

kakaobrain/torchgpipe pytorch的可扩展管道并行性库，可以有效地训练大型的，消耗内存的模型。

tensorflow/mesh 简化模型并行化 Mesh TensorFlow: Model Parallelism Made Easier

microsoft/DeepSpeed 一个深度学习优化库，它使分布式训练变得容易，高效和有效。

sql-machine-learning/elasticdl Kubernetes原生的深度学习框架。ElasticDL是一个基于TensorFlow 2.0的Kubernetes原生深度学习框架，支持容错和弹性调度。

uber/fiber 简化AI的分布式计算 该项目是实验性的，API不稳定。

petuum/adaptdl 资源自适应深度学习（DL）培训和调度框架。AdaptDL的目标是使分布式DL在动态资源环境（如共享集群和云）中变得轻松高效。

learning-at-home/hivemind 一个用于在互联网上训练大型神经网络的库

petuum/adaptdl 一个能动态调整并行度的深度神经网络训练框架。它支持多租户集群管理，可以平衡模型训练等待及完成时间，能够提高资源利用率。

huggingface/accelerate 一个简单的 API，将与多 GPU 、 TPU 、 fp16 相关的样板代码抽离了出来，保持其余代码不变。PyTorch 用户无须使用不便控制和调整的抽象类或编写、维护样板代码，就可以直接上手多 GPU 或 TPU。Accelerate 支持的集成包括：CPU 单 GPU 单一节点多 GPU 多节点多 GPU TPU 带有本地 AMP 的 FP16（路线图上的顶点）


# 图数据库 图算法

Tencent/plato

dgraph-io/dgraph

hugegraph/hugegraph

vtraag/leidenalg

erikbern/ann-benchmarks 最邻近搜索

vesoft-inc/nebula 分布式、可扩展、闪电般的图形数据库

milvus-io/milvus 大规模特征向量的最快相似度搜索引擎 基于Faiss、Annoy等开源库，并针对性做了定制，支持结构化查询、多模查询等业界比较急需的功能；Milvus支持cpu、gpu、arm等多种类型的处理器；同时使用mysql存储元数据，并且在共享存储的支持下，Milvus可以支持分布式部署。

vearch/vearch 用于嵌入式向量高效相似性搜索的分布式系统

dgraph-io/dgraph The Only Native GraphQL Database With A Graph Backend.

vesoft-inc/nebula 开放源代码图数据库，能够托管具有数十亿个顶点（节点）和数万亿条边（具有几毫秒的延迟）的超大规模图。

shobrook/communities 社区检测算法和可视化工具库

jm199504/Financial-Knowledge-Graphs 小型金融知识图谱构建流程


# 大数据
Qihoo360/Quicksql 体系结构图可帮助您更轻松地访问 Quicksql

seata/seata 简单可扩展的自主事务体系结构

apache/incubator-shardingsphere 分布式数据库中间件生态圈

Tencent/wwsearch wwsearch是企业微信后台自研的全文检索引擎

apache/airflow 一个以编程方式编写，安排和监视工作流的平台

apache/shardingsphere Distributed database middleware 分布式数据库中间件

opencurve/curve 网易自主设计研发的高性能、高可用、高可靠分布式存储系统，具有良好扩展性。

ClickHouse/ClickHouse 一个开源列式数据库系统，允许实时生成数据分析报告。

canonical/dqlite 可嵌入、复制和故障耐受性 SQL 引擎。

apache/iceberg 新兴的数据湖框架之一，开创性的抽象出”表格式“（table format）这一中间层，既独立于上层的计算引擎（如Spark和Flink）和查询引擎（如Hive和Presto），也和下层的文件格式（如Parquet，ORC和Avro）相互解耦。同时，Iceberg还提供了许多额外的能力：ACID事务；时间旅行（time travel），以访问之前版本的数据；完备的自定义类型、分区方式和操作的抽象；列和分区方式可以进化，而且进化对用户无感，即无需重新组织或变更数据文件；隐式分区，使SQL不用针对分区方式特殊优化；面向云存储的优化等；

apache/hudi 基于Hadoop兼容的存储，提供了以下流处理原语。Update/Delete Record、Change Streams 将HDFS和Hudi结合起来，提供对流处理的支持能力。如：支持记录级别的更新、删除，以及获取基于HDFS之上的Change Streams。哪些数据发生了变更。

TurboWay/bigdata_analyse  大数据分析项目，包括1 亿条淘宝用户行为分析 、1000 万条淘宝用户行为 、300 万条《野蛮时代》的玩家 、130 万条深圳通刷卡、10 万条厦门招聘、7000 条租房、6000 条倒闭企业、COVID-19 疫情、7 万条天猫订单数据

avinassh/fast-sqlite3-inserts 1分钟插入10亿行数据,写脚本请使用Rust

baidu/BaikalDB 分布式HTAP数据库 支持PB级结构数据的顺序和随机实时读取/写入。 B与MySQL协议兼容，并且支持MySQL样式SQL方言，通过该方言，用户可以将其数据存储从MySQL无缝迁移到BaikalDB。

# 硬件

mit-pdos/xv6-riscv xv6 是丹尼斯 · 里奇和肯 · 汤普森的Unix 版本 6 （v6）的重新实现。 xv6 松散地遵循 v6 的结构和风格，但使用 ANSI C 为现代 RISC-V 多处理器实施。

plctlab/PLCT-Open-Reports  PLCT实验室的公开演讲，或者决定公开的组内报告  RISCV LLVM 等。

plctlab/writing-your-first-riscv-simulator  《从零开始的RISC-V模拟器开发》配套的PPT和教学资料 

cccriscv/mini-riscv-os 从零开始为 RISC-V 构建最小的多任务操作系统内核

plctlab/riscv-operating-system-mooc  《从头写一个RISC-V OS》课程配套的资源 

darklife/darkriscv 一个晚上从零开始在 Verilog 实现 RISC-V cpu

ultraembedded/riscv  RISC-V CPU 核心 （RV32IM）

OpenXiangShan/XiangShan 开源高性能 RISC-V 处理器

SI-RISCV/e200_opensource 蜂鸟 E203 开源处理器核心

Lichee-Pi/Tang_FPGA_Examples LicheeTang FPGA例程

ultraembedded/biriscv 32 位超级RISC-V Cpu

liuqidev/8-bits-RISC-CPU-Verilog 基于有限状态机的8位RISC（精简指令集）CPU（中央处理器）简单结构和Verilog实现。 

Lichee-Pi/Tang_E203_Mini  LicheeTang 蜂鸟E203 Core 

litex-hub/linux-on-litex-vexriscv 使用 VexRiscv CPU 运行 Linux，这是一种 32 位的 Linux 功能 RISC-V CPU。

riscv2os/riscv2os 手把手帶你學習 RISC-V 到可以自製 RISC-V 處理器上的作業系統的電子書。

nf9/police_light Lichee Tang板实现警灯

sgmarz/osblog 在 Rust 中编写 RISC-V 操作系统

danjulio/lepton/tree/master/ESP32 基于 ESP32 的热像仪（Lepton 3.5）。

## 其他
modichirag/flowpm TensorFlow中的粒子网格模拟N体宇宙学模拟

huihut/interview C/C++ 技术面试基础知识总结

barry-ran/QtScrcpy Android实时显示控制软件

bennettfeely/bennett ztext 易于实现的3D网页排版。适用于每种字体。

DaveJarvis/keenwrite 基于Java的桌面Markdown编辑器，具有实时预览，字符串插值和公式

vinayak-mehta/present 基于终端的演示工具，具有颜色和效果。

willmcgugan/rich 一个终端内富文本和美化的python库。

occlum/occlum 蚂蚁集团自研的开源可信执行环境（Trusted Execution Environments，简称 TEE） OS 系统 Occlum ,大幅降低 SGX 应用开发的门槛.机密计算（Confidential Computing）使得数据始终保持加密和强隔离状态，从而确保了用户数据的安全和隐私。

matazure/mtensor 一个tensor计算库, 支持cuda的延迟计算

fofapro/vulfocus 一个漏洞集成平台，将漏洞环境 docker 镜像，放入即可使用，开箱即用。

crazycodeboy/awesome-flutter-cn  一个很棒的Flutter学习资源，官方教程，插件，工具，文章，App，视频教程等的资源列表 

xujiyou/zhihu-flutter  Flutter 高仿知乎 UI，非常漂亮，也非常流畅。 

nisrulz/flutter-examples 给初露头角的flutter开发人员的简单基本的应用程序示例。

microsoft/playwright-python 针对 Python 语言的纯自动化工具，它可以通过单个API自动执行 Chromium，Firefox 和 WebKit 浏览器，连代码都不用写，就能实现自动化功能。

hoffstadt/DearPyGui 一个针对Python的快速而强大的图形用户界面工具包，具有最小的依赖性

emeryberger/scalene 适用于Python的高性能，高精度CPU和内存分析器.用于Python脚本的CPU和内存分析器，能够正确处理多线程代码，还能区分Python代码和本机代码的运行时间。

raysan5/raylib 一个简单易用的视频游戏编程库 

rwv/chinese-dos-games 中文 DOS 游戏

nondanee/UnblockNeteaseMusic 解锁网易云音乐客户端变灰歌曲

fffaraz/awesome-cpp A curated list of awesome C++ (or C) frameworks, libraries, resources, and shiny things. Inspired by awesome-... stuff.

Genymobile/scrcpy 通过USB（或通过TCP / IP）连接的Android设备的显示和控制

openstf/minitouch 最小的Android多点触控事件生成器。

gozfree/gear-lib 一组通用的Ｃ基础库，用POSIX C实现，目标是为了跨平台兼容。适用于物联网，嵌入式，以及网络服务开发等场景。

tangtangcoding/C-C- C语言电子书与视频资料分享

fluttercandies/wechat_flutter Flutter版本微信，一个优秀的Flutter即时通讯IM开源库

CoderMikeHe/flutter_wechat 利用 Flutter 来高仿微信(WeChat) 7.0.0+ App

youxinLu/flutter_mall 一款Flutter开源在线商城应用程序

ducafecat/flutter_learn_news  flutter实战学习-新闻客户端 

freestyletime/FlutterNews  用Flutter写的新闻类小项目 

linyacool/WebServer C++11编写的Web服务器

sanic-org/sanic 异步 Python 3.7+ web 框架 

davidbrochart/nbterm 让你在终端中查看、编辑、执行Jupyter笔记。

SocialSisterYi/bilibili-API-collect  哔哩哔哩-API收集整理

0x727/ShuiZe_0x727 水泽-信息收集自动化工具 只需要输入根域名即可全方位收集相关资产，并检测漏洞。

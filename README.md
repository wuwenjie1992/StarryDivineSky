# mystars
个人的github stars，主要是机器学习、深度学习、NLP、GNN、大数据等内容。

### 机器学习
MNN : 一个轻量级的深度神经网络推理引擎
Tencent/TNN 移动端高性能、轻量级推理框架，同时拥有跨平台、高性能、模型压缩、代码裁剪等众多突出优势。
apache/incubator-tvm 用于深度学习系统的编译器堆栈。它旨在缩小以生产力为中心的深度学习框架与以性能和效率为重点的硬件后端之间的差距。TVM与深度学习框架一起使用，以提供对不同后端的端到端编译。
Weights and Biases : 使用 W&B 组织和分析机器学习实验
ALiPy 一个基于Python实现的主动学习工具包
Nevergrad 无梯度优化平台
combo 用于机器学习模型组合的 Python 工具箱

google/trax 代码更清晰的神经网络代码库

rushter/MLAlgorithms 机器学习算法
MLEveryday/100-Days-Of-ML-Code
csuldw/MachineLearning
luwill/machine-learning-code-writing
CDCS 中国数据竞赛优胜解集锦
mlpack/mlpack 快速、灵活的机器学习库
tensorflow/ranking  排名学习在TensorFlow中

scikit-survival 生存分析

leibinghe/GAAL-based-outlier-detection 基于盖尔的异常检测
yzhao062/pyod 异常检测

lavender28/Credit-Card-Score 申请信用评分卡模型

modin-project/modin 通过更改一行代码来扩展加速pandas 
vaexio/vaex 适用于Python的核外DataFrame，以每秒十亿行的速度可视化和探索大型表格数据
cupy/cupy NumPy-like API accelerated with CUDA https://cupy.chainer.org
pythran 将 Python 代码转成 C++ 代码执行 一个 AOT (Ahead-Of-Time - 预先编译) 编译器，大幅度提升性能。
RAPIDS Open GPU Data Science http://rapids.ai
cudf cuDF - GPU DataFrame Library
cuml cuML - RAPIDS Machine Learning Library
cugraph cuGraph - RAPIDS Graph Analytics Library
cusignal cuSignal - RAPIDS Signal Processing Library

AtsushiSakai/PythonRobotics 机器人算法
cuML GPU 机器学习算法
SQLFlow 连接 SQL 引擎的桥接，与机器学习工具包连接
FeatureLabs/featuretools
esa/pagmo2 大规模并行优化的科学库 生物启发式算法和进化算法
kingfengji/gcForest Deep forest

interpretml/interpret 训练可解释的机器学习模型和解释黑匣子系统
alexmojaki/heartrate 调试 Python程序执行的简单实时可视化
google-research/mixmatch 集成了自洽正则化的超强半监督学习 MixMatch 

bojone/keras_recompute 通过重计算来节省显存，参考论文《Training Deep Nets with Sublinear Memory Cost》。




### 参数优化
hyperopt/hyperopt 分布式超参数优化
optuna/optuna 超参数优化框架https://optuna.org
WillKoehrsen/hyperparameter-optimization 超参数优化
HDI-Project/BTB Bayesian Tuning and Bandits，auto-tuning系统的一个简单、可扩展的后端系统。
scikit-optimize/scikit-optimize 一个简单高效的库，可最大限度地减少（非常）昂贵且嘈杂的黑盒功能。它实现了几种基于顺序模型优化的方法。
automl/SMAC3 基于序列模型的算法配置 优化任意算法的参数
CMA-ES/pycma 基于CMA-ES 协方差矩阵的自适应策略 的Py实现和一些相关的数值优化工具。
SheffieldML/GPyOpt 使用GPy进行高斯过程优化
pytorch/botorch PyTorch中的贝叶斯优化
JasperSnoek/spearmint 机器学习算法的实用贝叶斯优化
facebookresearch/nevergrad 用于执行无梯度优化的Python工具箱
Yelp/MOE 用于现实世界的指标优化的全局黑匣子优化引擎。
fmfn/BayesianOptimization 具有高斯过程的全局优化的Python实现。
dragonfly/dragonfly  用于可扩展的贝叶斯优化
音调：可伸缩超参数调整
ray-project/ray Tune可伸缩超参数调整
keras-team/keras-tuner keras的超参数调整




### 梯度提升
AugBoost 梯度提升
DeepGBM 梯度提升
CatBoost 基于梯度提升对决策树的机器学习方法
GBDT-PL/GBDT-PL 梯度提升
mesalock-linux/gbdt-rs 梯度提升
Xtra-Computing/thundergbm 梯度提升
dmlc/xgboost 梯度提升


### 分布式机器学习
horovod/horovod 分布式训练框架
dask/dask  提供大规模性能 高级并行性
Qihoo360/XLearning
sql-machine-learning/elasticdl
kubeflow/kubeflow
alibaba/euler
Angel-ML/angel
ray-project/ray 快速简单的框架，用于构建和运行分布式应用程序。
Alink 基于Flink的通用算法平台
kakaobrain/torchgpipe pytorch的可扩展的管道并行性库，可以有效地训练大型的，消耗内存的模型。
tensorflow/mesh 简化模型并行化 Mesh TensorFlow: Model Parallelism Made Easier
microsoft/DeepSpeed 一个深度学习优化库，它使分布式训练变得容易，高效和有效。
sql-machine-learning/elasticdl Kubernetes原生的深度学习框架。ElasticDL是一个基于TensorFlow 2.0的Kubernetes原生深度学习框架，支持容错和弹性调度。
uber/fiber 简化AI的分布式计算 该项目是实验性的，API不稳定。


## 图数据库 图算法
Tencent/plato
dgraph-io/dgraph
hugegraph/hugegraph
vtraag/leidenalg
erikbern/ann-benchmarks 最邻近搜索
vesoft-inc/nebula 分布式、可扩展、闪电般的图形数据库
milvus-io/milvus 大规模特征向量的最快相似度搜索引擎
vearch/vearch 用于嵌入式向量高效相似性搜索的分布式系统

## 大数据
Qihoo360/Quicksql 体系结构图可帮助您更轻松地访问 Quicksql
seata/seata 简单可扩展的自主事务体系结构
apache/incubator-shardingsphere 分布式数据库中间件生态圈
Tencent/wwsearch wwsearch是企业微信后台自研的全文检索引擎
apache/airflow 一个以编程方式编写，安排和监视工作流的平台
apache/shardingsphere Distributed database middleware 分布式数据库中间件
opencurve/curve 网易自主设计研发的高性能、高可用、高可靠分布式存储系统，具有非常良好的扩展性。



## 图神经网络GNN
* 图机器学习库
 * stellargraph/stellargraph 星际图机器学习库
 * Deep Graph Library (DGL)
 * alibaba/euler 分布式图深度学习框架。
 * facebookresearch/PyTorch-BigGraph 从大型图形结构化数据生成嵌入
 * rusty1s/pytorch_geometric 用于PyTorch的深度图学习扩展库
 * shenweichen/GraphNeuralNetwork 图神经网络的实现和实验，gcn\graphsage\gat等。
 * THUDM/cogdl 图形表示学习工具包，实现的模型，非GNN基线:如Deepwalk，LINE，NetMF，GNN基线:如GCN，GAT，GraphSAGE
 * imsheridan/CogDL-TensorFlow 图表示学习工具包，使研究人员和开发人员可以轻松地训练和比较基线或自定义模型，以进行节点分类，链接预测和其他图任务。它提供了许多流行模型的实现，包括：非GNN基准，例如Deepwalk，LINE，NetMF；GNN基准，例如GCN，GAT，GraphSAGE。
 * CrawlScript/tf_geometric 高效友好的图神经网络库 节点分类:图卷积网络（GCN）、多头图注意力网络（GAT），链接预测：平均池、SAGPooling，图分类：图形自动编码器（GAE）
 * alibaba/graph-learn 旨在简化图神经网络应用的框架。从实际生产案例中提取解决方案。已在推荐，反作弊和知识图系统上得到应用和验证。
 * BUPT-GAMMA/OpenHINE 异构信息网络嵌入（OpenHINE）的开源工具包。实现的模型包括：DHNE，HAN，HeGAN，HERec，HIN2vec，Metapath2vec，MetaGraph2vec，RHINE。

ASTGCN 基于注意的时空图卷积网络，用于交通流量预测 
danielzuegner/robust-gcn 
benedekrozemberczki/ClusterGCN
zhiyongc/Graph_Convolutional_LSTM
Jiakui/awesome-gcn 该存储库用于收集GCN，GAT（图形关注）相关资源。
tkipf/gae
peter14121/intentgc-models 意图gc模型
PetarV-/GAT Graph Attention Networks 
GraphVite 高速、大规模图嵌入
williamleif/GraphSAGE
GeniePath-pytorch
xiangwang1223/neural_graph_collaborative_filtering

tkipf/gae Graph Auto-Encoders
tkipf/gcn
microsoft/gated-graph-neural-network-samples
deepmind/graph_nets 在Tensorflow中构建图网
woojeongjin/dynamic-KG 嵌入动态知识图
awslabs/dgl-ke 高性能，易于使用且可扩展的软件包，用于学习大规模知识图嵌入。


leoribeiro/struc2vec
shenweichen/GraphEmbedding
thunlp/OpenKE
Jhy1993/HAN 异构图注意力网络

thunlp/OpenNE
tkipf/keras-gcn
aditya-grover/node2vec
thunlp/Fast-TransX
thunlp/TensorFlow-TransX
Wentao-Xu/SEEK 知识图谱嵌入框架
thunlp/KB2E
RLgraph 用于深度强化学习的模块化计算图
hwwang55/RippleNet 将知识图谱作为额外信息，融入到CTR/Top-K推荐
THUDM/cogdl 用于图形表示学习的广泛研究平台
klicperajo/ppnp 预测然后传播：图形神经网络满足个性化PageRank
inyeoplee77/SAGPool Self-Attention Graph Pooling torch自我注意力图池化
thunlp/ERNIE 用知识图谱增强 BERT 的预训练效果 


Malllabiisc/CompGCN 针对多关系有向图的图神经网络
graphdml-uiuc-jlu/geom-gcn 几何图卷积网络 将节点映射为连续空间的一个向量（graph embedding），在隐空间查找邻居并进行聚合。
EstelleHuang666/gnn_hierarchical_pooling Hierarchical Graph Representation Learning 构建了一个多层次的、节点可微分的聚合 GNN 网络。在每一层中，完成信息的抽取，并将当前的图聚合为一个更粗粒度的图，供下一层使用。

limaosen0/Variational-Graph-Auto-Encoders 可变图自动编码器 链接预测
animutomo/gcmc Graph Convolution Matrix Completion 解决推荐系统中 矩阵补全 matrix completion 问题，并引入 side information（节点的额外信息）提升预测效果。

Ruiqi-Hu/ARGA 对抗正则化图自动编码器 Adversarially Regularized Graph Autoencoder，可用于图卷积的链路预测。进化路线 GAE -> VGAE -> ARGA 
brxx122/HeterSumGraph 用于提取文档摘要的异构图神经网络
chuxuzhang/KDD2019_HetGNN KDD2019论文中HetGNN的代码：异构图神经网络 
TAMU-VITA/L2-GCN GCN高效分层训练框架
safe-graph/DGFraud 基于深度图的工具箱，用于欺诈检测
safe-graph/graph-fraud-detection-papers 基于图的欺诈检测论文和资源
xiangwang1223/neural_graph_collaborative_filtering 神经图协同过滤（NGCF）是一种基于图神经网络的新推荐框架，通过执行嵌入传播，在用户项二部图中以高阶连通性的形式对协同信号进行显式编码。
Googlebaba/KDD2019-MEIRec 基于异质图神经网络的用户意图推荐
mori97/JKNet-dgl 跳跃知识网络的dgl实现




## 强化学习 Reinforcement Learning
ray-project/ray 快速简单的框架，用于构建和运行分布式应用程序。
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


## 神经网络结构搜索 Neural Architecture Search
huawei-noah/CARS 华为提出基于进化算法和权值共享的神经网络结构搜索
microsoft/nni 用于自动化机器学习生命周期的开源AutoML工具包，包括功能工程，神经体系结构搜索，模型压缩和超参数调整。
awslabs/autogluon 用于深度学习的AutoML工具包 https://autogluon.mxnet.io
researchmm/CDARTS 循环可微架构搜索
xiaomi-automl/FairDARTS 消除差异化架构搜索中的不公平优势





## NLP自然语言处理
### 文本分类 + Attention机制
  * tcxdgit/cnn_multilabel_classification 基于TextCNN和Attention的多标签分类
  * ilivans/tf-rnn-attention Tensorflow实现文本分类任务的关注机制。

### 文本摘要/指针生成网络
  * abisee/pointer-generator 使用指针生成器网络进行汇总
  * AIKevin/Pointer_Generator_Summarizer 指针生成器网络：具有关注，指向和覆盖机制的Seq2Seq，用于抽象性摘要。 tensorflow 2.0
  * kjc6723/seq2seq_Pointer_Generator_Summarizer 中文会话中生成摘要总结的项目  tensorflow 2.0
  * steph1793/Pointer_Transformer_Generator tensorflow 2.0
  * magic282/NeuSum 通过共同学习评分和选择句子进行神经文本摘要
  * dmmiller612/bert-extractive-summarizer BERT易于使用的提取文本摘要

### 文本相似度
  *  UKPLab/sentence-transformers 句子转换器：使用BERT / RoBERTa / XLM-RoBERTa＆Co.和PyTorch的多语言句子嵌入
  *   terrifyzhao/text_matching 常用文本匹配模型tf版本，数据集为QA_corpus，持续更新中 

huseinzol05/NLP-Models-Tensorflow 抽象总结 聊天机器人依赖解析器 实体标记 提取摘要 发电机 语言检测 神经机器翻译 光学字符识别 POS标签 问题答案 句子对 语音转文字 拼写校正 小队问题答案 抽干 文字扩充 文字分类 文字相似度 文字转语音 主题生成器 主题建模 无监督提取摘要 矢量化器 老少少的声码器 可视化 注意Attention

brightmart/albert_zh 使用TensorFlow 进行自我监督学习语言表示的Lite Bert的实现 预训练的汉语模型
bert4keras 更清晰、更轻量级的keras版bert

huawei-noah/Pretrained-Language-Model 华为诺亚方舟实验室开发的预训练语言模型及其相关优化技术NEZHA是一种经过预训练的中文语言模型，可以在多项中文NLP任务上实现最先进的性能TinyBERT是一种压缩的BERT模型，推理时可缩小7.5倍，加快9.4倍
google-research/ALBERT 用于语言表达自我监督学习的Lite BERT
bojone/attention  Attention机制的实现 tensorflow/keras
ouyanghuiyu/chineseocr_lite 超轻量级中文ocr
thu-coai/CrossWOZ 大规模的中文跨域任务导向对话数据集.它包含5个领域的6K对话会话和102K语音，包括酒店，餐厅，景点，地铁和出租车。
ShomyLiu/Neu-Review-Rec Pytorch的基于评论文本的深度推荐系统模型库。DeepCoNN(WSDM'17)、D-Attn(RecSys'17)、ANR(CIKM'18)、NARRE(WWW'18)、MPCN(KDD'18)、TARMF(WWW'18)、CARL(TOIS'19)、CARP(SIGIR'19)、DAML(KDD'19)
ShannonAI/service-streamer 服务流媒体 BERT服务,每秒处理1400个句子的BERT服务.
squareRoot3/Target-Guided-Conversation 目标指导的开放域对话,在开放域的聊天中目标引导.
weizhepei/CasRel 一种用于关系三重提取的新颖级联二进制标记框架.
qiufengyuyi/sequence_tagging 使用bilstm-crf，bert等方法进行序列标记任务
microsoft/unilm UniLM-NLP及更高版本的统一语言模型预训练
thunlp/ERNIE 用知识图谱增强 BERT 的预训练效果 

 * 1) 对于抽取并编码的知识信息，研究者首先识别文本中的命名实体，然后将这些提到的实体与知识图谱中的实体进行匹配。研究者并不直接使用 KG 中基于图的事实，相反他们通过知识嵌入算法（例如 TransE）编码 KG 的图结构，并将多信息实体嵌入作为 ERNIE 的输入。基于文本和知识图谱的对齐，ERNIE 将知识模块的实体表征整合到语义模块的隐藏层中。
 * 2) 与 BERT 类似，研究者采用了带 Mask 的语言模型，以及预测下一句文本作为预训练目标。除此之外，为了更好地融合文本和知识特征，研究者设计了一种新型预训练目标，即随机 Mask 掉一些对齐了输入文本的命名实体，并要求模型从知识图谱中选择合适的实体以完成对齐。





### Transformer优化
  * laiguokun/Funnel-Transformer Transformer优化，一种新的自我注意模型，可以将隐藏状态的序列逐渐压缩为较短的状态，从而降低了计算成本。
  * mit-han-lab/hardware-aware-transformers 用于高效自然语言处理的硬件感知型Transformers.实现高达3倍的加速和3.7倍的较小模型尺寸，而不会降低性能。
  * mit-han-lab/lite-transformer 具有长距离短距离注意的Lite transformer
  * DeBERTa：注意力分散的增强解码的BERT，使用两种新颖的技术改进了BERT和RoBERTa模型，显着提高了模型预训练的效率和下游任务的性能。
  * allenai/longformer 用于长文档的类似BERT的模型



## 推荐系统

shenweichen/DeepCTR
ChenglongChen/tensorflow-DeepFM
cheungdaven/DeepRec
lyst/lightfm
oywtece/dstn
shenweichen/DSIN
facebookresearch/dlrm 深度学习推荐模型（DLRM）的实现
vze92/DMR Deep Match to Rank Model for Personalized Click-Through Rate Prediction DMR：Matching和Ranking相结合的点击率预估模型 
kang205/SASRec 源于Transformer的基于自注意力的序列推荐模型
shichence/AutoInt 使用Multi-Head self-Attention进行自动的特征提取
xiangwang1223/neural_graph_collaborative_filtering 神经图协同过滤 
UIC-Paper/MIMN 点击率预测的长序列用户行为建模的实践
motefly/DeepGBM 结合了 GBDT 和神经网络的优点，在有效保留在线更新能力的同时，还能充分利用类别特征和数值特征。DeepGBM 由两大块组成，CatNN 主要侧重于利用 Embedding 技术将高维稀疏特征转为低维稠密特征，而 GBDT2NN 则利用树模型筛选出的特征作为神经网络的输入，并通过逼近树结构来进行知识蒸馏。
shenweichen/DeepMatch 用于推荐和广告的深度匹配模型库。训练模型和导出用户和项目的表示向量非常容易，可用于ANN搜索。
LeeeeoLiu/ESRM-KG 关键词生成的基于电商会话的推荐模型
zhuchenxv/AutoFIS 自动特征交互选择的点击率预测模型
pangolulu/exact-k-recommendation 解决推荐中带约束的Top-K优化问题


## 金融股票 时间序列

QUANTAXIS/QUANTAXIS 量化金融策略框架
ricequant/rqalpha 从数据获取、算法交易、回测引擎，实盘模拟，实盘交易到数据分析，为程序化交易者提供了全套解决方案

cedricporter/funcat 将同花顺、通达信、文华财经麦语言等的公式写法移植到了 Python 
georgezouq/awesome-deep-reinforcement-learning-in-finance 金融市场上使用的那些AI（RL/DL/SL/进化/遗传算法）的集合
wangshub/RL-Stock 如何用深度强化学习自动炒股
tensortrade-org/tensortrade 一个开源强化学习框架，用于训练，评估和部署强大的交易程序。



## 虚拟化
jesseduffield/lazydocker docker 简单终端 UI
KubeOperator/KubeOperator 
rancher/k3s Lightweight Kubernetes. 5 less than k8s. https://k3s.io
docker-slim/docker-slim 请勿更改Docker容器映像中的任何内容并将其最小化30倍

## 网络爬虫 下载
soimort/you-get youtube下载
shengqiangzhang/examples-of-web-crawlers python爬虫例子
itgoyo/Aria2  突破百度云限速合集


## 其他
modichirag/flowpm TensorFlow中的粒子网格模拟  N 体宇宙学模拟
huihut/interview C/C++ 技术面试基础知识总结
barry-ran/QtScrcpy Android实时显示控制软件 
minivision-ai/photo2cartoon 人像卡通化探索项目 
JasonWei512/Tacotron-2-Chinese 中文语音合成
TensorSpeech/TensorflowTTS Tensorflow 2的实时最新语音合成
deezer/spleeter 人声分离模型
cfzd/Ultra-Fast-Lane-Detection 论文“ 超快速结构感知深度车道检测 ”的实现



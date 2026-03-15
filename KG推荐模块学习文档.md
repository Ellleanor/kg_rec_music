# 知识图谱音乐推荐系统（MKR模型）学习文档

> 本文档面向完全零基础的初学者，从最基本的概念出发，逐步讲解本项目中知识图谱推荐模块（MKR）的完整实现原理。

---

## 目录

1. [项目整体架构](#一项目整体架构)
2. [什么是知识图谱？](#二什么是知识图谱)
3. [什么是推荐系统？](#三什么是推荐系统)
4. [MKR 模型的核心思想](#四mkr-模型的核心思想)
5. [数据是什么样的？](#五数据是什么样的)
6. [数据预处理流程](#六数据预处理流程)
7. [模型的核心组件](#七模型的核心组件)
8. [模型训练过程](#八模型训练过程)
9. [生成推荐结果](#九生成推荐结果)
10. [完整端到端流程图](#十完整端到端流程图)
11. [各文件功能速查表](#十一各文件功能速查表)

---

## 一、项目整体架构

本项目是一个**完整的音乐推荐系统**，由四个子模块组成：

```
kg_rec_music/
├── MKR/              ★ 核心：知识图谱推荐模型（本文档重点讲解）
├── MusicRec/         Django 后端服务（提供 API 接口）
├── MusicRec-Vue/     Vue 前端页面（用户看到的界面）
└── NetCloudMusic/    Scrapy 爬虫（从网易云音乐爬取数据）
```

**整个系统的工作方式**：
1. 爬虫从网易云音乐爬取歌曲、歌手、用户收听数据，存入 MongoDB
2. MKR 模型读取这些数据，训练一个推荐算法
3. 训练好的模型为每个用户预测他可能喜欢的歌曲
4. 推荐结果存入 MySQL 数据库
5. Django 后端查询数据库，Vue 前端展示推荐结果给用户

---

## 二、什么是知识图谱？

### 用大白话解释

知识图谱就是一张由"**实体**"和"**关系**"组成的网络图。

把现实世界中的事物用节点（实体）表示，把事物之间的关系用边（关系）表示，就构成了知识图谱。

### 本项目中的知识图谱

在音乐领域，知识图谱长这样：

```
《稻香》 ──[演唱者]──▶ 周杰伦
《稻香》 ──[收录于]──▶ 《魔杰座》专辑
《稻香》 ──[风格]──▶ 流行
周杰伦   ──[国籍]──▶ 台湾
```

每一条这样的"头实体 → 关系 → 尾实体"记录叫做一个**三元组**（Triple）。

知识图谱文件 `kg_final.txt` 就是由成千上万条这样的三元组组成的，只不过全部用数字索引表示：

```
0    0    51204    # 歌曲0 → 关系0（演唱者） → 实体51204（某歌手）
0    1    0        # 歌曲0 → 关系1（专辑）   → 实体0（某专辑）
0    2    51205    # 歌曲0 → 关系2（风格）   → 实体51205（某风格标签）
```

### 知识图谱有什么用？

知识图谱让计算机理解了事物之间的语义关系。比如：

> 你喜欢周杰伦的《稻香》→ 知识图谱告诉我们《七里香》也是周杰伦演唱的 → 所以你可能也喜欢《七里香》

这种推理能力正是知识图谱推荐比普通推荐更强的原因。

---

## 三、什么是推荐系统？

### 最简单的理解

推荐系统的核心问题是：**预测用户 U 对物品 I 的喜好程度**，得分越高，就推荐给他。

### 传统推荐的问题

最常见的传统方法叫**协同过滤**，原理是"找到和你品味相似的人，把他喜欢的推给你"。

但它有个致命缺点：**冷启动问题** —— 如果一首歌没人听过，系统完全不知道该推给谁。

### 知识图谱推荐的改进

引入知识图谱后，即使一首新歌没有收听记录，系统也能通过"这首歌是周杰伦唱的，而你喜欢周杰伦"这样的语义推理来产生推荐。

---

## 四、MKR 模型的核心思想

MKR 全称 **Multi-task learning for Knowledge graph enhanced Recommendation**，直译是"基于知识图谱的多任务学习推荐"。

### 4.1 什么是 Embedding（向量化）？

计算机无法直接处理"周杰伦"这个名字，需要把它变成一串数字（向量）。

例如，用 4 维向量表示：

```
周杰伦  → [0.2, -0.5, 0.8, 0.1]
流行音乐 → [0.7, 0.3, -0.2, 0.6]
《稻香》 → [0.3, -0.4, 0.7, 0.2]
```

这串数字叫做 **Embedding（嵌入向量）**，维度越高、信息越丰富，本项目中使用 4 维（`--dim=4`）。

相似的事物，其向量在数学空间中的距离也应该更近。

### 4.2 MKR 同时解决两个任务

MKR 的创新点在于用**同一个模型同时训练两个任务**：

```
任务一（RS）：推荐系统任务
    输入：用户 + 歌曲
    输出：用户喜欢这首歌的概率（0~1之间）

任务二（KGE）：知识图谱嵌入任务
    输入：头实体 + 关系
    输出：预测尾实体是什么
    例如：给定"《稻香》+ 演唱者"，预测出"周杰伦"
```

两个任务共享中间的表示，互相促进学习。

### 4.3 模型的三层结构

```
输入层：user_id / item_id / head_id / relation_id / tail_id
    ↓
低层（Low Layers）：CrossCompressUnit（交叉压缩单元）★核心创新
    ↓                ↓
高层-RS部分      高层-KGE部分
（计算推荐得分）  （预测尾实体）
```

---

## 五、数据是什么样的？

### 5.1 用户行为数据 `ratings_final.txt`

这是训练推荐系统（RS任务）用的数据，每行表示一条用户与歌曲的交互记录：

```
用户索引  歌曲索引  标签
0        1        1    ← 用户0 听过 歌曲1（喜欢，正样本）
0        2        1    ← 用户0 听过 歌曲2（喜欢，正样本）
0        999      0    ← 用户0 没有 听过 歌曲999（负样本，随机采样）
```

**正样本**：用户实际收听过的歌曲，标签为 1  
**负样本**：用户从未收听的随机歌曲，标签为 0

### 5.2 知识图谱数据 `kg_final.txt`

这是训练知识图谱嵌入（KGE任务）用的数据，每行是一个三元组：

```
头实体索引  关系索引  尾实体索引
0          0        51204    ← (歌曲0, 演唱者, 歌手51204)
0          1        0        ← (歌曲0, 专辑, 专辑0)
1          0        51204    ← (歌曲1, 演唱者, 歌手51204) ← 说明歌曲0和歌曲1是同一歌手
```

### 5.3 数据规模（本项目实际数据）

本项目数据共 **235,420 条知识图谱三元组**，涵盖：
- 音乐与歌手的关系
- 音乐与专辑的关系  
- 音乐与风格标签的关系

---

## 六、数据预处理流程

### 6.1 原始数据来源

```
MongoDB（爬虫爬取的网易云数据）
├── PlaylistInfo  歌单信息
├── SongDetail    歌曲详情（歌手、专辑、风格标签等）
└── UserInfo      用户信息（收听记录）
```

### 6.2 构建知识图谱 `data2kg.py`

这个脚本从 MongoDB 读取歌曲数据，按以下规则生成三元组：

| 关系类型 | 含义 | 示例 |
|---------|------|------|
| `music.track_id.artist_id` | 歌曲→歌手 | 《稻香》→周杰伦 |
| `music.track_id.album_id`  | 歌曲→专辑 | 《稻香》→《魔杰座》|
| `music.track_id.<风格名>`  | 歌曲→风格 | 《稻香》→流行 |

输出文件：
- `kg.txt`：原始三元组（字符串格式）
- `item_index2entity_id.txt`：歌曲原始ID → 内部实体索引的映射表

### 6.3 数字化处理 `preprocess.py`

将字符串 ID 全部转换为数字索引，并生成以下文件：

```
ratings_final.txt   ← 用户行为数据（含负采样）
kg_final.txt        ← 数字化知识图谱三元组
entity_music.json   ← 实体索引 → 原始歌曲ID（用于反查）
index_user_id.json  ← 用户索引 → 原始用户ID（用于反查）
```

---

## 七、模型的核心组件

MKR 模型代码位于 `MKR/model.py` 和 `MKR/layers.py`。

### 7.1 Embedding 层（查找表）

模型内部维护四张"查找表"，每个实体/用户/关系都对应表中的一行向量：

```python
# 四张 Embedding 矩阵
user_emb_matrix     形状: [用户总数,   4]  ← 每个用户一行4维向量
item_emb_matrix     形状: [歌曲总数,   4]  ← 每首歌一行4维向量
entity_emb_matrix   形状: [实体总数,   4]  ← 每个实体一行4维向量
relation_emb_matrix 形状: [关系总数,   4]  ← 每种关系一行4维向量
```

**查表操作**：
```python
# 输入 user_id = 0, 从矩阵中取出第0行
user_embedding = user_emb_matrix[user_id]   # → [0.2, -0.5, 0.8, 0.1]
item_embedding = item_emb_matrix[item_id]   # → [0.3, -0.4, 0.7, 0.2]
```

这些向量的初始值是随机的，训练过程中不断调整，使其越来越能代表实体的语义信息。

注意：**歌曲（item）和实体（entity）是同一事物的两种视角**——从推荐系统角度叫"物品（item）"，从知识图谱角度叫"实体（entity）"，它们共享一套参数，这正是知识图谱能帮助推荐的关键！

### 7.2 低层：CrossCompressUnit（交叉压缩单元）

这是 MKR 的 **核心创新**，也是最难理解的部分。

#### 为什么需要它？

歌曲在推荐系统中的 `item_embedding` 和在知识图谱中的 `entity_embedding` 需要互相传递信息：
- KG 任务学到了"《稻香》是周杰伦唱的"这个知识
- 这个知识应该传递给 RS 任务，帮助推荐

CrossCompressUnit 就是实现这种双向信息传递的桥梁。

#### 数学原理（逐步拆解）

**第一步：外积（Outer Product）**

把两个向量 `v`（item embedding）和 `e`（entity embedding）相乘，得到一个矩阵：

```
v = [v1, v2, v3, v4]  （形状: [4]）
e = [e1, e2, e3, e4]  （形状: [4]）

         e1   e2   e3   e4
v1  → [v1*e1, v1*e2, v1*e3, v1*e4]
v2  → [v2*e1, v2*e2, v2*e3, v2*e4]
v3  → [v3*e1, v3*e2, v3*e3, v3*e4]
v4  → [v4*e1, v4*e2, v4*e3, v4*e4]

结果: 4×4 的矩阵，捕获了 v 和 e 每个维度之间的交叉信息
```

**第二步：加权压缩（Weighted Compression）**

用4个权重向量，把 4×4 矩阵压缩回两个 4 维向量：

```python
# 更新 item embedding（v的方向）
v_output = C矩阵 × W_vv  +  C矩阵转置 × W_ev  +  bias_v
#          ↑ item→item     ↑ entity→item

# 更新 entity embedding（e的方向）  
e_output = C矩阵 × W_ve  +  C矩阵转置 × W_ee  +  bias_e
#          ↑ item→entity   ↑ entity→entity
```

直觉理解：
- `W_vv`：item 信息对 item 自身的影响
- `W_ev`：entity 信息对 item 的影响（**KG 知识流向 RS**）
- `W_ve`：item 信息对 entity 的影响（**RS 信息流向 KG**）
- `W_ee`：entity 信息对 entity 自身的影响

#### 代码实现（`layers.py`）

```python
class CrossCompressUnit(Layer):
    def __init__(self, dim, name=None):
        # 四个权重矩阵，每个形状 [dim, 1]
        self.weight_vv = tf.get_variable(name='weight_vv', shape=(dim, 1))
        self.weight_ev = tf.get_variable(name='weight_ev', shape=(dim, 1))
        self.weight_ve = tf.get_variable(name='weight_ve', shape=(dim, 1))
        self.weight_ee = tf.get_variable(name='weight_ee', shape=(dim, 1))
        self.bias_v = tf.get_variable(name='bias_v', shape=dim)
        self.bias_e = tf.get_variable(name='bias_e', shape=dim)

    def _call(self, inputs):
        v, e = inputs  # 两个向量，形状均为 [batch_size, dim]

        # 外积：把向量变成矩阵 [batch_size, dim, dim]
        v = tf.expand_dims(v, dim=2)   # [batch_size, dim, 1]
        e = tf.expand_dims(e, dim=1)   # [batch_size, 1, dim]
        c_matrix = tf.matmul(v, e)     # [batch_size, dim, dim]
        c_matrix_T = tf.transpose(c_matrix, perm=[0, 2, 1])  # 转置

        # 展开并加权压缩
        c_matrix   = tf.reshape(c_matrix,   [-1, dim])  # [batch_size*dim, dim]
        c_matrix_T = tf.reshape(c_matrix_T, [-1, dim])

        # 输出两个更新后的向量
        v_output = tf.reshape(
            tf.matmul(c_matrix, self.weight_vv) + tf.matmul(c_matrix_T, self.weight_ev),
            [-1, dim]) + self.bias_v
        e_output = tf.reshape(
            tf.matmul(c_matrix, self.weight_ve) + tf.matmul(c_matrix_T, self.weight_ee),
            [-1, dim]) + self.bias_e

        return v_output, e_output
```

### 7.3 高层 RS：计算推荐得分

经过低层 CrossCompressUnit 后，user embedding 和 item embedding 都已经融合了知识图谱的信息。

高层 RS 用**内积**（点积）计算用户和歌曲的匹配程度：

```python
# 内积：两个向量对应元素相乘再求和
scores = sum(user_embedding * item_embedding)

# 例如：
user_embedding = [0.2, -0.5, 0.8, 0.1]
item_embedding = [0.3, -0.4, 0.7, 0.2]
score = 0.2×0.3 + (-0.5)×(-0.4) + 0.8×0.7 + 0.1×0.2
      = 0.06 + 0.20 + 0.56 + 0.02 = 0.84  ← 得分较高，说明可能喜欢

# 再经过 Sigmoid 函数映射到 [0,1] 概率范围
probability = sigmoid(0.84) ≈ 0.70  ← 70% 概率用户喜欢这首歌
```

### 7.4 高层 KGE：预测尾实体

KGE 部分把"头实体向量"和"关系向量"拼接，通过 MLP（多层感知机）预测尾实体应该长什么样：

```python
# 拼接 head 和 relation 向量
head_relation_concat = concat([head_embedding, relation_embedding])  # 形状: [8]

# 经过 MLP 层，预测尾实体向量
tail_pred = MLP(head_relation_concat)  # 形状: [4]

# 和真实尾实体向量做内积，得分越高说明预测越准
score_kge = sigmoid( sum(tail_embedding * tail_pred) )
```

---

## 八、模型训练过程

### 8.1 训练参数配置（`main.py`）

```python
--n_epochs    20      # 训练轮数（跑完全部数据20遍）
--dim         4       # Embedding 维度（每个向量4个数字）
--L           1       # 低层 CrossCompressUnit 的数量
--H           2       # 高层 MLP 的层数
--batch_size  128     # 每次训练128条数据
--l2_weight   1e-6    # L2正则化权重（防止过拟合）
--lr_rs       1e-3    # RS任务学习率（步长：0.001）
--lr_kge      2e-4    # KGE任务学习率（步长：0.0002）
--kge_interval 4      # 每训练4轮RS，才训练1轮KGE
```

### 8.2 损失函数

**RS任务损失**（交叉熵损失）：

```
目标：模型预测的概率，要尽量接近真实标签（1或0）

Loss_RS = 交叉熵(预测概率, 真实标签) + L2正则化
         = -[y * log(p) + (1-y) * log(1-p)] + λ||参数||²

例如：真实标签 y=1（用户喜欢），预测概率 p=0.7
      Loss = -[1*log(0.7) + 0*log(0.3)] ≈ 0.357  ← 还需要继续优化

若 p=0.95，Loss = -log(0.95) ≈ 0.051  ← 很好了
```

**KGE任务损失**（最大化得分）：

```
目标：让模型对正确的三元组打出高分

Loss_KGE = -score_kge + L2正则化
         = -sigmoid(head+relation 预测 tail 的得分) + λ||参数||²
```

### 8.3 一个完整的训练轮次（`train.py`）

```
for epoch in range(20):
    ┌─────────────────────────────────────────────────────┐
    │ 第1步：RS训练（每轮必做）                              │
    │   打乱 ratings_final.txt 的顺序                      │
    │   按 batch_size=128 分批：                           │
    │     喂入：user_id, item_id, label (0或1)             │
    │     计算：交叉熵损失                                   │
    │     更新：user/item/CCU 的参数                        │
    └─────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────┐
    │ 第2步：KGE训练（每4轮才做1次）                        │
    │   打乱 kg_final.txt 的顺序                           │
    │   按 batch_size=128 分批：                           │
    │     喂入：head_id, relation_id, tail_id              │
    │     计算：KGE损失                                     │
    │     更新：entity/relation/CCU 的参数                  │
    └─────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────┐
    │ 第3步：评估（在验证集/测试集上）                       │
    │   打印：AUC（越接近1越好）和 ACC（准确率）              │
    │   例如：epoch 19  train auc: 0.88  eval auc: 0.73   │
    └─────────────────────────────────────────────────────┘

训练结束后，保存：
  vocab/user_emb_matrix.txt     ← 用户向量矩阵（用于增量更新）
  vocab/item_emb_matrix.txt     ← 歌曲向量矩阵
  vocab/entity_emb_matrix.txt   ← 实体向量矩阵
  vocab/relation_emb_matrix.txt ← 关系向量矩阵
  restore/xxx/mkr.ckpt          ← TF checkpoint（完整模型）
  result/xxx/saved_model.pb     ← TF Serving 格式（用于生产部署）
```

### 8.4 增量学习（支持新用户/新歌）

本项目还支持**不重新训练、只添加新用户/新歌曲**的增量学习方式：

```python
# 如果存在已训练好的 vocab/*.txt，则加载旧向量
old_user_emb = np.loadtxt('vocab/user_emb_matrix.txt')

# 新增的用户，用随机向量初始化
new_rows = np.random.normal(size=[新用户数, dim])
user_emb = np.vstack([old_user_emb, new_rows])

# 旧用户的向量（已有知识）被保留，接着继续训练
```

---

## 九、生成推荐结果

### 9.1 推理脚本 `predict_test.py`

训练完成后，使用推理脚本为每个用户生成 Top-N 推荐列表：

```python
# 核心逻辑（伪代码）
for user_id in all_users:
    scores = {}
    for item_id in all_songs:
        # 排除用户已经听过的歌曲
        if item_id not in user_history[user_id]:
            # 模型打分
            score = model.predict(user_id, item_id)
            scores[item_id] = score
    
    # 取得分最高的15首歌
    top15 = sorted(scores, key=lambda x: scores[x], reverse=True)[:15]
    
    # 将 内部索引 → 网易云原始歌曲ID
    top15_original_ids = [entity_music_map[idx] for idx in top15]
    
    # 写入文件
    write to user_song_prefer.txt
```

### 9.2 推荐结果格式

推荐结果写入 `user_song_prefer.txt`，格式如下：

```
用户原始ID  歌曲原始ID1  歌曲原始ID2  ...  歌曲原始ID15
849116     409931558   470736982   ...
```

### 9.3 写入数据库 `RecToMysql.py`

推荐结果从文件写入 MySQL 数据库，供 Django 后端查询：

```python
# MySQL 表结构
class UserSongRec(models.Model):
    user    = models.CharField()   # 用户ID
    related = models.CharField()   # 推荐歌曲ID
    sim     = models.FloatField()  # MKR预测得分（越高越推荐）
```

### 9.4 后端查询逻辑

用户登录后，Django 按得分倒序取 Top-N：

```python
# 查 MySQL 获取推荐列表
recommendations = UserSongRec.objects.filter(user=user_id).order_by('-sim')[:15]

# 再查 MongoDB 获取歌曲详细信息（封面图、歌手名等）
for rec in recommendations:
    song_detail = SongDetail.find_one({'id': int(rec.related)})
```

---

## 十、完整端到端流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                       数据采集层                                  │
│  NetCloudMusic (Scrapy爬虫)  →  MongoDB                         │
│  PlaylistInfo / SongDetail / UserInfo                           │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                       KG 构建层                                   │
│  data2kg.py                                                     │
│  → item_index2entity_id.txt（歌曲ID映射）                        │
│  → user_artists.dat（用户行为原始数据）                           │
│  → kg.txt（原始知识图谱三元组）                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                       数据预处理层                                │
│  preprocess.py                                                  │
│  → ratings_final.txt（用户-物品-标签）                           │
│  → kg_final.txt（数字化KG三元组）                                │
│  → entity_music.json / index_user_id.json（反查映射表）          │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                       模型训练层                                  │
│  main.py → train.py → MKR模型（model.py + layers.py）           │
│  ┌─────────────────────────────────────────────┐               │
│  │  Embedding层（用户/歌曲/实体/关系 向量）       │               │
│  │       ↓                                     │               │
│  │  低层 CrossCompressUnit                      │               │
│  │  （item ↔ entity 双向信息交换）               │               │
│  │       ↓              ↓                      │               │
│  │  高层RS（内积）    高层KGE（MLP）              │               │
│  │  推荐得分         预测尾实体                  │               │
│  └─────────────────────────────────────────────┘               │
│  输出：vocab/*.txt + mkr.ckpt + saved_model.pb                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                       推理预测层                                  │
│  predict_test.py                                                │
│  对每个用户，对所有歌曲打分，取 Top-15                            │
│  → user_song_prefer.txt                                        │
│  RecToMysql.py → MySQL (UserSongRec 表)                        │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                       服务展示层                                  │
│  Django (MusicRec) 查 MySQL → 再查 MongoDB → 返回 JSON API      │
│  Vue (MusicRec-Vue) 展示推荐歌曲给用户                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 十一、各文件功能速查表

### MKR/ 目录

| 文件 | 功能 | 何时运行 |
|------|------|---------|
| `main.py` | 程序入口，解析参数，启动训练 | 训练时：`python main.py` |
| `model.py` | MKR 模型类（网络结构定义） | 被 train.py 调用 |
| `layers.py` | Dense层 和 CrossCompressUnit层 | 被 model.py 调用 |
| `train.py` | 训练循环、评估、模型保存 | 被 main.py 调用 |
| `data_loader.py` | 加载 ratings_final.txt 和 kg_final.txt | 被 main.py 调用 |
| `preprocess.py` | 原始数据 → 训练格式的预处理 | 数据准备阶段 |
| `data2kg.py` | 从 MongoDB 构建知识图谱文件 | 数据准备阶段 |
| `predict_test.py` | 批量推理，生成用户推荐列表 | 训练完成后 |
| `kkk.py` | 另一版本的推理脚本 | 训练完成后 |

### MKR/data/music/ 目录

| 文件 | 内容 | 格式 |
|------|------|------|
| `ratings_final.txt` | 用户-歌曲-标签 | `user_idx \t item_idx \t label` |
| `kg_final.txt` | 数字化KG三元组 | `head \t relation \t tail` |
| `kg.txt` | 原始KG三元组 | 字符串格式 |
| `user_artists.dat` | 原始用户收听数据 | `userId \t songId \t score` |
| `item_index2entity_id.txt` | 歌曲ID → 实体索引 | `song_id \t entity_idx` |
| `entity_music.json` | 实体索引 → 原始歌曲ID | JSON字典 |
| `index_user_id.json` | 用户索引 → 原始用户ID | JSON字典 |

### MKR/model/music/ 目录

| 目录/文件 | 内容 |
|---------|------|
| `vocab/*.txt` | 训练好的 Embedding 矩阵（用于增量学习） |
| `restore/时间戳/mkr.ckpt.*` | TensorFlow checkpoint（完整模型权重） |
| `result/时间戳/saved_model.pb` | TF Serving 格式（用于生产部署） |

---

## 附录：关键概念汇总

| 术语 | 解释 |
|------|------|
| **Embedding** | 将ID/名称等离散量转换为连续向量（一串数字）的技术 |
| **三元组** | 知识图谱的基本单位：（头实体, 关系, 尾实体） |
| **RS** | Recommendation System，推荐系统任务 |
| **KGE** | Knowledge Graph Embedding，知识图谱嵌入任务 |
| **CrossCompressUnit** | MKR核心创新，实现RS和KGE之间的信息双向交换 |
| **内积/点积** | 两向量对应元素相乘再求和，度量两向量的相似程度 |
| **外积** | 两向量生成矩阵，捕获两向量每个维度之间的交叉信息 |
| **AUC** | 模型评估指标，值域[0,1]，越接近1模型越准 |
| **Sigmoid** | 将任意实数映射到(0,1)区间的函数，输出可解释为概率 |
| **L2正则化** | 防止模型参数过大、过拟合的技术，给损失函数加惩罚项 |
| **批量训练(Batch)** | 每次只用一小批数据（如128条）更新参数，加快训练 |
| **正样本/负样本** | 正样本=用户真实喜欢的歌(标签1)，负样本=随机未听歌(标签0) |
| **冷启动** | 新用户/新物品没有历史记录时，推荐系统无法工作的问题 |
| **多任务学习** | 同时训练多个相关任务，任务间互相促进，比单独训练效果更好 |

---

*文档生成时间：2026年3月15日*  
*对应项目：kg_rec_music（基于MKR的音乐知识图谱推荐系统）*

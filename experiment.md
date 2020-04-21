

# AstroClassification

### 2.6 21:17   caojh更新

- pkl文件中存储数据dataframe的二进制形式，其中将answer列改为数字，对应关系为{"star": 0, "galaxy": 1, "qso": 2}，并删除了id列。现在读完整的训练集只需要两分钟。

- 模型用Random Forest，n_estimators=10

  先pca->750维，用medium来train，用full来test：

  - 训练时间25.3s
  - Train score: {'accuracy': 0.9975236085557581,'f1': 0.9934531625317428}
  -  Test score: {'accuracy': 0.9524987923273988, 'f1': 0.8382630805902855}

  不用pca，用medium来train，用full来test：

  - 训练时间63.2s
  - Train score: {'accuracy': 0.9983345395568655, 'f1': 0.994567191973572}
  - Test score: {'accuracy': 0.9676099592443196, 'f1': 0.8919070389039198}

  先pca->750维，用full前80%来train，后20%来test：

  - 训练时间141.3s
  -  Train score: {'accuracy': 0.9986375516912889, 'f1': 0.9960921028010502}
  - Test score: {'accuracy': 0.961093090579331, 'f1': 0.866631163406836}

  不用pca，用full前80%来train，后20%来test：

  - 训练时间401.7s
  - Train score: {'accuracy': 0.9986855098717554, 'f1': 0.9956542270730346}
  - Test score: {'accuracy': 0.9661243067908339, 'f1': 0.8811825648128023}

### 2.7 11.07 caojh更新：

- validation set重新存储为pkl文件。

- 模型用Random Forest，n_estimators=10

  先pca->750维，用full来train，用validation来test：

  - Train time: 181.53187441825867
  - Train score: {'accuracy': 0.9987286738970069, 'f1': 0.9961105309060985}
  - Test score: {'accuracy': 0.9622030804096021, 'f1': 0.8686221219322383}

  直接训练：

  -  Train time: 538.9405434131622
  -  Train score: {'accuracy': 0.9987757600489696, 'f1': 0.9959931420938423}
  - Test score: {'accuracy': 0.9674332717810978, 'f1': 0.885273528861346}

  先标准化，pca->750维，再训练：

  - Train time: 461.1937208175659
  - Train score: {'accuracy': 0.9986937952659234, 'f1': 0.9959827209113853}
  - Test score: {'accuracy': 0.9430974064126239, 'f1': 0.8032911077217327} ??? 反而下降了？

  先标准化，再训练：

  - Train time: 1144.545776605606
  
  - Train score: {'accuracy': 0.9991384978122378, 'f1': 0.9968619888883445}
  
  - Test score: {'accuracy': 0.9717978848413631, 'f1': **0.891567251312332**}
  
  - **目前最佳**，存储在 model_rf_891567.pkl
  
  - |              | precision | recall | f1-score | support |
    | :----------: | :-------: | :----: | :------: | :-----: |
    |      0       |   0.99    |  0.99  |   0.99   | 160000  |
    |      1       |   0.93    |  0.88  |   0.90   |  23376  |
    |      2       |   0.80    |  0.76  |   0.78   |  7248   |
    |   accuracy   |           |        |   0.97   | 190624  |
    |  macro avg   |   0.90    |  0.88  |   0.89   | 190624  |
    | weighted avg |   0.97    |  0.97  |   0.97   | 190624  |
  
  - 应该问题在类别不平衡上
  



### 2.7 18.00 lixy更新:

- 直接调用classify.rf中的train，test函数，用full进行训练并在validation上预测，主要区别在于提取特征:
  - 进行流量归一化，不进行平滑处理
  - 将波长2600的光谱分为50等份，并统计每段的均值、标准差和峰值、偏值，由每个样本统计得到4个长度均为50的向量，并拼接为200维向量
  - 调用train，test函数采用RandomForest进行训练、预测
  
- 训练效果：
  - Train time: 143.57545375823975
  
  - Train score: {'accuracy': 0.9991332660175753, 'f1': 0.9971259832719356}
  
  - Test score: {'accuracy': 0.9763408594930334, 'f1': 0.9209636155848386}
  
  - |              | precision | recall | f1-score | support |
    | :----------: | :-------: | :----: | :------: | :-----: |
    |      0       |   0.99    |  0.99  |   0.99   | 160000  |
    |      1       |   0.94    |  0.91  |   0.93   |  23376  |
    |      2       |   0.88    |  0.83  |   0.85   |  7248   |
    |   accuracy   |           |        |   0.98   | 190624  |
    |  macro avg   |   0.94    |  0.91  |   0.92   | 190624  |
    | weighted avg |   0.98    |  0.98  |   0.98   | 190624  |
  
- 补充：流量归一化之后采用平滑处理，最终macro average略有提升

  - Train time: 161.81601238250732

  - Train score: {'accuracy': 0.9991768643064297, 'f1': 0.9972064368148651}

  - Test score: {'accuracy': 0.9791893990263556, 'f1': **0.9299095851191229**}

  - 

    |              | precision | recall | f1-score | support |
    | :----------: | :-------: | :----: | :------: | :-----: |
    |      0       |   0.99    |  0.99  |   0.99   | 160000  |
    |      1       |   0.95    |  0.92  |   0.94   |  23376  |
    |      2       |   0.89    |  0.84  |   0.86   |  7248   |
    |   accuracy   |           |        |   0.98   | 190624  |
    |  macro avg   |   0.94    |  0.92  |   0.93   | 190624  |
    | weighted avg |   0.98    |  0.98  |   0.98   | 190624  |

   - 在上一步基础上，改变平滑处理的窗口大小，（之前默认窗口大小为7） 
  |                 |       5       |        7      |      9       |      11       |
  | :-------------: | :-----------: | :-----------: | :----------: | :-----------: |
  |  windows size   | 0.92895530592 | 0.92990958511 | 0.9285834443 | 0.92648703925 |



### 2.8 12:14 caojh更新

full前80%来train，后20%来validate，val_sets来test，模型用gbdt，参数为：

```python
boost_round = 50
early_stop_rounds = 10
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',  # multiclassova
    'num_class': 3,
    'num_iterations': 100,
    'learning_rate': 0.1,
    'num_leaves': 32,
    'device_type': 'cpu',
    'metric': 'multi_logloss',
    'num_threads': 40,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
```

- Train time: 12.179835081100464 sec

- Train score:
  macro f1: 0.9671390294944482

            		 precision  recall  f1-score   support
             0       1.00      1.00     1.00      385067
             1       0.97      0.97     0.97      56219
             2       0.93      0.94     0.93      17447
            
      accuracy                          0.99      458733
      macro avg       0.97      0.97    0.97      458733
      weighted avg    0.99      0.99    0.99      458733
- Test score:
  macro f1: 0.9574201666435972

             		 precision  recall  f1-score   support
             0       1.00      1.00     1.00      160000
             1       0.97      0.96     0.97      23376
             2       0.91      0.91     0.91      7248
      
      accuracy                          0.99      190624
      macro avg      0.96      0.96     0.96      190624
      weighted avg   0.99      0.99     0.99      190624
  

目前最佳：

```python
boost_round = 50
early_stop_rounds = 50

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',  # multiclassova
    'num_class': 3,
    'num_iterations': 10000,
    'learning_rate': 0.1,
    'num_leaves': 100,
    'device_type': 'cpu',
    'metric': 'multi_logloss',
    'num_threads': 40,
    'max_depth': 8,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
0.9697850134563225
```

- Train score:
  macro f1: 0.99971281606622
  ```
  precision    recall  f1-score   support
            0       1.00      1.00      1.00    385067
             1       1.00      1.00      1.00     56219
             2       1.00      1.00      1.00     17447
      accuracy                           1.00    458733
     macro avg       1.00      1.00      1.00    458733
  weighted avg       1.00      1.00      1.00    458733
  ```

- Test score:
  macro f1: 0.9697850134563225

  ```
   precision    recall  f1-score   support
              0       1.00      1.00      1.00    160000
             1       0.98      0.97      0.98     23376
             2       0.94      0.93      0.94      7248
      accuracy                           0.99    190624
     macro avg       0.97      0.97      0.97    190624
  weighted avg       0.99      0.99      0.99    190624      
  ```


### 2.8 16.11 lixy 更新

full前80%来train，后20%来validate，val_sets来test，模型用gbdt，沿用之前参数，主要不同在于特征提取方法中加入argmin, argmax，表征光谱中吸收峰和发射峰的位置

- Train score:
  macro f1: 0.9997498020618419
  ```
                  precision    recall  f1-score   support
               0       1.00      1.00      1.00    385067
               1       1.00      1.00      1.00     56219
               2       1.00      1.00      1.00     17447
        accuracy                           1.00    458733
       macro avg       1.00      1.00      1.00    458733
    weighted avg       1.00      1.00      1.00    458733
  ```

- Test score:
  ```
    macro f1: 0.9708099321394084
                  precision    recall  f1-score   support
               0       1.00      1.00      1.00    160000
               1       0.98      0.98      0.98     23376
               2       0.94      0.94      0.94      7248
        accuracy                           0.99    190624
       macro avg       0.97      0.97      0.97    190624
    weighted avg       0.99      0.99      0.99    190624
  ```
- Confusion Matrix:
  ```
    [[159659    190    151]
     [   237  22844    295]
     [   119    342   6787]]
  ```
- 补充，加入min, max两个值之后f1 score未有明显上升.

### 2.8 18:05 caojh更新

- 整合了预处理部分，在preprocess/integrate.py中，具体流程如下：

  <img src="fig\preprocess.jpg" style="zoom:30%;" />

- 将PCA 750维特征和区间400维特征拼接后，模型参数不变，训练效果得到提升，如下：

  ```
  Train score::
  macro f1: 0.9997238755662178
                precision    recall  f1-score   support
             0       1.00      1.00      1.00    385067
             1       1.00      1.00      1.00     56219
             2       1.00      1.00      1.00     17447
      accuracy                           1.00    458733
     macro avg       1.00      1.00      1.00    458733
  weighted avg       1.00      1.00      1.00    458733
  
  Test score:
  macro f1: 0.9712274753428779
                precision    recall  f1-score   support
             0       1.00      1.00      1.00    160000
             1       0.98      0.98      0.98     23376
             2       0.94      0.94      0.94      7248
      accuracy                           0.99    190624
     macro avg       0.97      0.97      0.97    190624
  weighted avg       0.99      0.99      0.99    190624
  
  [[159666    187    147]
   [   221  22866    289]
   [   129    334   6785]]
   
  Feature name list: ['pca', 'means', 'theta', 'skews', 'kurts', 'argmaxs', 'argmins', 'maxs', 'mins']
  Feature importance list: [0.39766245778269826, 0.07703518925963816, 0.06799613212438864, 0.10338158835152823, 0.07606822035679751, 0.05496307299914514, 0.08866684417786622, 0.05981193155541853, 0.0744145633925193]
  ```



### 2.9 11:36 caojh更新

- 重写了lgb_classifier.py，封装了load, train, test等函数。

- 由于预处理数据时间较长，因此把预处理后的数据存下来了，在运行python命令后加入参数 `--load-dataXY` 可以直接读取存好的预处理后的数据，否则将重新生成并存储。

- 由于我本地的pycharm配置有一点问题，import matplotlib会报错（但服务器端不会），因此暂时将与画图相关的部分注释掉了，如果要画训练时的loss变化图，将

  ```python
  plt.figure()
  ax = lgb.plot_metric(results)
  plt.savefig("lgb_metric.png")
  plt.show()
  ```

  注释回来。如果要画各feature的importance图，将

  ```python
  plot_feature_importance(gbm.feature_importance())
  ```

  及其函数定义注释回来。
  
  
### 2.11 15:49 caojh更新

加入了data augmention。

- 用完整数据集，三个类别扩充[0.01,1,1]，lgb参数不变，效果略有上升（之前换成sklearn的train_test_split后相同参数test macro f1为0.97297）：

  ```
  Test score:
  macro f1: 0.9746593453238144
                precision    recall  f1-score   support
             0       1.00      1.00      1.00    160000
             1       0.98      0.98      0.98     23376
             2       0.94      0.95      0.95      7248
      accuracy                           0.99    190624
     macro avg       0.97      0.98      0.97    190624
  weighted avg       0.99      0.99      0.99    190624
  Confusion matrix:
  [[159604    222    174]
   [   156  22970    250]
   [    76    292   6880]]
  ```




### 2.10 14.32 lixy更新

- 从所有数据中提取出qso, galaxy两类，训练和测试样本分别为`X_train, Y_train`，选取其中20%为验证集，并经过平滑处理后，沿用之间区间统计的特征工程方法，提取400维向量，并调用gbdt进行训练，沿用之前的参数，训练结果如下

- Train time: 13.520026206970215
  Train score:
  macro f1: 0.9998494264390219

  ```
                precision    recall  f1-score   support
             1       1.00      1.00      1.00     56301
             2       1.00      1.00      1.00     17381
      accuracy                           1.00     73682
     macro avg       1.00      1.00      1.00     73682
  weighted avg       1.00      1.00      1.00     73682
  ```

- Test score:
  macro f1: 0.9675190166350669

  ```
                precision    recall  f1-score   support
             1       0.98      0.99      0.98     23376
             2       0.96      0.94      0.95      7248
      accuracy                           0.98     30624
     macro avg       0.97      0.97      0.97     30624
  weighted avg       0.98      0.98      0.98     30624
  Confusion matrix: 
  [[23059   317]
   [  399  6849]]
  ```
- 主要发现：梯度上升决策树可能本身对样本不平衡不敏感，单独提取qso, galaxy两类难以提高macro-f1 score



### 2.10 20:10 caojh更新

- 把lgb的metric换成了macro-f1，early_stop_rounds改成100，其他参数不变，效果得到提升：

  ```
    Test score:
    macro f1: 0.9752608987601077
                  precision    recall  f1-score   support
               0       1.00      1.00      1.00    160000
               1       0.98      0.98      0.98     23376
               2       0.95      0.95      0.95      7248
        accuracy                           0.99    190624
       macro avg       0.98      0.97      0.98    190624
    weighted avg       0.99      0.99      0.99    190624
    Confusion matrix:
    [[159740    141    119]
     [   202  22914    260]
     [   117    279   6852]]
  ```

  

### 3.31 15:20 lixy更新

最近实验集中在搭建一维卷积神经网络和相应参数调整上，调整频率较高因此遗漏了实验结果的记录，在此做一次实验思路和结果的总结和回顾。

神经网络具体结构具体参考自天池比赛 [天文数据挖掘大赛——龙樱组开源方案](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.15.7e03681bqf5wdL&postId=5043)，并在其基础上进行一定修改，将dropout改为batch normalization，将最后一层全连接激活函数由tanh函数改为relu函数等。

- 一维卷积神经网络 + Dropout

  - 网络结构

    ```python
    def ConvNet_DropOut(input_shape=(2600, 1, 1), classes=3) -> Model:
        seed = 710
        X_input = Input(input_shape)
        X_list = []
        for step in range(3, 19, 2):
            Xi = X_input
            for _ in range(3):
                Xi = Conv2D(32, (step, 1), strides=(1, 1),
                            padding='same', activation='relu', 	
                            kernel_initializer=glorot_uniform(seed))(Xi)
                Xi = AveragePooling2D((3, 1), strides=(3, 1))(Xi)
            X_list.append(Xi)
        X = concatenate(X_list, axis=3)
        X = Flatten()(X)
        for nodes in [1024, 512, 256]:
            X = Dense(nodes, activation='relu', 
                      kernel_initializer=glorot_uniform(seed))(X)
            X = Dropout(0.8)(X)
        X = Dense(classes, activation='softmax', 
                  kernel_initializer=glorot_uniform(seed))(X)
        model = Model(inputs=X_input, outputs=X, name='ConvNet')
        return model
    
    ```

  - batch size = 256

    epoch=25时取到macro f1最大值：0.9816884312451949

  	```
      Test score:
      macro f1: 0.9811327474489104
                    precision    recall  f1-score   support
                 0       1.00      1.00      1.00    160000
                 1       0.99      0.98      0.98     23376
                 2       0.96      0.96      0.96      7248
          accuracy                           1.00    190624
         macro avg       0.98      0.98      0.98    190624
      weighted avg       1.00      1.00      1.00    190624
      Confusion matrix:
      [[159817    121     62]
       [   189  22982    205]
       [   118    192   6938]]
    ```
  
  该模型存储于`./model/checkpoint_correct_03-30-00-10.h5`中
  
  - batch size = 128

    epoch=29时取到macro f1最大值：0.9810167027934155

    ```
    Test score:
    macro f1: 0.9810167027934155
                  precision    recall  f1-score   support
               0       1.00      1.00      1.00    160000
               1       0.99      0.98      0.98     23376
               2       0.96      0.96      0.96      7248
        accuracy                           1.00    190624
       macro avg       0.98      0.98      0.98    190624
    weighted avg       1.00      1.00      1.00    190624
    Confusion matrix:
    [[159782    131     87]
     [   206  22942    228]
     [   102    161   6985]]
    ```
    该模型存储于`./model/checkpoint_correct_03-30-00-11.h5`中

  - epoch - f1 score曲线变化趋势如图：

    ![](fig/dropout_curve.png)

    

- 一维卷积神经网络 + Batch Normalization

  drop out机制对于过拟合有效，但对于batch_size等参数敏感，转而采用batch normalization。

  - 网络结构

    ```python
    def ConvNet_BN(input_shape=(2600, 1, 1), classes=3) -> Model:
        seed = 710
        X_input = Input(input_shape)
        X_list = []
        for step in range(3, 19, 2):
            Xi = X_input
            for _ in range(3):
                Xi = Conv2D(32, (step, 1), strides=(1, 1), use_bias=False,
                            padding='same', kernel_initializer=glorot_uniform(seed))(Xi)
                Xi = BatchNormalization()(Xi)
                Xi = Activation('relu')(Xi)
                Xi = AveragePooling2D((3, 1), strides=(3, 1))(Xi)
            X_list.append(Xi)
        X = concatenate(X_list, axis=3)
        X = Flatten()(X)
        for nodes in fc_layers:
            X = Dense(nodes, use_bias=False, kernel_initializer=glorot_uniform(seed))(X)
            X = BatchNormalization()(X)
            X = Activation('relu')(X)
        X = Dense(classes, use_bias=False, kernel_initializer=glorot_uniform(seed))(X)
        X = BatchNormalization()(X)
        X = Activation('softmax')(X)
        model = Model(inputs=X_input, outputs=X, name='ConvNet')
        return model
    ```

  - batch size = 256 + large fully-connected layers

    全连接层为`fc_layers = [1024, 512, 256]`，macro f1最大为0.9823131759657721

    ```
    Test score:
    macro f1: 0.9823131759657721
                  precision    recall  f1-score   support
               0       1.00      1.00      1.00    160000
               1       0.99      0.99      0.99     23376
               2       0.96      0.96      0.96      7248
        accuracy                           1.00    190624
       macro avg       0.98      0.98      0.98    190624
    weighted avg       1.00      1.00      1.00    190624
    Confusion matrix:
    [[159827    117     56]
     [   122  23052    202]
     [    86    206   6956]]
    ```

    由于log文件丢失，因此何时达到峰值与变化曲线信息丢失 :-<

    该模型保存在`./model/checkpoint_correct_03-30-11-30-32.h5`中

  - batch size = 128 + large fully-connected layers

    epoch=12时取到macro f1最大值：0.9827169677156951，全连接层为`[1024, 512, 256]`

    ```
    Test score:
    macro f1: 0.9827169677156951
                  precision    recall  f1-score   support
               0       1.00      1.00      1.00    160000
               1       0.99      0.99      0.99     23376
               2       0.97      0.96      0.96      7248
        accuracy                           1.00    190624
       macro avg       0.98      0.98      0.98    190624
    weighted avg       1.00      1.00      1.00    190624
    Confusion matrix:
    [[159831    110     59]
     [   141  23050    185]
   [    88    202   6958]]
    ```

    该模型保存在`./model/checkpoint_correct_03-30-15-30-41.h5`中

  - batch size = 256 + small fully-connected layers

    epoch=22时取到macro f1最大值：0.9825546143678788，全连接层为`[512, 256]`

    ```
    Test score:
    macro f1: 0.9825546143678788
                  precision    recall  f1-score   support
               0       1.00      1.00      1.00    160000
               1       0.99      0.99      0.99     23376
               2       0.96      0.96      0.96      7248
        accuracy                           1.00    190624
       macro avg       0.98      0.98      0.98    190624
    weighted avg       1.00      1.00      1.00    190624
    Confusion matrix:
  [[159818    112     70]
     [   130  23049    197]
     [    88    190   6970]]
    ```

    该模型保存在`./model/checkpoint_correct_03-30-15-31-55.h5`中

  - epoch - f1 score曲线示意图 

    ![](fig/bn_curve.png)
  
### 4.3 14.50 lixy更新

对参数组合进行探索以后发现效果逐渐收敛，尝试对网络结构进行微调，将卷积层数由3层改为6层，batch = 128,在原数据集与增强后数据集上均有提升，但值得一提的是验证集上表现最好的epoch并不出现在训练效果逐渐收敛的后期，可能存在epoch参数过拟合验证集的现象。

- 原数据集

  epoch=6时取到macro f1最大值：0.983868834820611

  ```
  Test score:
  macro f1: 0.983868834820611
                precision    recall  f1-score   support
             0       1.00      1.00      1.00    160000
             1       0.99      0.99      0.99     23376
             2       0.97      0.96      0.97      7248
      accuracy                           1.00    190624
     macro avg       0.98      0.98      0.98    190624
  weighted avg       1.00      1.00      1.00    190624
  Confusion matrix:
  [[159811    110     79]
   [   182  23040    154]
   [    97    161   6990]]
  ```

- aug3数据集

  epoch=14时取到macro f1最大值：0.9832982745367626

  ```
  Test score:
  macro f1: 0.9832982745367626
                precision    recall  f1-score   support
             0       1.00      1.00      1.00    160000
             1       0.99      0.99      0.99     23376
             2       0.97      0.96      0.96      7248
      accuracy                           1.00    190624
     macro avg       0.98      0.98      0.98    190624
  weighted avg       1.00      1.00      1.00    190624
  Confusion matrix:
  [[159805    120     75]
   [   144  23058    174]
   [    92    175   6981]]
  macro f1: 0.9832982745367626, highest score: 0.9832982745367626
  ```

- aug6数据集

  epoch=15时取到macro f1最大值：0.9835375033241983

  ```
  Test score:
  macro f1: 0.9835375033241983
                precision    recall  f1-score   support
             0       1.00      1.00      1.00    160000
             1       0.99      0.99      0.99     23376
             2       0.97      0.96      0.97      7248
      accuracy                           1.00    190624
     macro avg       0.98      0.98      0.98    190624
  weighted avg       1.00      1.00      1.00    190624
  Confusion matrix:
  [[159798    129     73]
   [   134  23080    162]
   [    79    191   6978]]
  macro f1: 0.9835375033241983, highest score: 0.9835375033241983
  ```

- epoch - macro f1曲线示意图

  ![](fig/6_conv_curve.png)

## Conclusion

### lixy

- 
这次比赛是初次涉足数据挖掘类比赛，因此最大的收获是了解比赛的标准流程与常用方法，并且在较为简单的神经网络上训练得到还不错的成绩；主要的问题在于对于神经网络的领域知识知之甚少，更别谈CV与NLP今年来层出不穷的新模型，由于经验的不足因此一直在浅层卷积+全连接层，较为简单的网络结构上进行训练并调参，效果上自然存在上限。

### caojh

- 交叉验证，完整数据集太大，应该在小数据集上实验

- 网络结构，BN效果好，卷积层数也可以多一点

- 数据预分析还不够，数据清洗和缺失值等都没有做

- 没有分析模型分错样本，根据分错的样本特征重新设计预处理，提取特征等操作
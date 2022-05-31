import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc,roc_curve
from sklearn.ensemble import RandomForestClassifier
from torch import ne
from BoostMain import *
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class RandomForest():
    def __init__(self, n_estimators=35, min_samples_split=2, min_gain=1,
                 max_depth=3, max_features=None):

        self.n_estimators = n_estimators #树的数量
        self.min_samples_split = min_samples_split #每棵树中最小的分割数，比如 min_samples_split = 2表示树切到还剩下两个数据集时就停止
        self.min_gain = min_gain   #每棵树切到小于min_gain后停止
        self.max_depth = max_depth  #每棵树的最大层数
        self.max_features = max_features #每棵树选用数据集中的最大的特征数

        self.trees = []
        self.tree_feature_id = []##决策树对应特征list
        # 建立森林(bulid forest)
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, min_samples_leaf=self.min_gain,
                                      max_depth=self.max_depth)
            self.trees.append(tree)
    '''
        训练，每棵树使用随机的数据集(bootstrap)和随机的特征
        随机数据已经拿到，随机特征先是涉及到随机特征个数问题，书上是推荐 log_2 (n_features)
        将处理过的数据存在sub_X中，进行fit生成决策树

    '''
    def fit(self, X, Y):
        # 训练，每棵树使用随机的数据集(bootstrap)和随机的特征
        # every tree use random data set(bootstrap) and random feature
        sub_sets = self.get_bootstrap_data(X, Y)
        n_features = X.shape[1]

        if self.max_features == None:
            # self.max_features = int(np.log2(n_features))
            self.max_features = n_features
        for i in range(self.n_estimators):
            # 生成随机的特征
            sub_X, sub_Y = sub_sets[i]
            idx = np.random.choice(n_features, self.max_features, replace=False) #不重复最好吧
            sub_X = sub_X[:, idx]
            self.trees[i].fit(sub_X, sub_Y)
            self.tree_feature_id.append(idx)
    '''
        使用每棵树对应的特征进行预测，对应特征存在feature_indices里
    '''
    def predict(self, X):
        y_preds = []
        for i in range(self.n_estimators):
            idx = self.tree_feature_id[i]
            sub_X = X[:, idx]
            y_pre = self.trees[i].predict(sub_X)
            y_preds.append(y_pre)
        y_preds = np.array(y_preds).T
        y_pred = []
        for y_p in y_preds:
            # np.bincount()可以统计每个索引出现的次数
            # np.argmax()可以返回数组中最大值的索引
            # cheak np.bincount() and np.argmax() in numpy Docs
            y_pred.append(np.bincount(y_p.astype('int')).argmax())
        return y_pred


    '''
        通过bootstrap的方式获得n_estimators组数据
        步骤：先合并原始数据XY
              随机打乱后，有放回的挑选m（numbers of samples）个样本  并将其拆回XY存到data_sets的list中返回
              重复n_estimators次  
        返回：数据list [[X_1,Y_1],[X_2,Y_2],...,[X_nestimators,Y_nestimators]] 
    '''
    def get_bootstrap_data(self, X, Y):

        m = X.shape[0] #行数
        Y = Y.reshape(m, 1)
        # 合并X和Y，方便bootstrap 
        X_Y = np.hstack((X, Y)) #np.vstack():在竖直方向上堆叠   /np.hstack():在水平方向上平铺
        np.random.shuffle(X_Y) #随机打乱
        data_sets = []
        for _ in range(self.n_estimators):
            idm = np.random.choice(m, m, replace=True) #在range(m)中,有重复的选取 m个数字
            bootstrap_X_Y = X_Y[idm, :]
            bootstrap_X = bootstrap_X_Y[:, :-1]
            bootstrap_Y = bootstrap_X_Y[:, -1:]
            data_sets.append([bootstrap_X, bootstrap_Y])
        return data_sets

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # X_train = np.loadtxt('HW6/adult_dataset/feature.txt')
    # y_train = np.loadtxt('HW6/adult_dataset/label.txt')
    # X_test = np.loadtxt('HW6/adult_dataset/test_f.txt')
    # y_test = np.loadtxt('HW6/adult_dataset/test_l.txt')

    X_train = np.loadtxt('HW6/adult_dataset/adult_train_feature.txt')
    y_train = np.loadtxt('HW6/adult_dataset/adult_train_label.txt')
    X_test = np.loadtxt('HW6/adult_dataset/adult_test_feature.txt')
    y_test = np.loadtxt('HW6/adult_dataset/adult_test_label.txt')


    # clf = RandomForestClassifier(n_estimators=10)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print(y_pred)
    clf = RandomForest()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    fpr, tpr, thresholds =  roc_curve(y_test, y_pred, pos_label=1) #正样本为1
    print("AUC=",auc(fpr,tpr))
#     n = []
#     aucs_RF = []
#     aucs_B = []
#     for i in range(10):
#         n_estimators = i*10 + 10
#         clf_RF = RandomForest(n_estimators=n_estimators)
#         clf_B = AdaBoostClassifierByhand(n_estimators=n_estimators)

#         clf_RF.fit(X_train, y_train)
#         y_pred_RF = clf_RF.predict(X_test)
#         fpr_RF, tpr_RF, thresholds_RF =  roc_curve(y_test, y_pred_RF, pos_label=1) #正样本为1
#         aucs_RF.append(auc(fpr_RF,tpr_RF))
        
#         clf_B.fit(X_train, y_train)
#         y_pred_B = clf_B.predict(X_test)
#         fpr_B, tpr_B, thresholds_B =  roc_curve(y_test, y_pred_B, pos_label=1) #正样本为1
#         aucs_B.append(auc(fpr_B,tpr_B))

#         n.append(n_estimators)
    
# plt.plot(n, aucs_RF, "r", marker='D', markersize=5, label="RF")
# plt.plot(n, aucs_B, "g", marker='D', markersize=5, label="AdaBoost")
# # 绘制坐标轴标签
# plt.xlabel("基学习器个数")
# plt.ylabel("AUC")
# plt.legend(['RF','AdaBoost'])
# plt.savefig("RF_VS_AdaBoost.jpg")
# plt.show()


#     n = []
#     aucs_RF = []
#     aucs_B = []
#     for i in range(10):
#         n_estimators = i*5 + 5
#         clf_RF = RandomForest(n_estimators=n_estimators)
#         clf_B = AdaBoostClassifierByhand(n_estimators=n_estimators)

#         clf_RF.fit(X_train, y_train)
#         y_pred_RF = clf_RF.predict(X_test)
#         fpr_RF, tpr_RF, thresholds_RF =  roc_curve(y_test, y_pred_RF, pos_label=1) #正样本为1
#         aucs_RF.append(auc(fpr_RF,tpr_RF))
        
#         clf_B.fit(X_train, y_train)
#         y_pred_B = clf_B.predict(X_test)
#         fpr_B, tpr_B, thresholds_B =  roc_curve(y_test, y_pred_B, pos_label=1) #正样本为1
#         aucs_B.append(auc(fpr_B,tpr_B))

#         n.append(n_estimators)
    
# plt.plot(n, aucs_RF, "r", marker='D', markersize=5, label="RF")
# plt.plot(n, aucs_B, "g", marker='D', markersize=5, label="AdaBoost")
# # 绘制坐标轴标签
# plt.xlabel("基学习器个数")
# plt.ylabel("AUC")
# plt.legend(['RF','AdaBoost'])
# plt.savefig("RF_VS_AdaBoost.jpg")
# # plt.savefig("auc7_RF.jpg")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sympy import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc,roc_curve
from sklearn.model_selection import cross_val_score

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class AdaBoostClassifierByhand:
    def __init__(self, n_estimators=25):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []
    '''
        用一层的决策树（树桩）作为基学习器
        返回存储该基学习器信息的字典
    '''

    def fit(self, X, y):
        weight = np.ones(len(X)) / len(X)
        nums = 0 #基学习器的个数
        rs = 1 #决策树的随机种子
        while nums < self.n_estimators:
            
            dtc = DecisionTreeClassifier(criterion="gini",random_state=rs)
            clf = dtc.fit(X, y)
            err = 1-accuracy_score(y,clf.predict(X))
            print("estimator {} err={}".format(nums+1,err))
            if err > 0.5:     #但是貌似break也不会改变什么
                print("best_err > 0.5,nums_estimators=",nums)
                rs += 2
                continue
            if err == 0:
                print("best_err == 0,nums_estimators={},finished".format(nums))
                break
            nums += 1
            rs += 2
            alpha = 1/2 * np.log((1-err)/err) #+1e-5是因为数太大会溢出？？
            y_pre = clf.predict(X)           
            y_pre = 2*(y_pre) - 1
            y_ = 2*y - 1
            weight = weight * np.exp(-alpha * y_pre * y_)            
            weight = weight / np.sum(weight)
            self.estimators.append(clf)
            self.alphas.append(alpha)
        if nums < self.n_estimators:
            self.n_estimators = nums
        print("基学习器的个数：",self.n_estimators,"个")
        return self

    
    def predict(self, X):
        y_pre = np.empty((len(X), self.n_estimators)) #二维数组，存n_estimators个基学习器的预测结果
        for i in range(self.n_estimators):
            clf = self.estimators[i]
            y_p = clf.predict(X)
            y_pre[:,i] = y_p
        y_pre = y_pre * np.array(self.alphas)
        return 1*(np.sum(y_pre, axis=1)>0)
    
    def score(self, X, y):
        y_pre = self.predict(X)
        return np.mean(y_pre == y)

if __name__ == '__main__':
    X_train = np.loadtxt('HW6/adult_dataset/adult_train_feature.txt')
    y_train = np.loadtxt('HW6/adult_dataset/adult_train_label.txt')
    X_test = np.loadtxt('HW6/adult_dataset/adult_test_feature.txt')
    y_test = np.loadtxt('HW6/adult_dataset/adult_test_label.txt')

    # X_train = np.loadtxt('HW6/adult_dataset/feature.txt')
    # y_train = np.loadtxt('HW6/adult_dataset/label.txt')
    # X_test = np.loadtxt('HW6/adult_dataset/test_f.txt')
    # y_test = np.loadtxt('HW6/adult_dataset/test_l.txt')
    print("*********************start************************")
    # from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifierByhand()
    # score = cross_val_score(clf,X_train,y_train,cv=5)
    # print(score)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    fpr, tpr, thresholds =  roc_curve(y_test, y_pred, pos_label=1) #正样本为1
    print("AUC=",auc(fpr,tpr))
    # print(clf.estimators)
    # mean = clf.score(X_test, y_test)
    # print(mean)
    # 0.986013986013986 
    # mean_ = AdaBoostClassifier().fit(X_train, y_train).score(X_test, y_test)
    # 0.965034965034965

    # print("meanByMyself = ",mean)
    # print("mean = ",mean_)
    # n = []
    # aucs = []
    # for i in range(10):
    #     n_estimators = i * 5 + 5
    #     clf = AdaBoostClassifierByhand(n_estimators=n_estimators)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     fpr, tpr, thresholds =  roc_curve(y_test, y_pred, pos_label=1) #正样本为1
    #     aucs.append(auc(fpr,tpr))
    #     n.append(n_estimators)
    

    # plt.plot(n, aucs, "g", marker='D', markersize=5, label="RF")
    # #绘制坐标轴标签
    # plt.xlabel("基学习器个数")
    # plt.ylabel("AUC")
    # plt.savefig("aucBoost.jpg")
    # plt.show()
    print("*********************end************************")
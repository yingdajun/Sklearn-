
# coding: utf-8

# In[1]:


# 练习各类算法


# In[2]:


import sklearn 


# In[4]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()
iris.data.shape, iris.target.shape


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)                           


# In[6]:


from sklearn import metrics
# print(metrics)


# In[8]:


# 准确率
import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3,9,9,8,5,8]
y_true = [0, 1, 2, 3,2,6,3,5,9]

accuracy_score(y_true, y_pred)


accuracy_score(y_true, y_pred, normalize=False)  # 类似海明距离，每个类别求准确后，再求微平均



# In[10]:


#准确率平均度
from sklearn import metrics
metrics.precision_score(y_true, y_pred, average='micro')  # 微平均，精确率


metrics.precision_score(y_true, y_pred, average='macro')  # 宏平均，精确率

metrics.precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro')  # 指定特定分类标签的精确率



# In[11]:


#召回率
metrics.recall_score(y_true, y_pred, average='micro')


metrics.recall_score(y_true, y_pred, average='macro')



# In[12]:


# F1
metrics.f1_score(y_true, y_pred, average='weighted')  

 


# In[13]:


# 混淆矩阵
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)


# In[15]:


# 分类报告：precision/recall/fi-score/均值/分类个数
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 2, 0]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))


# In[17]:


# kappa score
#kappa score是一个介于(-1, 1)之间的数. score>0.8意味着好的分类；0或更低意味着不好（实际是随机标签）
from sklearn.metrics import cohen_kappa_score
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
cohen_kappa_score(y_true, y_pred)


# # ROC值

# In[18]:


# 计算ROC值
import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y_true, y_scores)


# In[22]:


# ROC曲线
import numpy as np
from sklearn.metrics import roc_curve
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr,tpr,thresholds = roc_curve(y, scores, pos_label=2)
print(roc_curve(y, scores, pos_label=2))


# # 距离

# In[23]:


# 海明距离
from sklearn.metrics import hamming_loss
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
hamming_loss(y_true, y_pred)



# In[25]:


# Jaccard距离
import numpy as np
from sklearn.metrics import jaccard_similarity_score
y_pred = [0, 2, 1, 3,4]
y_true = [0, 1, 2, 3,4]
print(jaccard_similarity_score(y_true, y_pred))

print(jaccard_similarity_score(y_true, y_pred, normalize=False))



# # 回归

# In[27]:


# 可释方差值
from sklearn.metrics import explained_variance_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
explained_variance_score(y_true, y_pred)  


# In[28]:


# 平均绝对误差
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_absolute_error(y_true, y_pred)


# In[29]:


# 均方误差
from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)


# In[30]:


# 中值绝对误差
from sklearn.metrics import median_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
median_absolute_error(y_true, y_pred)


# In[31]:


# R方值，确定系数
from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
r2_score(y_true, y_pred)  


# # 合理绘图

# In[37]:


# 这个图很好用的
# 函数plot_confusion_matrix是绘制混淆矩阵的函数
#，CalculationResults则为只要给入y的预测值 + 实际值，以及分类的标签大致内容，就可以一次性输出：f1值，acc,recall以及报表


# In[33]:


get_ipython().magic('matplotlib inline')
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,recall_score,classification_report,confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def CalculationResults(val_y,y_val_pred,simple = False,                       target_names = ['class_-2_Not_mentioned','class_-1_Negative','class_0_Neutral','class_1_Positive']):
    # 计算检验
    F1_score = f1_score(val_y,y_val_pred, average='macro')
    if simple:
        return F1_score
    else:
        acc = accuracy_score(val_y,y_val_pred)
        recall_score_ = recall_score(val_y,y_val_pred, average='macro')
        confusion_matrix_ = confusion_matrix(val_y,y_val_pred)
        class_report = classification_report(val_y, y_val_pred, target_names=target_names)
        print('f1_score:',F1_score,'ACC_score:',acc,'recall:',recall_score_)
        print('\n----class report ---:\n',class_report)
        #print('----confusion matrix ---:\n',confusion_matrix_)

        # 画混淆矩阵
            # 画混淆矩阵图
        plt.figure()
        plot_confusion_matrix(confusion_matrix_, classes=target_names,
                              title='Confusion matrix, without normalization')
        plt.show()
        return F1_score,acc,recall_score_,confusion_matrix_,class_report

# 函数plot_confusion_matrix是绘制混淆矩阵的函数
#，CalculationResults则为只要给入y的预测值 + 实际值，以及分类的标签大致内容，就可以一次性输出：f1值，acc,recall以及报表


# In[36]:


CalculationResults(y_true,y_pred,,simple = False)


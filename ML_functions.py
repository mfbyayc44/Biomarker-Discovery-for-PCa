
def top12_protein_depletion(data,protein_tobe_removed):
    list_ = []
    for elem in protein_tobe_removed:
        if elem in data:
            list_.append(elem)
    return list_

def remove_depletion_protein(data):
    
    pd_alpha_glycoprotein = ['P19652']
    pd_glycoprotein= ["P02763","P19562"]
    pd_antitrypsin=["P01009","P20848",'P35030']
    pd_albumin = ["P02768"]
    pd_apolipoprotein = ["P02647","P02652","P02654","P02649","P02655","O95445","P02656","P05090"]
    pd_fibrinogen = ["Q08830","O75636","P02679","P02671","P02675"]
    pd_Ig = ["P24071","P01591","P55899","P08637","P12318","P31994","P01591",
        "P11912","P04207","P06309","P01763","P01714",'P06309','P01701',
        'P01768_UNMAPPED','P04207','P01701','P01877']
    pd_transferrin =["P02786","P02787",'P02788']
    protein_depletion = pd_glycoprotein + pd_antitrypsin + pd_albumin + pd_apolipoprotein + pd_fibrinogen + pd_Ig + pd_transferrin + pd_alpha_glycoprotein
    protein_to_be_removed = top12_protein_depletion(data.columns,protein_depletion)
    data = data.drop(protein_to_be_removed,axis=1)
    print(f"{len(protein_to_be_removed)} columns will be removed")
    return data

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

def univariate_selectKbest(X,y):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2 
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    k = featureScores.nlargest(10,'Score')
    return k

def convert_label_numerical(data):
    from sklearn.preprocessing import LabelEncoder
    classes = list(data.Label)
    class_encoder = LabelEncoder()
    y = class_encoder.fit_transform(classes)
    labels = list(class_encoder.classes_)
    return y,labels

def remove_highcorr(data):
    correlation = data.corr(method = 'pearson')
    highcorr = {}
    list = []
    for index in range(correlation.shape[1]):
        column = correlation.iloc[:,index]
        list = []
        for value in column:
            if 0.8 <= value < 1:
                list.append(value)
                highcorr[column.name] = list
    new_highcorr = []
    for keys,values in highcorr.items():
        if len(values)>= 2:
            new_highcorr.append(str(keys))
    print("{0} genes are found to be strongly correlated".format(len(new_highcorr)))
    for keys in new_highcorr:
        del data[keys]
    print(data.shape)
    return data,new_highcorr



def RF_feature_selection(X,y,protein_columns):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=2000,oob_score = True)
    rf.fit(X,y)
    dict_ = {'importances':rf.feature_importances_,
             'ID':protein_columns}
    feature_importances = pd.DataFrame(dict_,index = None).sort_values('importances',ascending = False)   
    return feature_importances
    #print("The accuracy of the model is: ",rf.oob_score_)
    
def draw_confusion_matrix(y_test,result, binary = True):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix (y_test,result)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax,annot_kws={'size':15},cmap=None); #annot=True to annotate cells

    # labels, title and ticks
    sns.set(font_scale=1.8)
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    if(binary): 
        ax.set_ylim([0,2]) ## correct the display of sns plot
        ax.xaxis.set_ticklabels(['Cancer', 'Control']); ax.yaxis.set_ticklabels(['Cancer', 'Control']);
    else:
        ax.xaxis.set_ticklabels(['Sur', 'Con','Non','Post','Pre']); 
        ax.yaxis.set_ticklabels(['Sur', 'Con','Non','Post','Pre']);
        
    
    
def draw_tsne(x_subset,y_subset,label,P):
    import time
    import seaborn as sns
    from sklearn.manifold import TSNE
    RS = 0 
    time_start = time.time()
    tsne = TSNE(random_state=RS).fit_transform(x_subset)
    print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    print("tsne shape: ", tsne.shape)
    list_ = P

    for i in list_:
        tsne = TSNE(random_state=RS,perplexity=i).fit_transform(x_subset)
        classes = list(label)
        ax = sns.scatterplot(tsne[:,0],tsne[:,1],hue = classes, legend = 'full')
        ax.set_title(f"t-sne plot (Perplexity = {i})")
        ax.set_xlabel("t-sne dimension one")
        ax.set_ylabel("t-sne dimension two")
        plt.legend(bbox_to_anchor=(1.2,1) ,loc = "upper right",fontsize = 10)
        plt.show()
        
        
def draw_ROC(fpr,tpr,train_auc,ML):
    from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
    from matplotlib import style
    style.use('ggplot')
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, 'b-', label= f"{ML}" + '(AUC = %.03f)' % train_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')
    plt.title('cancer vs. control')
    plt.legend(loc="lower right", prop={'size':10})
    plt.show()
    
    
    
    

    
    
    
    
    
    
    




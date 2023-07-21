import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import os
import glob
import sklearn
import nilearn as nl
from sklearn.manifold import TSNE
import re
from sklearn.cluster import KMeans
import math 
from numpy import dot
from numpy.linalg import norm
import warnings

warnings.filterwarnings('ignore')


"""
This function performs the following task:
-> takes a metadata table containing the mandatory demographics such 
as Subject ID, Age, Sex, and the group they belong to.
-> returns the descriptive stats such as no. of subjects in each 
group, avg., min, max age, and the gender count
"""
def Comp_Desc_Stats(metadata):
    Demographics = pd.DataFrame()
    # count subjects in each group
    Demographics["Subjects"] = metadata.groupby(["Group"]).count().Subject
    Demographics["Avg_Age"] = round(metadata.groupby(["Group"]).mean("Age"),2)
    Demographics["Std_Age"] = round(metadata.groupby(["Group"]).std(),2)
    Demographics["Min_Age"] = metadata.groupby(["Group"]).min("Age")
    Demographics["Max_Age"]  = metadata.groupby(["Group"]).max("Age")
    # count sex in each group 
    df = metadata.groupby(["Group","Sex"]).count().Subject.to_frame(name = 'Sex_Count').reset_index()
    Demographics["Male"] = df.loc[df["Sex"]=="M", "Sex_Count"].values
    Demographics["Female"] = df.loc[df["Sex"]=="F", "Sex_Count"].values
    Demographics["Group"] = Demographics.index
    Demographics.reset_index(inplace=True, drop=True)
    total_sub = metadata.loc[:,"Subject"].count()
    avg_age = round(metadata.loc[:,"Age"].mean(),2)
    std_age = round(metadata.loc[:,"Age"].std(),2)
    min_age = metadata.loc[:,"Age"].min()
    max_age = metadata.loc[:,"Age"].max()
    male_count = metadata.groupby(["Sex"]).count().Subject.values[1]
    female_count = metadata.groupby(["Sex"]).count().Subject.values[0]
    overall_ser = pd.Series([ total_sub, avg_age, std_age, min_age, max_age, male_count, female_count, "Overall"])
    overall = pd.DataFrame(overall_ser).transpose()
    overall.columns = Demographics.columns.values
    Demographics = pd.concat([Demographics, overall], ignore_index=True)
    return Demographics

"""
The below function does the following task:
---------------------------------------------
Creates an histogram for a continuous demographic variable such as age 
""" 
def Plot_Age_Hist(Ages, title, x_loc, path):
    mu = round(Ages.loc[:,"Age"].mean(),2)
    std = round(Ages.loc[:,"Age"].std(),2)
    min_age = round(Ages.loc[:,"Age"].min(), 0)
    max_age = round(Ages.loc[:,"Age"].max(),0)

    Ages["Age_Bins"] = pd.cut(x=Ages["Age"], bins=10)
    Ages["Age_Bins"] = Ages["Age_Bins"].apply(lambda x: pd.Interval(left=int(round(x.left)), right=int(round(x.right))))
    Ages_Piv = Ages.pivot_table("Age", index="Age_Bins", aggfunc='count', fill_value=0)
    Ages_Piv.plot(kind="bar", stacked=True, width=0.9, color="cadetblue", edgecolor="black", linewidth=0.5)
    # number of fixed locators
    x = np.arange(0,10)
    label = np.arange(min_age,max_age+1,step=(max_age-min_age)/10 )
    labels = []
    for i in np.arange(0,len(label)-1):
        # get the middle point of the age interval
        labels.append(round((label[i]+label[i+1])/2,None))
    print(labels, x)
    plt.xticks(ticks=x, labels=labels, rotation=None )  # Set label locations.
    # if axis_labels==True:
    plt.xlabel("Ages")
    plt.ylabel("No. of participants")
    plt.tight_layout()
    plt.title(title)
    plt.tight_layout()
    plt.text(x_loc, Ages_Piv.max().max()-5, r'$\mu='+str(mu)+',\ \sigma='+str(std)+'$')
    plt.savefig(path + title+ "_age_hist_with_title.png", bbox_inches='tight', dpi=300)


"==============================================================================="
"""
The below function does the following task:
---------------------input---------------------
1. A dataframe with two columns (age and gender)
2. title of the histogram to appear on the graph
3. Path to save the figure
---------------------output--------------------
An stacked bar chart showing the age and gender frequency in each interval
""" 
def Plot_Age_Gend_Hist(sub_demographics, title, x_loc, y_loc, path):
    mu = round(sub_demographics.loc[:,"Age"].mean(),1)
    sigma = round(sub_demographics.loc[:,"Age"].std(),1)
    # create an age bin column for each row where an age belongs to
    sub_demographics["Age_bins"] = pd.cut(x=sub_demographics.loc[:,"Age"], bins=10)
    sub_demographics["Age_bins"] = sub_demographics.loc[:,"Age_bins"].apply(lambda x: pd.Interval(left=int(round(x.left)), right=int(round(x.right))))
    # get the count of male and female subjects in all age intervals
    piv_age_sex = sub_demographics.pivot_table("Age", index="Age_bins", columns="Gender", aggfunc='count', fill_value=0)

    min_age = round(sub_demographics.loc[:,"Age"].min(), 0)
    max_age = round(sub_demographics.loc[:,"Age"].max(),0)
    # number of points or locations on the x axis
    x = np.arange(0,10)
    label = np.arange(min_age,max_age+1,step=(max_age-min_age)/10 )
    labels = []
    for i in np.arange(0,len(label)-1):
        # get the middle point of the age interval
        labels.append(round((label[i]+label[i+1])/2,None))
    print(labels)
    # set the column names (to appear as legend)
    piv_age_sex.columns = ["Male", "Female"]

    ax = piv_age_sex.plot(kind="bar", stacked=True, color=["azure", "white"], fill=True, width=0.8, edgecolor="gray", hatch="/.")
    # get all bars in the plot
    bars = ax.patches
    patterns = ['', '.']  # set hatch patterns in the correct order
    hatches = []  # list for hatches in the order of the bars
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    for bar, hatch in zip(bars, hatches):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)
    # generate legend. this is important to set explicitly, otherwise no hatches will be shown!
    ax.legend()

    plt.rcParams.update({'font.size': 14})
    plt.xticks(ticks=x, labels=labels, rotation=None)
    plt.title(title)
    plt.xlabel("Ages ", fontsize=14)
    plt.ylabel("No. of participants", fontsize=14)
    # plt.grid(linewidth=0.5)
    plt.tight_layout()
    plt.text(x_loc, y_loc, r'$\mu='+str(mu)+',\ \sigma='+str(sigma)+'$')
    plt.savefig(path + title+ "_age_gender_hist_with_title.png", bbox_inches='tight', dpi=300)


"=================================================================================="
"""
The below function does the following task:
-----------------------------------------
Visualize the relationship between chronological ages and the estimated 
brain ages using scatter plots
-----------------------------------------
Takes the actual and predicted brain ages as input
"""
" ========= when both healthy and cases ============"
def Plot_true_vs_pred_grouped(Y_pred_hc,Y_test_hc, Y_pred_case,  Y_test_case, title, R_sqq, path):
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 20})
    sp_names = [ "Healthy Control", "Brain Tumor"]
    sct1 = plt.scatter(Y_test_hc, Y_pred_hc, c='tan', s=50, marker="v")
    sct2 = plt.scatter(Y_test_case, Y_pred_case, c='cadetblue', s=50, marker="x")
    plt.legend((sct1, sct2), labels=sp_names)
    Plot_Scatter(R_sqq,title, path)

"========== healthy controls only ================"
def Plot_true_vs_pred(Y_test_hc, Y_pred_hc, title, R_sqq, path):
    plt.figure(figsize=(8, 6))
    sct1 = plt.scatter(Y_test_hc, Y_pred_hc, c='saddlebrown', s=20)
    # plt.legend((sct1, sct2), labels=sp_names)
    Plot_Scatter(R_sqq, title, path)

"========= Reusable code block ==================="
def Plot_Scatter(R_sqq,title,path):
    plt.yscale('linear')
    plt.xscale('linear')
    plt.rcParams.update({'font.size': 16})
    p1 = 5
    p2 = 80
    plt.plot([p1, p2], [p1, p2], 'dimgray')
    plt.plot([p1+10, p2+10], [p1, p2], 'dimgray', linestyle="--")
    plt.plot([p1, p2], [p1+10, p2+10], 'dimgray', linestyle="--")
    plt.text(10, 60, "$R^2 = " + str(R_sqq) +"$", fontsize =16)
    plt.xlabel("Chronological Age")
    plt.ylabel("Estimated Brain Age")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path+"true_vs_pred_scat_plot.png", dpi=300)


def Plot_true_vs_BAG_grouped(Y_pred_hc,Y_test_hc, Y_pred_case,  Y_test_case, title, path):
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    BAG_test = (Y_pred_hc-np.array(Y_test_hc).reshape(1,-1)).transpose()
    BAG_bt = (Y_pred_case-np.array(Y_test_case).reshape(1,-1)).transpose()

    sp_names = [ "Healthy Control", "Brain Tumor"]

    sct1 = plt.scatter( Y_test_hc, BAG_test, marker="v", c="tan", s=50)
    sct2 = plt.scatter( Y_test_case, BAG_bt, marker="x", c="cadetblue", s=50)

    plt.legend((sct1, sct2), labels=sp_names)

    Y_test_arr = np.array(Y_test_hc).flatten()
    BAG_test_arr = np.array(BAG_test).flatten()
    # calc the trendline
    z = np.polyfit(Y_test_arr, BAG_test_arr, 1)
    p = np.poly1d(z)
    plt.plot(Y_test_arr,p(Y_test_arr),"r--", color="tan")

    Y_bt_arr = np.array(Y_test_case).flatten()
    BAG_bt_arr = np.array(BAG_bt).flatten()
    # calc the trendline
    z = np.polyfit(Y_bt_arr, BAG_bt_arr, 1)
    p = np.poly1d(z)
    plt.plot(Y_bt_arr,p(Y_bt_arr),"r--", color="cadetblue")

    plt.ylabel("BAG (years)")
    plt.xlabel("Chronological Age")
    plt.axhline(0, color='gray', linestyle="dotted")
    plt.tight_layout()
    plt.savefig(path+"true_vs_BAG_scat_plot.png", dpi=300)


####### Plot the age-gap box plots for different features
def Box_plot_age_gap_diff_feat(Y_test, Pred_brain_age, xlabels, path):
    plt.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots()
    bp = plt.boxplot(Pred_brain_age-Y_test.reshape(-1,1), 1, "", positions=np.arange(1,15,2),  widths=1, 
                    patch_artist=True, boxprops=dict(facecolor="white"))

    for box in bp['boxes']:
        # change outline color
        box.set(linewidth=1)
        # change hatch
        box.set(hatch = '\\')

    plt.ylabel("Brain-EAD (years)")
    if (xlabels==True):
        ax.set_xticklabels(["CAT12 ROI", "Desikan ROI", "Destrieux ROI", "CAT12 ROI + Desikan ROI", 
        "CAT12 ROI + Destrieux ROI","Desikan ROI + Destrieux ROI", "All region wise" ], rotation=90)
    else:

        plt.xticks([1], [''])
        ax.set_xlim(0,15)
        plt.axhline(0, color='gray', linestyle="dotted")
        plt.savefig(path + "_no_labels.png", bbox_inches='tight', dpi=300)




####### Plot the age-gap box plots for different features
def Box_plot_ages(Y_test, Pred_brain_age, xlabels, path):
    plt.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots()
    bp = plt.boxplot(Pred_brain_age-Y_test.reshape(-1,1), 1, "", positions=np.arange(1,15,2),  widths=1, 
                    patch_artist=True, boxprops=dict(facecolor="white"))

    for box in bp['boxes']:
        # change outline color
        box.set(linewidth=1)
        # change hatch
        box.set(hatch = '\\')
 
    plt.ylabel("Brain-EAD (years)")
    if (xlabels==True):
        ax.set_xticklabels(["CAT12 ROI", "Desikan ROI", "Destrieux ROI", "CAT12 ROI + Desikan ROI", 
        "CAT12 ROI + Destrieux ROI","Desikan ROI + Destrieux ROI", "All region wise" ], rotation=90)
    else:

        plt.xticks([1], [''])
        ax.set_xlim(0,15)
        plt.axhline(0, color='gray', linestyle="dotted")
        plt.savefig(path + "_no_labels.png", bbox_inches='tight', dpi=300)


def Box_Plot_Age_Gap(Y_pred_test, Y_test, Y_pred_bt, Y_bt, path):
    plt.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots()
    BAG_test = (Y_pred_test-np.array(Y_test).reshape(1,-1)).transpose()
    BAG_bt = (Y_pred_bt-np.array(Y_bt).reshape(1,-1)).transpose()

    # combine the two arrays with unequal lengths
    data = np.array([BAG_test, BAG_bt])

    # box plot of two arrays or columns with unequal lengths 
    bp = plt.boxplot(data, patch_artist=True, widths=0.5)

    i=0
    for box in bp['boxes']:  # loop through the boxes
        # apply different conditions to the boxes
        if (i==0):
            # change outline color
            box.set(facecolor="wheat")
        else:
            box.set(facecolor="cadetblue")
        i = i+1

    plt.ylabel("BAG (years)")
    ax.set_xticklabels(["Healthy Control", "Brain Tumor"])
    plt.axhline(0, color='gray', linestyle="dotted")
    plt.savefig(path + "_age_gap_box_plot.png", bbox_inches='tight', dpi=300)


def Plot_Survival_vs_BAG(Survival_days, BAG_bt, path):
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()

    BAG_bt = abs(BAG_bt)

    # create a dataframe on with BAG and Survival_days as columns and sort it
    x = pd.DataFrame({"BAG": pd.Series(BAG_bt), "Survival_days": Survival_days.astype(int)})
    x = x.sort_values(by=["Survival_days"], ascending=False)

    plt.scatter(x.loc[:,"BAG"], x.loc[:,"Survival_days"], marker="x", c="cadetblue")
    
    # insert comma when the legend label text is truncated or sliced showing only first alphabet
    plt.legend(("Brain Tumor",))

    BAG_bt_arr = np.array(x.loc[:,"BAG"], dtype=float).reshape(60)
    Survival_arr = np.array(x.loc[:,"Survival_days"], dtype=float).reshape(60)
    # calc the trendline
    z = np.polyfit(BAG_bt_arr, Survival_arr, 1)
    p = np.poly1d(z)
    plt.plot(BAG_bt_arr, p(BAG_bt_arr),"r--", color="tan")

    # set y tick labels after a certain 'n' here n=1 
    # used when length of the points is very much
    ax.set_yticks(ax.get_yticks()[::1])

    plt.ylim(0,2000)
    plt.xlabel("BAG (years)")
    plt.ylabel("Survival (days)")
    plt.tight_layout()
    plt.savefig(path+"BAG_vs_survival_scat_plot.png", dpi=300, bbox_inches='tight')


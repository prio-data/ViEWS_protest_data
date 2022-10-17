from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score,roc_curve

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)



def df_eval_scores(
    models,
    ev_name,
    calib,
    steps,
    round_to = 3, 
    path= None,
) -> pd.DataFrame:
    """Store selected evaluation scores in pd.DataFrame for list of models."""

    scores = {"Model": []}

    for s in steps:
        scores.update({f"{ev_name}_{s}": []})

    for model in models:
        
        scores["Model"].append(model['modelname'])
        mname = model['modelname']
        
        for step in steps:
            if ev_name == "Brier":
                scores[f"{ev_name}_{step}"].append(brier_score_loss(
                    y_true = model[f'df_test_{calib}'][f'actuals_step{step}'],
                    y_prob = model[f'df_test_{calib}'][f'{mname}_{calib}_step{step}']))
            if ev_name == "AUROC":
                scores[f"{ev_name}_{step}"].append(roc_auc_score(
                    y_true = model[f'df_test_{calib}'][f'actuals_step{step}'],
                    y_score = model[f'df_test_{calib}'][f'{mname}_{calib}_step{step}']))
            if ev_name == "AP":
                scores[f"{ev_name}_{step}"].append(average_precision_score(
                    y_true = model[f'df_test_{calib}'][f'actuals_step{step}'],
                    y_score = model[f'df_test_{calib}'][f'{mname}_{calib}_step{step}']))
                
    out = pd.DataFrame(scores)
    df_out = out.set_index("Model")
    df_out.columns = pd.MultiIndex.from_tuples(
        tuple(df_out.columns.str.split("_"))
    )

    if path:
        # Write to tex. file. 
        tex = df_out.reset_index().to_latex(index=False)

        # Get meta infromation
        now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        meta = f"""
        %Date: {now}
        %Output created by protest_paper.ipynb.
        %Compare {ev_name} for all models.
        \\
        """
        tex = meta + tex
        with open(path, "w") as f:
            f.write(tex)
        print(f"Wrote scores table to {path}")

    return df_out

def plot_parcoord_allsteps(
    df,
    steps,
    legend_label= None,
    cmap = "tab20",
    reverse = False,
    path=  None,
):
    
    df = df.copy()
    df = df.loc[:,df.columns.get_level_values(0).isin(steps)]
    
    ys1 = df.values
    print(ys1)
    print(ys1.min(axis=0))
    ymins1 = np.array(
        (min([ys1.min(axis=0)[index] for index in [0,3,6,9]]), 
         min([ys1.min(axis=0)[index] for index in [1,4,7,10]]),
         min([ys1.min(axis=0)[index] for index in [2,5,8,11]]),
         min([ys1.min(axis=0)[index] for index in [0,3,6,9]]), 
         min([ys1.min(axis=0)[index] for index in [1,4,7,10]]),
         min([ys1.min(axis=0)[index] for index in [2,5,8,11]]),
         min([ys1.min(axis=0)[index] for index in [0,3,6,9]]), 
         min([ys1.min(axis=0)[index] for index in [1,4,7,10]]),
         min([ys1.min(axis=0)[index] for index in [2,5,8,11]]),
         min([ys1.min(axis=0)[index] for index in [0,3,6,9]]), 
         min([ys1.min(axis=0)[index] for index in [1,4,7,10]]),
         min([ys1.min(axis=0)[index] for index in [2,5,8,11]]),
        ))
    
    ymaxs1 = np.array(
        (max([ys1.max(axis=0)[index] for index in [0,3,6,9]]), 
         max([ys1.max(axis=0)[index] for index in [1,4,7,10]]), 
         max([ys1.max(axis=0)[index] for index in [2,5,8,11]]), 
         max([ys1.max(axis=0)[index] for index in [0,3,6,9]]), 
         max([ys1.max(axis=0)[index] for index in [1,4,7,10]]), 
         max([ys1.max(axis=0)[index] for index in [2,5,8,11]]), 
         max([ys1.max(axis=0)[index] for index in [0,3,6,9]]), 
         max([ys1.max(axis=0)[index] for index in [1,4,7,10]]), 
         max([ys1.max(axis=0)[index] for index in [2,5,8,11]]), 
         max([ys1.max(axis=0)[index] for index in [0,3,6,9]]), 
         max([ys1.max(axis=0)[index] for index in [1,4,7,10]]), 
         max([ys1.max(axis=0)[index] for index in [2,5,8,11]]), 
        ))
    
    print(len(ymins1))
    print(len(ymaxs1))
    
    #ys1.min(axis=0)
    #ymaxs1 = ys1.max(axis=0)

    dys1 = ymaxs1 - ymins1
    print(dys1)
    ymins1 -= dys1 * 0.05  # add 0.05 padding below and above
    ymaxs1 += dys1 * 0.05
    
    if reverse: # Brier
        ymaxs1[2], ymins1[2] = ymins1[2], ymaxs1[2]  # reverse axis 6 to have less crossings
        ymaxs1[5], ymins1[5] = ymins1[5], ymaxs1[5]  # 
        ymaxs1[8], ymins1[8] = ymins1[8], ymaxs1[8]
        ymaxs1[11], ymins1[11] = ymins1[11], ymaxs1[11]
        #ymaxs1[10], ymins1[10] = ymins1[10], ymaxs1[10]
        
    dys1 = ymaxs1 - ymins1
    # Transform data using broadcasting to be compatible with the main axis
    zs1 = np.zeros_like(ys1)
    zs1[:, 0] = ys1[:, 0]
    zs1[:, 1:] = (ys1[:, 1:] - ymins1[1:]) / dys1[1:] * dys1[0] + ymins1[0]

    fig, host = plt.subplots(figsize=(12, 5))

    # Set up and adapt individual axes
    axes = [host] + [host.twinx() for i in range(ys1.shape[1] - 1)]
    len(axes)

    for i, ax in enumerate(axes):
        # Set the tick range manually, adapting from host
        # Note that the actual lines will be plotted according to the
        # transformed values above (i.e. all in terms of axis 0.)
        # So essentially these are cosmetic axes.
        ax.set_ylim(ymins1[i], ymaxs1[i])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if ax != host:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_ticks_position("right")
            # Reset drawing position of non-host axes (i fraction of len cols)
            #ax.spines["right"].set_position(("axes", i / (ys1.shape[1] - 1)))
            if i in [1]:
                ax.spines["right"].set_position(("axes", 0.07))
            if i in [2]:
                ax.spines["right"].set_position(("axes", 0.07+0.07))
            if i in [3]:
                ax.spines["right"].set_position(("axes", 0.07+0.07+0.12))
            if i in [4]:
                ax.spines["right"].set_position(("axes", 0.26+0.07))
            if i in [5]:
                ax.spines["right"].set_position(("axes", 0.26+0.07+0.07))
            if i in [6]:
                ax.spines["right"].set_position(("axes", 0.26+0.07+0.07+0.12))
            if i in [7]:
                ax.spines["right"].set_position(("axes", 0.26+0.07+0.07+0.12+0.07))
            if i in [8]:
                ax.spines["right"].set_position(("axes", 0.26+0.07+0.07+0.12+0.07+0.07))
            if i in [9]:
                ax.spines["right"].set_position(("axes", 0.26+0.07+0.07+0.12+0.07+0.07+0.12))
            if i in [10]:
                ax.spines["right"].set_position(("axes", 0.26+0.07+0.07+0.12+0.07+0.07+0.12+0.07))
            if i in [11]:
                ax.spines["right"].set_position(("axes", 0.26+0.07+0.07+0.12+0.07+0.07+0.12+0.07+0.07))

    host.set_xlim(0, 10)
    host.set_xticklabels(['AP','3\nAUROC','Brier','AP','6\nAUROC','Brier','AP','12\nAUROC','Brier','AP','36\nAUROC','Brier'], fontsize=12)
    host.tick_params(axis="x", which="major", pad=7)
    host.spines["right"].set_visible(False)
    host.xaxis.tick_top()
    host.set_xticks([0,0.7,1.4,2.6,3.3,4,5.2,5.9,6.6,7.8,8.5,9.2])

    #host.set_title(f"Evaluation scores", fontsize=18, pad=12)

    cm = plt.cm.get_cmap(cmap)
    colors = cm.colors

    for j in range(ys1.shape[0]):
        #For j submission, plot the row values by column
        host.plot(
            [0,0.7,1.4,2.6,3.3,4,5.2,5.9,6.6,7.8,8.5,9.2], zs1[j, :], c=colors[(j - 1) % len(colors)]
        )
    if legend_label:
        legend_label = legend_label
    else:
        legend_label = df.index
        
    axis_coords = 1
    host.legend(
       labels=legend_label,
        loc="center",
        bbox_to_anchor=(axis_coords, 0, 0.05, 1.1),
        title=f"Models",
    )
    
    plt.tight_layout()
    
    if path:
        fig.savefig(
                    path,
                    dpi=300,
                    facecolor="white",
                    bbox_inches="tight",
                )
        print((f"Wrote {path}."))
        plt.show
        
def boot_evalmetric(
    model, 
    step,
    eval_fun,
    calib,
    set_seed, # check for reproducibility
    n_bootstraps,
    
):
    bootsrapped_scores = []
    rng = np.random.RandomState(set_seed)
    
    mname = model['modelname']
    predicted = model[f'df_test_{calib}'][f'{mname}_{calib}_step{step}'].values
    actuals = model[f'df_test_{calib}'][f'actuals_step{step}'].values  
    
        
    for i in range(n_bootstraps): 
        # bootstrap by sampling with replacement
        indices = rng.randint(low=0, high=len(predicted), size=len(actuals)) # Return random integers from low (inclusive) to high (exclusive).
        if len(np.unique(actuals[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
            
        if eval_fun == 'average_precision':
            score = average_precision_score(
                y_true=actuals[indices], 
                y_score = predicted[indices],
            )
        if eval_fun == 'area_under_roc':
            score = roc_auc_score(
                y_true=actuals[indices], 
                y_score=predicted[indices],
            )
    
        bootsrapped_scores.append(score)
        
    df_bootsrapped_scores = pd.DataFrame(bootsrapped_scores, columns=[f'{mname}'+f'_{eval_fun}_{step}'])
       
    return df_bootsrapped_scores

def plot_bootstrapped_diff(
    df,
    titles,
    modellist1,
    modelllist2,
    legendtrue,
    steps,
    ymin,
    ymax,
    path_out
):

    for t,m1,m2 in zip(titles,modellist1,modelllist2):
        boot_df = pd.DataFrame()

        for i in steps:
            boot_df[f'diff{i}'] = df[f'{m1}{i}']-df[f'{m2}{i}']
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)

        lower_bounds = []
        upper_bounds = []
        stdlower_bounds = []
        stdupper_bounds = []
        models_list = []

        for col in boot_df.columns:
            sorted_scores = np.array(boot_df[col])
            sorted_scores.sort()

            # Computing the lower and upper bound of the 95% confidence interval
            confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
            confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
            std_lower = boot_df.mean() - (boot_df.std()*2)
            std_upper = boot_df.mean() + (boot_df.std()*2)
            lower_bounds.append(confidence_lower)
            upper_bounds.append(confidence_upper)

        plt.vlines('diff3',lower_bounds[0],upper_bounds[0],color='black',lw=1)
        plt.vlines('diff6',lower_bounds[1],upper_bounds[1],color='black',lw=1)
        plt.vlines('diff12',lower_bounds[2],upper_bounds[2],color='black',lw=1)
        plt.vlines('diff36',lower_bounds[3],upper_bounds[3],color='black',lw=1)
        test = ax.get_position()
        ax.plot(boot_df.mean(),'o',color='black', lw = 0.8,  markeredgewidth = 1)
        ax.axhspan(std_lower[0],std_upper[0],xmin = 0.02,xmax = 0.07,
              facecolor ='lightgrey', alpha = 0.3)
        ax.axhspan(std_lower[1],std_upper[1],xmin = 0.325,xmax = 0.37,
               facecolor ='lightgrey', alpha = 0.3)
        ax.axhspan(std_lower[2],std_upper[2],xmin = 0.62,xmax =0.68,
                facecolor ='lightgrey', alpha = 0.3)
        ax.axhspan(std_lower[3],std_upper[3],xmin = 0.935,xmax =0.98,
                facecolor ='lightgrey', alpha = 0.3)
        ax.plot(lower_bounds,'_',
                markeredgewidth = 2,
                markersize=12,
                color='black')
        ax.plot(upper_bounds,'_',
                markeredgewidth = 2,
                markersize=12,
                color='black')
        plt.axhline(y = 0, color = 'black', linestyle = '--',linewidth=2)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(b=True, axis='y', color='grey', linestyle='--',alpha=0.2)
        ax.grid(b=True, which='minor', axis='y', color='grey', linestyle='--',alpha=0.2)
        for spine in plt.gca().spines.values():
                spine.set_visible(False)

        ax.set_xticklabels(steps)
        ax.set_ylim(ymin, ymax)
        plt.ylabel('Difference in AP')
        plt.xlabel('Steps')
        plt.title(t)


        if legendtrue:
            legendlabels= ''
            plt.legend(
                   labels=[legendlabels],
                    loc="center",
                    bbox_to_anchor=(0, 0, 2.5, 1.1),
                    title=f"Models",
                )
        plt.show()
        plt.savefig(path_out,bbox_inches='tight')
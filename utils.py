import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def uniform_value_from_range(min, max):
    value = min + (max - min) * np.random.rand()
    return value

def dist_euclidean(XY0, XY1):
    return np.sqrt( (XY0[0]-XY1[0])*(XY0[0]-XY1[0]) + (XY0[1]-XY1[1])*(XY0[1]-XY1[1]) )

def get_interpolated_xs_ys(xs,ys, yes_plot=False):
    # Given a set of (xs,ys) coordinates, it will find the intersection and compute the average.
    # It will take the maximum of the x0 and the minimum of the x_last.
    # Even if the x's are not the same, interpolation is used.
    # INPUT:
    # xs: list of x-coordinates
    # ys: list of y-coordinates
    # OUTPUT:
    # mean_x_axis, mean_y_axis, std_y_axis
    
    # 1) Get the intersection among the x coordinates
    lastmin = min([np.max(i) for i in xs])
    firstmax = max([i[0] for i in xs])

    # 2) Set the x range
#     xx = list(np.arange(firstmax,lastmin,step))
#     if lastmin not in xx:
#         xx.append(lastmin)
    xx = np.linspace(firstmax,lastmin,1000,endpoint=True)

    # 3) Get the mean on interpolated x
    mean_x_axis = [i for i in xx]
    ys_interp = [np.interp(mean_x_axis, xs[i], ys[i]) for i in range(len(xs))]
    mean_y_axis = np.mean(ys_interp, axis=0)
    std_y_axis = np.std(ys_interp, axis=0)

    # 4) Plot within the intersection range
    if yes_plot:
        plt.plot(mean_x_axis, mean_y_axis)
        plt.fill_between(mean_x_axis, mean_y_axis-std_y_axis, mean_y_axis+std_y_axis, alpha=0.2)
        
    return mean_x_axis, mean_y_axis, std_y_axis

def plot_df_scoreboard(df_dict,selected_keys,ttitle,xlabel,ylabel, xlim=None, ylim=None,  leg_loc=4):
    # INPUT:
    # - df_dict: dictionary of df
    # - selected_keys: you can choose what to average
    # - xlabel, ylabel: column names you want to plot
    # - xlim: optionally, you can set the right x-axis limit
    # OUTPUT:
    # - Plot each individual lines as well as the average

    plt.figure(num=None, figsize=(12, 5.5), dpi=80, facecolor='w', edgecolor='k')
    # style
    plt.style.use('seaborn-darkgrid')
    # create a color palette
    palette = plt.get_cmap('Set1')
    # font size
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    
    xs = []
    ys = []
    for i,key in enumerate(selected_keys):
#         x = df_dict[key].TimeStepsTotal
#         y = df_dict[key].Avg100Reward
        x = df_dict[key][xlabel]
        y = df_dict[key][ylabel]
        plt.plot(x,y, marker='', color=palette(i), linewidth=1, alpha=0.9, label=key)
        # Store for the average plot
        xs.append(list(x))
        ys.append(list(y))

    # Compute interpolated average across seeds
    xmean, ymean, ystd = get_interpolated_xs_ys(xs,ys)
    plt.plot(xmean,ymean,'--', marker='', color='black', linewidth=2, alpha=0.5, label='Mean')
    plt.fill_between(xmean, ymean-ystd, ymean+ystd,color='black',alpha=0.1)

    # Add legend
    plt.legend(loc=leg_loc, ncol=2, fontsize = 15)
    # Add titles
    plt.title(ttitle, loc='left', fontsize=18, fontweight=0, color='black')
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel(ylabel,fontsize=15)
    
    if xlim:
        plt.xlim(right=xlim)
    if ylim:
        plt.ylim(top=ylim)

def plot_df_scoreboard_compare_n_clean(list_df_dict,list_labels,ttitle,xlabel,ylabel,xxlabel,yylabel, xlim=None, ylim=None,  leg_loc=4, noLegend=False, xlim_min=None, ylim_min=None, trialtype=None, idx_custom_color=None):
    # INPUT:
    # - df_dict1, df_dict2: dictionary of df
    # - selected_keys: you can choose what to average
    # - xlabel, ylabel: column names you want to plot
    # - xlim: optionally, you can set the right x-axis limit
    # OUTPUT:
    # - Plot each individual lines as well as the average

    # REF: 
    # Simple linestyles can be defined using the strings "solid", "dotted", "dashed" or "dashdot". 
    # More refined control can be achieved by providing a dash tuple (offset, (on_off_seq)). 
    # For example, (0, (3, 10, 1, 15)) means (3pt line, 10pt space, 1pt line, 15pt space) with no offset. See also Line2D.set_linestyle.
    
#     mystyles = ['--','-.',(0, (3, 5, 1, 5, 1, 5)),':','-',(0, (5, 5, 1, 10, 1, 5))]

    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') # DEFAULT
    # plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k') # 3x3 figure
    # style
    plt.style.use('seaborn-darkgrid')
    # create a color palette
    palette = plt.get_cmap('Set1')
    # font size
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    
    c = 0
#     for df_dict,label,style in zip(list_df_dict,list_labels,mystyles):
    for df_dict,label in zip(list_df_dict,list_labels):
        # df#1
        style = '-'
        xs1 = []
        ys1 = []
        for i,key in enumerate(df_dict.keys()):
            if trialtype:
                dfx = df_dict[key]
                dfxx = dfx[dfx.trialtype==trialtype]
                if trialtype == 'Go':
                    x = dfxx[xlabel].rolling(100).mean().dropna()
                    y = dfxx[ylabel].rolling(100).mean().dropna()
                else:
                    x = dfxx[xlabel].rolling(100).mean().dropna()
                    y = dfxx[ylabel].rolling(100).mean().dropna()
            else:
                x = df_dict[key][xlabel]
                y = df_dict[key][ylabel]
            # plt.plot(x,y, marker='', color=palette(i), linewidth=1, alpha=0.9, label=key)
            # Store for the average plot
            xs1.append(list(x))
            ys1.append(list(y))

        # Compute interpolated average across seeds
        if c==5:
            c+=1 # 5 is yellow... hard to see.
        xmean1, ymean1, ystd1 = get_interpolated_xs_ys(xs1,ys1)
        # Change palette color if custom index is given
        if idx_custom_color:
            curr_color = palette(idx_custom_color[c])
        else:
            curr_color = palette(c)
        plt.plot(xmean1,ymean1,linestyle=style, marker='', color=curr_color, linewidth=2, alpha=0.8, label='%s'%label)
        plt.fill_between(xmean1, ymean1-ystd1, ymean1+ystd1,color=curr_color,alpha=0.1)
        
        print('AvgStd(%s):'%label,np.mean(ystd1))
        c += 1

    # Add legend
    if not noLegend:
        plt.legend(loc=leg_loc, ncol=1, fontsize = 14)
    # Add titles
    plt.title(ttitle, loc='left', fontsize=18, fontweight=0, color='black')
    plt.xlabel(xxlabel,fontsize=18)
    plt.ylabel(yylabel,fontsize=18)
    
    # Scientific notation
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    if xlim:
        plt.xlim(right=xlim)
    if ylim:
        plt.ylim(top=ylim)
    if xlim_min:
        plt.xlim(left=xlim_min)
    plt.xlim(left=-1000)
    if ylim_min:
        plt.ylim(bottom=ylim_min)
        
def plot_df_scoreboard_compare_n(list_df_dict,list_labels,ttitle,xlabel,ylabel, xlim=None, ylim=None,  leg_loc=4):
    # INPUT:
    # - df_dict1, df_dict2: dictionary of df
    # - selected_keys: you can choose what to average
    # - xlabel, ylabel: column names you want to plot
    # - xlim: optionally, you can set the right x-axis limit
    # OUTPUT:
    # - Plot each individual lines as well as the average

    # REF: 
    # Simple linestyles can be defined using the strings "solid", "dotted", "dashed" or "dashdot". 
    # More refined control can be achieved by providing a dash tuple (offset, (on_off_seq)). 
    # For example, (0, (3, 10, 1, 15)) means (3pt line, 10pt space, 1pt line, 15pt space) with no offset. See also Line2D.set_linestyle.
    mystyles = ['--','-.',(0, (3, 5, 1, 5, 1, 5)),':','-',(0, (5, 5, 1, 10, 1, 5))]
    plt.figure(num=None, figsize=(12, 5.5), dpi=80, facecolor='w', edgecolor='k')
    # style
    plt.style.use('seaborn-darkgrid')
    # create a color palette
    palette = plt.get_cmap('Set1')
    # font size
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    
    c = 0
    for df_dict,label,style in zip(list_df_dict,list_labels,mystyles):
        # df#1
        xs1 = []
        ys1 = []
        for i,key in enumerate(df_dict.keys()):
            x = df_dict[key][xlabel]
            y = df_dict[key][ylabel]
            # plt.plot(x,y, marker='', color=palette(i), linewidth=1, alpha=0.9, label=key)
            # Store for the average plot
            xs1.append(list(x))
            ys1.append(list(y))

        # Compute interpolated average across seeds
        if c==5:
            c+=1 # 5 is yellow... hard to see.
        xmean1, ymean1, ystd1 = get_interpolated_xs_ys(xs1,ys1)
        plt.plot(xmean1,ymean1,linestyle=style, marker='', color=palette(c), linewidth=2, alpha=0.5, label='Mean(%s)'%label)
        plt.fill_between(xmean1, ymean1-ystd1, ymean1+ystd1,color=palette(c),alpha=0.1)
        c += 1

    # Add legend
    plt.legend(loc=leg_loc, ncol=1, fontsize = 15)
    # Add titles
    plt.title(ttitle, loc='left', fontsize=18, fontweight=0, color='black')
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel(ylabel,fontsize=15)
    
    if xlim:
        plt.xlim(right=xlim)
    if ylim:
        plt.ylim(top=ylim)

def plot_df_scoreboard_compare(df_dict1,label1,df_dict2,label2,ttitle,xlabel,ylabel, xlim=None, ylim=None,  leg_loc=4):
    # INPUT:
    # - df_dict1, df_dict2: dictionary of df
    # - selected_keys: you can choose what to average
    # - xlabel, ylabel: column names you want to plot
    # - xlim: optionally, you can set the right x-axis limit
    # OUTPUT:
    # - Plot each individual lines as well as the average

    plt.figure(num=None, figsize=(12, 5.5), dpi=80, facecolor='w', edgecolor='k')
    # style
    plt.style.use('seaborn-darkgrid')
    # create a color palette
    palette = plt.get_cmap('Set1')
    # font size
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    
    # df#1
    xs1 = []
    ys1 = []
    for i,key in enumerate(df_dict1.keys()):
        x = df_dict1[key][xlabel]
        y = df_dict1[key][ylabel]
        # plt.plot(x,y, marker='', color=palette(i), linewidth=1, alpha=0.9, label=key)
        # Store for the average plot
        xs1.append(list(x))
        ys1.append(list(y))

    # Compute interpolated average across seeds
    xmean1, ymean1, ystd1 = get_interpolated_xs_ys(xs1,ys1)
    plt.plot(xmean1,ymean1,'-', marker='', color='blue', linewidth=2, alpha=0.5, label='Mean(%s)'%label1)
    plt.fill_between(xmean1, ymean1-ystd1, ymean1+ystd1,color='blue',alpha=0.1)

    # df#2
    xs2 = []
    ys2 = []
    for i,key in enumerate(df_dict2.keys()):
        x = df_dict2[key][xlabel]
        y = df_dict2[key][ylabel]
        # plt.plot(x,y, marker='', color=palette(i), linewidth=1, alpha=0.9, label=key)
        # Store for the average plot
        xs2.append(list(x))
        ys2.append(list(y))

    # Compute interpolated average across seeds
    xmean2, ymean2, ystd2 = get_interpolated_xs_ys(xs2,ys2)
    plt.plot(xmean2,ymean2,'-', marker='', color='red', linewidth=2, alpha=0.5, label='Mean(%s)'%label2)
    plt.fill_between(xmean2, ymean2-ystd2, ymean2+ystd2,color='red',alpha=0.1)

    # Add legend
    plt.legend(loc=leg_loc, ncol=1, fontsize = 15)
    # Add titles
    plt.title(ttitle, loc='left', fontsize=18, fontweight=0, color='black')
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel(ylabel,fontsize=15)
    
    if xlim:
        plt.xlim(right=xlim)
    if ylim:
        plt.ylim(top=ylim)
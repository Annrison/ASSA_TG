import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_seed(fpath,top_n,select_general=False):
    fseed_sen = open(fpath, 'r', encoding='utf-8')
    all_seed = []
    for line in fseed_sen: # positive & negative
        seeds_asp_sen = []
        for asp_line in line.split(' | '): # positive or negative
            seeds_sen = []
            for tok in asp_line.split():
                word, weight = tok.split(':')
                seeds_sen.append(word) # word id of seed
            seeds_sen = seeds_sen[:top_n]
            seeds_asp_sen.append(seeds_sen) # seed word (all aspect)
        all_seed.append(seeds_asp_sen)
        
    d = {}
    if select_general:
        p = [l[0] for l in all_seed] # positive
        n = [l[1] for l in all_seed] # negative        
    else:
        p = [l[0][5:] for l in all_seed] # positive
        n = [l[1][5:] for l in all_seed] # negative                
    d['pos'] = p
    d['neg'] = n    
    return d

# seeds from all rounds
def seeds_to_df(path,rounds,domain,top_n,select_general, asp_dict):
    
    aspect_list = asp_dict[domain]
    dall = {}
    for r in range(1,rounds+1):
        f0 = f'{path}{r}.txt'
        a0 = read_seed(f0,top_n,select_general)
        dall[f'R{r}'] = a0
    temp = []
    for r in dall:
        for s in dall[r]:
            for a in dall[r][s]:
                temp.append([r,s,a])
    
    df = pd.DataFrame(data=temp, columns=['round','sentiment','seeds'])      
    df['aspect'] = np.resize(aspect_list,len(df))
    
    return df

def join_list(l):
    return " ".join(l)


# 讀取所有 performance 裡面的 csv，加起來之後取平均
def read_result(model_ver, TRound, pdir):
    path = f'{pdir}/{model_ver}/performance/ASSA_R'
    df = pd.DataFrame()
    
    for r in range(0,TRound):
        dff = '{0}{1}.csv'.format(path,r)
        df_temp = pd.read_csv(dff)
        df_temp['Round'] = r
        df = pd.concat([df, df_temp])
    
    mean_data = df.groupby(['Epoch']).mean().reset_index()
    df['name'] = model_ver    
    mean_data['name'] = model_ver 
    
    return mean_data

# 畫出不同round之間的差別
def plot_result_round(df, score_type, graph_type, TRound):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 5)

    for r in range(0,TRound):
        df_sel = df[df['Round'] == r]
        x = df_sel['Epoch']
        axs[0].plot(x, df_sel[f'A{score_type}'], label = f"Round{r}")    
        axs[0].legend()
    axs[0].set_title('Axis [0, 0]')

    for r in range(0,TRound):
        df_sel = df[df['Round'] == r]
        x = df_sel['Epoch']
        axs[1].plot(x, df_sel[f'S{score_type}'], label = f"Round{r}")    
        axs[1].legend()

    axs[0].set_title(f'Aspect {score_type}', fontsize=16)
    axs[1].set_title(f'Sentiment {score_type}', fontsize=16)

    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel=score_type)

    fig.suptitle(f'{graph_type} Graph', fontsize=18, y=1.05)
    
# 畫出不同list之間的差別
def plot_result_list(df, score_type, name_list, title, labels=None):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 5)

    if score_type != 'Loss':
        query_acol = f'A{score_type}'
        query_scol = f'S{score_type}'
    else:
        query_acol = f'{score_type}'
        query_scol = f'{score_type}'
        
    for idx, name in enumerate(name_list):
        df_sel = df[df['name'] == name]
        x = df_sel['Epoch']
        
        if labels == None:
            axs[0].plot(x, df_sel[query_acol], label = name)    
            axs[0].legend()
        else:
            axs[0].plot(x, df_sel[query_acol], label = labels[idx])    
            axs[0].legend()            
    axs[0].set_title('Axis [0, 0]')
        
    for idx, name in enumerate(name_list):
        df_sel = df[df['name'] == name]
        x = df_sel['Epoch']
        
        if labels == None:
            axs[1].plot(x, df_sel[query_scol], label = name)    
            axs[1].legend()
        else:
            axs[1].plot(x, df_sel[query_scol], label = labels[idx])    
            axs[1].legend()            
        
    # for name in name_list:        
    #     df_sel = df[df['name'] == name]
    #     x = df_sel['Epoch']
    #     axs[1].plot(x, df_sel[query_scol], label = name)    
    #     axs[1].legend()
    
    # axs[0].set_ylim([0.72, 0.75])
    # axs[1].set_ylim([0.79, 0.82])
    axs[0].set_title(f'Aspect {score_type}', fontsize=16)
    axs[1].set_title(f'Sentiment {score_type}', fontsize=16)
    
    axs[0].set_facecolor('white')
    axs[1].set_facecolor('white')

    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel=score_type)

    fig.suptitle(f'{title} {score_type}', fontsize=18, y=1.05)

    
def read_mverlist(mver_l,Round_total,pdir, epoch_filter = None):
    all_data = pd.DataFrame()
    for mver in mver_l:
        df_temp = read_result(mver, Round_total, pdir)
        all_data = pd.concat([all_data, df_temp])
        
    if epoch_filter != None:
        all_data = all_data[all_data['Epoch'] <= epoch_filter]
        
    return all_data
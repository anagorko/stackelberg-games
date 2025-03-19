import os
import pandas
import seaborn
import matplotlib.pyplot as plt

def load_result(file_path: str) -> pandas.DataFrame:
    file = open(file_path, 'r')
    header = file.readline().split(';')
    df = pandas.read_csv(file, sep=';')
    df['network_type'] = header[0].strip()
    return df
    
plots_dir_path = 'output/plots/'
os.makedirs(plots_dir_path, exist_ok=True)

"""
Figures from San Francisco experiments
"""

res_dir_path = 'output/sf_results/'
df = pandas.concat([load_result(res_dir_path + file) for file in os.listdir(res_dir_path)])
df['plot'] =  df['network_type'].astype(str)

palette = 'colorblind'
graph_size_label = 'The number of subgraph nodes'
presented_stats = ['lp_lower_bound', 'mc_same', 'mc_plusone', 'mc_minusone']
presented_labels = ['LP', 'MC h', 'MC h+1', 'MC h-1']
df['stat'] = df['stat'].replace(presented_stats, presented_labels)

for plot_name in df['plot'].unique():
    data = df.loc[(df['plot'] == plot_name) & (df['stat'].isin(presented_labels))]

    seaborn.set(font_scale=2.)
    seaborn.set_style("white")

    fig, ax = plt.subplots(figsize=(12, 6))
    mc_plot = seaborn.lineplot(data, x='graph_size', y='value', hue='stat', style='stat', palette=palette, linewidth=3)
    mc_plot.set(ylabel='Defender utility', xlabel=graph_size_label)
    mc_plot.legend_.set_title(None)
    plt.savefig(plots_dir_path + plot_name + '_mc.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)


"""
Figures from random experiments
"""

res_dir_path = 'output/random_results/'
df = pandas.concat([load_result(res_dir_path + file) for file in os.listdir(res_dir_path)])
df['plot'] =  df['network_type'].astype(str)

palette = 'colorblind'
graph_size_label = 'The number of graph nodes'
observation_label = 'Observation length'
attack_time_label = 'Attack time'

for plot_name in df['plot'].unique():
    data = df.loc[df['plot'] == plot_name]
    data = data.rename(columns={'observation' : observation_label, 'attack_time' : attack_time_label})

    seaborn.set(font_scale=2.)
    seaborn.set_style("white")
    
    fig, ax = plt.subplots(ncols=2, figsize=(15, 6))
    fig.tight_layout()
    utility_plot = seaborn.lineplot(data[data['stat'] == 'lp_lower_bound'], ax=ax[0], x='graph_size', y='value', hue=observation_label, style=attack_time_label, palette=palette, markers=True, markersize=12, linewidth=2.)
    utility_plot.set(ylabel='Defender utility', xlabel=graph_size_label)
    runtime_plot = seaborn.lineplot(data[data['stat'] == 'lp_runtime'], ax=ax[1], x='graph_size', y='value', hue=observation_label, style=attack_time_label, palette=palette, markers=True, markersize=10, linewidth=2.)
    runtime_plot.set(yscale='log', ylabel='Runtime [s]', xlabel=graph_size_label)
    seaborn.despine()

    for a in ax:
        a.get_legend().remove()

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc='lower left', ncol=4, bbox_to_anchor=(.125,-.175))
    fig.legend(handles[4:], labels[4:], loc='lower left', ncol=3, bbox_to_anchor=(.125,-.275))

    plt.savefig(plots_dir_path + plot_name + '_util_time.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
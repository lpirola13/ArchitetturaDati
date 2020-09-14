import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import recordlinkage as rl
from recordlinkage.compare import String
import time
from recordlinkage.preprocessing import clean, phonetic
from sklearn import metrics
import statistics


def weighted_sum(row):
    return 0.75 * row['Company_score'] + 0.25 * row['Sector_score']


def merge_results(df, dfa, dfb, compare_threshold):
    merge = dfa.reindex(df.index, level=1).join(dfb.reindex(df.index, level=0))
    merge['Score'] = df.apply(weighted_sum, axis=1)
    merge = merge.sort_values('Score', ascending=False)
    merge['Score'] = merge['Score'].apply(lambda x: 1 if x >= compare_threshold else 0)
    merge = merge[['Security', 'Company', 'GICS Sector', 'Sector', 'Score']].drop_duplicates(subset=['Company'],
                                                                                             keep='first')
    merge = merge.reset_index()
    merge = merge.sort_values('level_1')
    return merge


if __name__ == "__main__":
    # READ CSV
    sp500 = pd.read_csv('sp500.csv')
    forbes = pd.read_csv('forbes.csv')
    true = pd.read_csv('trueindex.csv')

    # PREPROCESSING SP500
    len(sp500)
    sp500.head()
    fig_sp500 = plt.figure(figsize=(9, 4))
    heatmap_sp500 = sns.heatmap(sp500.isnull(), cbar=False)
    plt.title("SP500: Missing Values ")
    plt.savefig("MV-sp500.png", dpi=700, bbox_inches='tight')
    plt.show()
    sp500['Date first added'].isnull().sum()

    sp500['GICS Sector'].unique()
    sp500['GICS Sub Industry'].unique()

    sp500_pre = sp500.copy()
    sp500_pre['GICS Sector'] = clean(sp500_pre['GICS Sector'])
    sp500_pre['GICS Sub Industry'] = clean(sp500_pre['GICS Sub Industry'])
    sp500_pre['Security'] = clean(sp500_pre['Security'])
    sp500_pre["Security"].str.count(r'\bcorp\b').sum()
    sp500_pre['Security'] = sp500_pre["Security"].str.replace(r'\bcorp\b', '', regex=True)
    sp500_pre["Security"].str.count(r'\binc\b').sum()
    sp500_pre['Security'] = sp500_pre["Security"].str.replace(r'\binc\b', '', regex=True)
    sp500_pre["Security"].str.count(r'\bco\b').sum()
    sp500_pre['Security'] = sp500_pre["Security"].str.replace(r'\bco\b', '', regex=True)
    sp500_pre["Security"].str.count(r'\bltdp\b').sum()
    sp500_pre['Security'] = sp500_pre["Security"].str.replace(r'\bltd\b', '', regex=True)
    sp500_pre["Security"].str.count(r'\bplc\b').sum()
    sp500_pre['Security'] = sp500_pre["Security"].str.replace(r'\bplc\b', '', regex=True)
    sp500_pre["Security"].str.count(r'\bhldgp\b').sum()
    sp500_pre['Security'] = sp500_pre["Security"].str.replace(r'\bhldg\b', '', regex=True)
    sp500_pre["Security"].str.count(r'\bpcl\b').sum()
    sp500_pre['Security'] = sp500_pre["Security"].str.replace(r'\bplc\b', '', regex=True)
    sp500_pre["Security"].str.count(r'\bthe\b').sum()
    sp500_pre['Security'] = sp500_pre["Security"].str.replace(r'\bthe\b', '', regex=True)

    # PREPROCESSING FORBES
    len(forbes)
    forbes.head()
    fig_forbes = plt.figure(figsize=(9, 4))
    heatmap_forbes = sns.heatmap(forbes.iloc[:, 1:10].isnull(), cbar=False)
    plt.title("Forbes: Missing Values ")
    plt.savefig("MV-forbes.png", dpi=700, bbox_inches='tight')
    plt.show()
    forbes['Sector'].isnull().sum()
    forbes['Industry'].isnull().sum()

    forbes['Sector'].unique()
    forbes['Industry'].unique()

    forbes_pre = forbes.copy()
    forbes_pre['Sector'] = clean(forbes_pre['Sector'])
    forbes_pre['Industry'] = clean(forbes_pre['Industry'])
    forbes_pre['Company'] = clean(forbes_pre['Company'])
    forbes_pre["Company"].str.count(r'\bcorp\b').sum()
    forbes_pre['Company'] = forbes_pre["Company"].str.replace(r'\bcorp\b', '', regex=True)
    forbes_pre["Company"].str.count(r'\binc\b').sum()
    forbes_pre['Company'] = forbes_pre["Company"].str.replace(r'\binc\b', '', regex=True)
    forbes_pre["Company"].str.count(r'\bco\b').sum()
    forbes_pre['Company'] = forbes_pre["Company"].str.replace(r'\bco\b', '', regex=True)
    forbes_pre["Company"].str.count(r'\bltd\b').sum()
    forbes_pre['Company'] = forbes_pre["Company"].str.replace(r'\bltd\b', '', regex=True)
    forbes_pre["Company"].str.count(r'\bplc\b').sum()
    forbes_pre['Company'] = forbes_pre["Company"].str.replace(r'\bplc\b', '', regex=True)
    forbes_pre["Company"].str.count(r'\bhldg\b').sum()
    forbes_pre['Company'] = forbes_pre["Company"].str.replace(r'\bhldg\b', '', regex=True)
    forbes_pre["Company"].str.count(r'\bpcl\b').sum()
    forbes_pre['Company'] = forbes_pre["Company"].str.replace(r'\bplc\b', '', regex=True)
    forbes_pre["Company"].str.count(r'\bthe\b').sum()
    forbes_pre['Company'] = forbes_pre["Company"].str.replace(r'\bthe\b', '', regex=True)

    sp500_pre['Security_sort'] = phonetic(sp500_pre['Security'], method="soundex")
    forbes_pre['Company_sort'] = phonetic(forbes_pre['Company'], method="soundex")

    len(forbes_pre[forbes_pre['Industry'].isna() | forbes_pre['Sector'].isna()])

    # FULL
    indexer = rl.Index()
    indexer.full()
    candidates = indexer.index(forbes_pre, sp500_pre)
    print(len(candidates))

    # SORTED NEIGHBORHOOD
    indexer = rl.Index()
    indexer.sortedneighbourhood(left_on="Security_sort", right_on="Company_sort", window=31)
    candidates = indexer.index(sp500_pre, forbes_pre)
    print(len(candidates))

    mean_fmeasures = []
    mean_times = []
    methods = ['jaro', 'jarowinkler', 'levenshtein', 'damerau_levenshtein', 'smith_waterman', 'lcs']
    names = ['Jaro', 'Jaro-Winkler', 'Levenshtein', 'Damerau-Levenshtein', 'Smith-Waterman', 'LCS']

    i = 0
    for method in methods:
        print(method)
        thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        precisions = []
        recalls = []
        fmeasures = []
        times = []
        for threshold in thresholds:
            compare = rl.Compare([
                String('Security', 'Company', method=method, label="Company_score"),
                String('GICS Sector', 'Sector', method=method, label="Sector_score")
            ])
            start = time.time()
            features = compare.compute(candidates, sp500_pre, forbes_pre)
            end = time.time()
            times.append(end - start)
            matches = merge_results(features, forbes, sp500, threshold)
            precisions.append(metrics.precision_score(true['Score'], matches['Score']))
            recalls.append(metrics.recall_score(true['Score'], matches['Score']))
            fmeasures.append(metrics.f1_score(true['Score'], matches['Score']))
        mean_fmeasures.append(statistics.mean(fmeasures))
        mean_times.append(statistics.mean(times))
        fig = plt.figure(figsize=(7, 4))
        ax1 = fig.add_subplot(111)
        ax1.plot(thresholds, precisions, label='Precision', color='c', linestyle='--')
        ax1.plot(thresholds, recalls, label='Recall', color='g', linestyle='--')
        ax1.plot(thresholds, fmeasures, label='F-measure', color='r', )
        plt.xticks(thresholds)
        plt.xlabel('Threshold')
        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.15, 1))
        plt.title('Performance: ' + names[i])
        plt.savefig(methods[i] + ".png", dpi=700, bbox_inches='tight')
        i = i + 1

    fig_times = plt.figure(figsize=(8, 4))
    y_pos = np.arange(len(names))
    plt.bar(y_pos, mean_times, align='center', alpha=0.5)
    plt.xticks(y_pos, names)
    plt.xticks(rotation=90)
    plt.ylabel('time (s)')
    plt.title('AVG Execution time')
    plt.savefig("times.png", dpi=700, bbox_inches='tight')

    fig_avgfmeasures = plt.figure(figsize=(8, 4))
    ax = fig_avgfmeasures.add_subplot(1, 1, 1)
    mean_fmeasures = [round(x, 3) for x in mean_fmeasures]
    table_data = pd.DataFrame(mean_fmeasures, index=names)
    table = ax.table(cellText=table_data.values, rowLabels=table_data.index, loc='center')
    table.set_fontsize(12)
    table.scale(1, 3)
    ax.axis('off')
    fig_avgfmeasures.suptitle('AVG F-measure', fontsize=14)
    plt.savefig("avgfmeasure.png", dpi=700, bbox_inches='tight')

    fig_avgtimes = plt.figure(figsize=(8, 4))
    ax = fig_avgtimes.add_subplot(1, 1, 1)
    mean_times = [round(x, 3) for x in mean_times]
    table_data = pd.DataFrame(mean_times, index=names)
    table = ax.table(cellText=table_data.values, rowLabels=table_data.index, loc='center')
    table.set_fontsize(12)
    table.scale(0.5, 3)
    ax.axis('off')
    fig_avgtimes.suptitle('AVG Execution Time', fontsize=14)
    plt.savefig("avgtime.png", dpi=700, bbox_inches='tight')

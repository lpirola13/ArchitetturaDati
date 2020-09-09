import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import recordlinkage as rl
from recordlinkage.compare import String
import time
from recordlinkage.preprocessing import clean, phonetic


def weighted_sum(row):
    return 0.75 * row['Company_score'] + 0.25 * row['Sector_score']


def merge_results(df, dfa, dfb):
    merge = dfa.reindex(features.index, level=1).join(dfb.reindex(features.index, level=0))
    merge['Score'] = df.apply(weighted_sum, axis=1)
    merge = merge.sort_values('Score', ascending=False)
    merge[['Security', 'Company', 'GICS Sector', 'Sector', 'Score']].drop_duplicates(subset=['Security'], keep='first')
    return merge


def make_confusion_matrix(matches, true):
    matches = matches[['Security', 'Company', 'GICS Sector', 'Sector', 'Score']].drop_duplicates(subset=['Security'],keep='first')
    matches = matches.reset_index()
    matches.loc[454:len(matches), 'level_0'] = 9999
    matches = matches.set_index(['level_0', 'level_1'])
    matches = matches.reset_index()
    merge = pd.merge(true, matches, left_on='sp500', right_on='level_1')
    merge.loc[(merge['forbes'] == merge['level_0']) & ((merge['forbes'] != 9999) & (merge['level_0'] != 9999)), 'Pred']= 'TP'
    merge.loc[(merge['forbes'] != merge['level_0']) & ((merge['forbes'] != 9999) & (merge['level_0'] != 9999)), 'Pred']= 'FN'
    merge.loc[(merge['forbes'] == 9999) & (merge['level_0'] == 9999), 'Pred']= 'TN'
    merge.loc[(merge['forbes'] != 9999) & (merge['level_0'] == 9999), 'Pred'] = 'FN'
    merge.loc[(merge['forbes'] == 9999) & (merge['level_0'] != 9999), 'Pred']= 'FP'
    print(merge['Pred'].value_counts())
    values = merge['Pred'].value_counts()
    recall = values[0]/(values[0] + values[1])
    precision = values[0]/(values[0] + values[2])
    fmeasure = 2 * (recall * precision) / (recall + precision)
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('f-measure: ' + str(fmeasure))
    return precision, recall, fmeasure


if __name__ == "__main__":
    # READ CSV
    sp500 = pd.read_csv('sp500.csv')
    forbes = pd.read_csv('forbes.csv')
    true = pd.read_csv('true.csv', sep=';')
    true_columns = list(true.columns)
    true_columns[0] = 'forbes'
    true_columns[1] = 'sp500'
    true.columns = true_columns

    # PREPROCESSING SP500
    len(sp500)
    sp500.head()
    fig = plt.figure(figsize=(9, 4))
    heatmap = sns.heatmap(sp500.isnull(), cbar=False)
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
    fig = plt.figure(figsize=(9, 4))
    heatmap = sns.heatmap(forbes.iloc[:, 1:10].isnull(), cbar=False)
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


    sp500_pre['Security_sort'] = phonetic(sp500_pre['Security'], method="metaphone")
    forbes_pre['Company_sort'] = phonetic(forbes_pre['Company'], method="metaphone")

    len(forbes_pre[forbes_pre['Industry'].isna() | forbes_pre['Sector'].isna()])

    # FULL
    indexer = rl.Index()
    indexer.full()
    candidates = indexer.index(forbes_pre, sp500_pre)
    print(len(candidates))

    # SORTED NEIGHBORHOOD
    indexer = rl.Index()
    indexer.sortedneighbourhood(left_on="Company_sort", right_on="Security_sort", window=13)
    candidates = indexer.index(forbes_pre, sp500_pre)
    print(len(candidates))

    times = np.zeros(6)
    precision = np.zeros(6)
    recall = np.zeros(6)
    fmeasure = np.zeros(6)


    # JARO
    compare = rl.Compare([
        String('Company', 'Security', method='jaro', label="Company_score"),
        String('Sector', 'GICS Sector', method='jaro', label="Sector_score")
    ])
    start = time.time()
    features = compare.compute(candidates, forbes_pre, sp500_pre)
    end = time.time()
    times[0] = end - start
    print("elapsed: " + str(times[0]))
    matches = merge_results(features, sp500, forbes)
    precision[0], recall[0], fmeasure[0] = make_confusion_matrix(matches, true)

    # JARO-WINKLER
    compare = rl.Compare([
        String('Company', 'Security', method='jarowinkler', label="Company_score"),
        String('Sector', 'GICS Sector', method='jarowinkler', label="Sector_score")
    ])
    start = time.time()
    features = compare.compute(candidates, forbes_pre, sp500_pre)
    end = time.time()
    times[1] = end - start
    print("elapsed: " + str(times[1]))
    matches = merge_results(features, sp500, forbes)
    precision[1], recall[1], fmeasure[1] = make_confusion_matrix(matches, true)

    # LEVENSHTEIN
    compare = rl.Compare([
        String('Company', 'Security', method='levenshtein', label="Company_score"),
        String('Sector', 'GICS Sector', method='levenshtein', label="Sector_score")
    ])
    start = time.time()
    features = compare.compute(candidates, forbes_pre, sp500_pre)
    end = time.time()
    times[2] = end - start
    print("elapsed: " + str(times[2]))
    matches = merge_results(features, sp500, forbes)
    precision[2], recall[2], fmeasure[2] = make_confusion_matrix(matches, true)

    # DAMERAU-LEVENSHTEIN
    compare = rl.Compare([
        String('Company', 'Security', method='damerau_levenshtein', label="Company_score"),
        String('Sector', 'GICS Sector', method='damerau_levenshtein', label="Sector_score")
    ])
    start = time.time()
    features = compare.compute(candidates, forbes_pre, sp500_pre)
    end = time.time()
    times[3] = end - start
    print("elapsed: " + str(times[3]))
    matches = merge_results(features, sp500, forbes)
    precision[3], recall[3], fmeasure[3] = make_confusion_matrix(matches, true)

    # Q-GRAM
    compare = rl.Compare([
        String('Company', 'Security', method='qgram', label="Company_score"),
        String('Sector', 'GICS Sector', method='qgram', label="Sector_score")
    ])
    start = time.time()
    features = compare.compute(candidates, forbes_pre, sp500_pre)
    end = time.time()
    times[4] = end - start
    print("elapsed: " + str(times[4]))
    matches = merge_results(features, sp500, forbes)
    precision[4], recall[4], fmeasure[4] = make_confusion_matrix(matches, true)

    # COSINE
    compare = rl.Compare([
        String('Company', 'Security', method='cosine', label="Company_score"),
        String('Sector', 'GICS Sector', method='cosine', label="Sector_score")
    ])
    start = time.time()
    features = compare.compute(candidates, forbes_pre, sp500_pre)
    end = time.time()
    times[5] = end - start
    print("elapsed: " + str(times[5]))
    matches = merge_results(features, sp500, forbes)
    precision[5], recall[5], fmeasure[5] = make_confusion_matrix(matches, true)

    objects = ('Jaro', 'Jaro-Winkler', 'Levenshtein', 'Damerau-Levenshtein', 'Q-Gram', 'Cosine')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, times, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.xticks(rotation=90)
    plt.ylabel('time (s)')
    plt.title('Execution time')
    plt.show()

    measures = pd.DataFrame({"Precision": precision
                           ,"Recall": recall
                           , "F-measure": fmeasure}
                           ,index=['Jaro', 'Jaro-Winkler', 'Levenshtein', 'Damerau-Levenshtein', 'Q-Gram', 'Cosine'])

    ax = measures.plot.bar()
    ax.set_ylim([0.9, 1])
    plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import recordlinkage as rl
from recordlinkage.compare import String
from recordlinkage.preprocessing import clean, phonetic


def weighted_sum(row):
    return 0.75 * row['Company_score'] + 0.25 * row['Sector_score']


def merge_results(df, dfa, dfb):
    merge = dfa.reindex(features.index, level=1).join(dfb.reindex(features.index, level=0))
    merge['Score'] = df.apply(weighted_sum, axis=1)
    merge = merge.sort_values('Score', ascending=False)
    return merge


if __name__ == "__main__":
    # READ CSV
    sp500 = pd.read_csv('sp500.csv')
    forbes = pd.read_csv('forbes.csv')

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

    sp500_pre['Security_sort'] = phonetic(sp500_pre['Security'], method="metaphone")
    forbes_pre['Company_sort'] = phonetic(forbes_pre['Company'], method="metaphone")

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

    # JARO-WINKLER
    compare = rl.Compare([
        String('Company', 'Security', method='jarowinkler', label="Company_score"),
        String('Sector', 'GICS Sector', method='jarowinkler', label="Sector_score")
    ])
    features = compare.compute(candidates, forbes_pre, sp500_pre)
    matches = merge_results(features, sp500, forbes)
    matches[['Security', 'Company', 'GICS Sector', 'Sector', 'Score']].drop_duplicates(subset=['Security'],
                                                                                       keep='first').to_csv(
        'jaro-winkler.csv')

    # DAMERAU-LEVENSHTEIN
    compare = rl.Compare([
        String('Company', 'Security', method='damerau_levenshtein', label="Company_score"),
        String('Sector', 'GICS Sector', method='damerau_levenshtein', label="Sector_score")
    ])
    features = compare.compute(candidates, forbes_pre, sp500_pre)
    matches = merge_results(features, sp500, forbes)
    matches[['Security', 'Company', 'GICS Sector', 'Sector', 'Score']].drop_duplicates(subset=['Security'],
                                                                                       keep='first').to_csv(
        'damerau-levenshtein.csv')

    # Q-GRAM
    compare = rl.Compare([
        String('Company', 'Security', method='qgram', label="Company_score"),
        String('Sector', 'GICS Sector', method='qgram', label="Sector_score")
    ])
    features = compare.compute(candidates, forbes_pre, sp500_pre)
    matches = merge_results(features, sp500, forbes)
    matches[['Security', 'Company', 'GICS Sector', 'Sector', 'Score']].drop_duplicates(subset=['Security'],
                                                                                       keep='first').to_csv(
        'q-gram.csv')

    # SMITH-WATERMAN
    compare = rl.Compare([
        String('Company', 'Security', method='smith_waterman', label="Company_score"),
        String('Sector', 'GICS Sector', method='smith_waterman', label="Sector_score")
    ])
    features = compare.compute(candidates, forbes_pre, sp500_pre)
    matches = merge_results(features, sp500, forbes)
    matches[['Security', 'Company', 'GICS Sector', 'Sector', 'Score']].drop_duplicates(subset=['Security'],
                                                                                       keep='first').to_csv(
        'smith-waterman.csv')

    # LCS
    compare = rl.Compare([
        String('Company', 'Security', method='lcs', label="Company_score"),
        String('Sector', 'GICS Sector', method='lcs', label="Sector_score")
    ])
    features = compare.compute(candidates, forbes_pre, sp500_pre)
    matches = merge_results(features, sp500, forbes)
    matches[['Security', 'Company', 'GICS Sector', 'Sector', 'Score']].drop_duplicates(subset=['Security'],
                                                                                       keep='first').to_csv('lcs.csv')

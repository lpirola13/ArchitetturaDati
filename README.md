# Comparazione di misure di similarità tra stringhe nell'ambito del record linkage 
[![made-with-python](https://img.shields.io/badge/MADE%20WITH-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

<div align="justify">
Lo scopo di questo progetto è quello di confrontare l’efficienza di diversi metodi per il confronto approssimativo di stringhe applicate nell’ambito del record 
linkage. Il progetto è basato sull’articolo "A Comparison of Personal Name Matching: Techniques and Practical Issues" di P.Christen, nel quale le diverse tecniche
sono utilizzate con l’obiettivo specifico di confrontare i nomi di persona. In questo progetto le misure verranno utilizzate per confrontare nomi commerciali di 
società. Più precisamente, le misure verranno utilizzate per identificare quali tra le 2000 aziende più grandi e influenti al mondo secondo Forbes sono anche 
presenti nell’indice di borsa statunitense Standard & Poor 500. I vari records sono stati uniti attraverso un record linkage basato su soglia, mentre le 
prestazioni dei singoli metodi sono state valutate grazie a un terzo dataset, il quale è stato costruito per essere etichettato come ground truth.
</div>

## Requirements
Per installare i requisiti:

    pip install -r requirements.txt
    
## Usage
Per eseguire il confronto:

    python3 script.py

## Data
The datasets used are:
* [forbes.csv](forbes.csv) - classifica delle 2000 aziende più influenti al mondo stilata da Forbes.
* [sp500.csv](sp500.csv) - aziende pubbliche americane quotate in borsa e appartenenti all’indice S&P 500.
* [trueindex.csv](trueindex.csv) - ground truth dataset.

## References
* P. Christen. A comparison of personal name matching: Techniques and practical issues. In Sixth IEEE International Conference on Data Mining-Workshops (ICDMW’06). IEEE, 2006.

## Authors
* Lorenzo Pirola &nbsp;
[![gmail](https://img.shields.io/badge/Gmail-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:l.pirola13@campus.unimib.it) &nbsp;
[![github](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)](https://github.com/lpirola13) &nbsp;
[![linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lorenzo-pirola-230275197/)

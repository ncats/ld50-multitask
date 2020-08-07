# Large-scale Modeling of Multi-Species Acute Toxicity Endpoints using Consensus of Multi-Task Deep Learning Methods
This repository contains multitask deep learning models developed using acute toxicity data, primarily focusing on the endpoints: lethal dose fifty (LD<sub>50</sub>); lethal dose low (LDLO); and toxic dose low (TDLO). Please note that the data was obtained from [ChemIDPlus](https://chem.nlm.nih.gov/chemidplus/). However, neither ChemIDplus nor NLM endorse the developed models and the associated publication.

## Results

Our best models are based on a consensus of best developed individual models. We compared our best models against the multi-task deep learning models by [Sosnin *et al*.](https://doi.org/10.1021/acs.jcim.8b00685) While they report models for a total of 29 toxicity endpoints, our models are based on a total of 59 endpoints. A total of 18 LD<sub>50</sub> endpoints were in common. The results for these 18 endpoints are listed below. The performance measure reported is root mean squared error (lower is better).


species | route | cpds (ours) | cpds (Sosnin *et al*) | score (ours) | score<sup>a</sup> (Sosnin *et al*) |
| :---: | :---: | :---: | :---: | :---: | :---: |
mouse | intraperitoneal | 36295 | 37202 | 0.41 | 0.41 |
mouse | oral | 23373 | 24355 | 0.39 | 0.42 |
mouse | intravenous | 16978 | 17742 | 0.43 | 0.43 |
rat | oral | 10190 | 10743 | 0.52 | 0.63 |
mouse | subcutaneous | 6769 | 7221 | 0.51 | 0.51 |
rat | intraperitoneal | 5021 | 5041 | 0.52 | 0.53 |
rat | intravenous | 2472 | 2538 | 0.52 | 0.64 |
rat | subcutaneous | 1896 | 2014 | 0.63 | 0.58 |
mouse | unreported | 1739 | 1804 | 0.47 | 0.51 |
rabbit | skin | 1495 | 1734 | 0.53 | 0.54 |
mammal<sup>b</sup>| unreported | 1129 | 1121 | 0.42 | 0.40 |
rabbit | oral | 894 | 910 | 0.58 | 0.43 |
rat | skin | 835 | 930 | 0.61 | 0.68 |
rat | unreported | 806 | 838 | 0.58 | 0.56 |
guinea pig | oral | 793 | 799 | 0.59 | 0.55 |
rabbit | intravenous | 792 | 764 | 0.66 | 0.70 |
rat | oral | 322 | 966 | 0.63 | 0.60 |
rat | intraperitoneal | 318 | 1029 | 0.52 | 0.61

<sup>a</sup> the scores are from the supplementary information of the original article; <sup>b</sup> the mammalian species and route are unspecified

## Other Models

We also report single-task models using baseline methods: random forest and deep neural networks. The scripts used for modeling can be found under `scripts\`. An example `notebooks/create_fold_data.ipynb` to create the training and test sets by joining the descriptors and task details for different folds of cross-validation is provided.


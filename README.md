# xETM (Cross-lingual Embedded Topic Model)
## 1 Dependency
Install the required dependencies using the following commond. A fresh environment is recommended.

```
pip install -r requirements.txt
```

## 2 Usage

The demo dataset can be downloaded from [here](https://www.dropbox.com/s/6h55c58phhn2gmo/xETM-demo-dataset.zip?dl=0).

- Step 1, `test_preprocessing.ipynb` formats the dataset into the required format for xETM. The formatted dataset is put in `out/` folder.
- Step 2, `run.py` trains the model.
```
# simultaneously learn word embedding and topic embedding 
python run.py --data_path out --train_embeddings 1 --mode train --epochs 50
```
- Step 3, `run.py` gets learned topic words and infers topic distributions of given documents.
```
python run.py --data_path out --train_embeddings 1 --mode eval --load_from results/D_300_K_50_Epo_50_Opt_adam
```

## Miscellanea
- `data_20ng.ipynb` is the notebook used for testing the data preparation codes from the [original repo](https://github.com/adjidieng/ETM/blob/master/scripts/data_20ng.py).
- `test_data_io.ipynb` is the notebook used for testing data batch generator.
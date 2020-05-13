"""
Split functions are based on chemprop (https://github.com/wengong-jin/chemprop)

To get chemprop, run the following command:

git clone https://github.com/wengong-jin/chemprop

Note: This script file must be inside the chemprop folder. Also it could take a while for the script to run 
as it is used to process about 80,000 compounds and their target values.

"""

import pandas as pd
from chemprop.data import utils
from random import randint

def get_partition_as_df(partition, target_names):
	dict = {'RTECS_ID': partition.compound_names(), 'SMILES': partition.smiles(), 'targets': partition.targets()}
	df = pd.DataFrame(dict)
	df2 = pd.DataFrame(df)
	df2[target_names] = pd.DataFrame(df2.targets.values.tolist(), index= df2.index)
	df2.drop('targets', axis=1, inplace=True)
	return df2


def random_split(dataset, seed_val):
	# get target names (assuming that 1st column contains molecule names and 2nd column contains smiles and rest of the columns are targets)
	df = pd.read_csv(dataset, sep=",", index_col=None, dtype={'RTECS_ID': str})
	cols = list(df.columns)
	target_names = cols[2:]

	mol_dataset = utils.get_data(dataset, use_compound_names=True)
	train, valid, test = utils.split_data(mol_dataset, sizes=(0.7, 0.1, 0.2), seed=seed_val)
	train_df = get_partition_as_df(train, target_names)
	train_df = train_df[['RTECS_ID']]
	valid_df = get_partition_as_df(valid, target_names)
	valid_df = valid_df[['RTECS_ID']]
	test_df = get_partition_as_df(test, target_names)
	test_df = test_df[['RTECS_ID']]
	return train_df, valid_df, test_df

def scaffold_split(dataset, seed):
	# get target names (assuming that 1st column contains molecule names and 2nd column contains smiles and rest of the columns are targets)
	df = pd.read_csv(dataset, sep=",", index_col=None, dtype={'RTECS_ID': str})
	cols = list(df.columns)
	target_names = cols[2:]

	mol_dataset = utils.get_data(dataset, use_compound_names=True)
	train, valid, test = utils.split_data(mol_dataset, split_type="scaffold_balanced", sizes=(0.7, 0.1, 0.2), seed=seed_val)
	train_df = get_partition_as_df(train, target_names)
	train_df = train_df[['RTECS_ID']]
	valid_df = get_partition_as_df(valid, target_names)
	valid_df = valid_df[['RTECS_ID']]
	test_df = get_partition_as_df(test, target_names)
	test_df = test_df[['RTECS_ID']]
	return train_df, valid_df, test_df


# example usage to split the dataset 5 times

dataset = "../data/full_dataset.csv"

out_dir = '../data/random_split/'

print('started splitting!')

for k in range(5):
	print('fold: %s' % k)
	seed_val = randint(1, 50000)
	print('using seed: %s' % seed_val)
	train, valid1, valid2 = random_split(dataset, seed_val) # change split type and see value here
	test = pd.concat([valid1, valid2], ignore_index=True, sort =False)
	train_file_path = out_dir + 'train_fold_'+ str(k) +'.csv'
	test_file_path = out_dir + 'test_fold_'+ str(k) +'.csv'
	train.to_csv(train_file_path, index=None)
	test.to_csv(test_file_path, index=None)
	print(train.shape)
	print(test.shape)

print('finished splitting!')
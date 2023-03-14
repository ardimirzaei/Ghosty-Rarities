# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:16:29 2022

@author: ArdiMirzaei
"""

#%%

import pandas as pd
from tqdm import tqdm
import os

#%%
json_files = os.listdir('metadata/')

col_names = ['Name'
            , 'Body'
            , 'Eyes'
            , 'Head'
            , 'Mouth'
            , 'Accessories'
            ]

all_jsons = pd.DataFrame(columns = col_names)

for j in tqdm(json_files):
    blank_df = {}
    f = pd.read_json(f'metadata/{j}')
    blank_df['Name'] = pd.unique(f.name)[0]
    for _attr in f['attributes']:
        blank_df[_attr['trait_type']] = _attr['value']
    
    all_jsons = all_jsons.append(pd.DataFrame.from_dict(blank_df, orient = 'index').T)
    
# Convert to dataframe    
all_jsons.reset_index(inplace = True, drop = True)

#%%
# Dummy variable to create a wide dataframe. 
ghosty_dummies = pd.get_dummies(all_jsons, columns = [c for c in col_names if "Name" not in c])
ghosty_dummies.index = ghosty_dummies['Name']
ghosty_dummies.drop('Name', axis = 1, inplace = True)
#%%
# Sum over the column and divide by the number of Ghosts in the series. 
feautures_scored = ghosty_dummies.sum(axis =0).sort_values(ascending = False)/len(json_files) # should be 1000



#%%
# Multiple by the feature by a score. 
for _col in tqdm(ghosty_dummies.columns):
    ghosty_dummies[_col]  = ghosty_dummies[_col] * feautures_scored[_col]

#%%
# Sum of all the features
ghosts_ranked = ghosty_dummies.sum(axis = 1).sort_values()

#%%
# Top 10, lower is better (rarer)
top_10 = all_jsons.merge(pd.DataFrame(ghosts_ranked[:10]).reset_index(), how = 'right', left_on='Name', right_on='Name')

top_10.columns = [c for c in top_10.columns[:-1]] + ['Rarity Score']


#%%

all_ghosts_ranked = all_jsons.merge(pd.DataFrame(ghosts_ranked.reset_index()), how = 'right', left_on='Name', right_on='Name')
all_ghosts_ranked.columns = [c for c in all_ghosts_ranked.columns[:-1]] + ['Rarity Score']

# Change index to be ranked by feature rarity. 
all_ghosts_ranked.index = all_ghosts_ranked['Rarity Score'].rank(method = 'min')
all_ghosts_ranked.index.name = 'Rank'

# Export to CSV
all_ghosts_ranked.to_csv('GhostyDropGME_FeatureRanks.csv')

#%%
# ARCHIVE
# 
all_jsons.iloc[:,1:].sum(axis = 1).value_counts()

import pandas as pd
import numpy as np

# had to edit first cleaning script to include YearBuilt
ames = pd.read_csv("data/housing_clean_ed.csv")

# Cut out anything over 4,000 sq ft 
# Drops two outliers
ames.drop(ames[ames.GrLivArea >= 4000].index, inplace=True)

# Transform ordinal categorical to numerical
ames_rank = ames.filter(regex='Qual$|QC$|Qu$|Cond$').drop(['OverallQual','OverallCond'], axis=1)
ames_rank.replace({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
ames[ames_rank.columns] =ames_rank

# Change month to str to make categorical
ames.MoSold = ames.MoSold.astype(str)

# Create Age Features
ames['HouseAge'] = ames.YrSold - ames.YearBuilt
ames['RemodelAge'] = ames.YrSold - ames.YearRemodAdd

# deal with some houses do not have GarageYrBlt
ames.GarageYrBlt = ames.GarageYrBlt.replace('None', 100000).astype(float)
ames['GarageAge'] = ames.YrSold - ames.GarageYrBlt
ames.loc[ames.GarageAge < 0, 'GarageAge'] = 'None'

# drop year columns now that we have age features
ames = ames.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'],axis=1)

# create has Y/N features 
# could prob use dictionary and mapping here
ames['HasBsmt'] = np.where(ames['TotalBsmtSF']==0, 0, 1)
ames['HasFireplace'] = np.where(ames['Fireplaces']==0, 0, 1)
ames['HasGarage'] = np.where(ames['GarageArea']==0, 0, 1)
ames['HasWoodDeck'] = np.where(ames['WoodDeckSF']==0, 0, 1)
ames['HasOpenPorch'] = np.where(ames['OpenPorchSF']==0, 0, 1)
ames['HasEncPorch'] = np.where(ames['EnclosedPorch']==0, 0, 1)
ames['Has3SsnPorch'] = np.where(ames['3SsnPorch']==0, 0, 1)
ames['HasScreenPorch'] = np.where(ames['ScreenPorch']==0, 0, 1)
ames['HasPool'] = np.where(ames['PoolArea']==0, 0, 1)
ames['HasFence'] = np.where(ames['Fence']=='None', 0, 1)

# Create Porch Y/N so pot get rid of all other Porch Y/N features above
ames['HasPorch'] = ames.OpenPorchSF + ames.EnclosedPorch + ames['3SsnPorch'] + ames.ScreenPorch
ames['HasPorch'] = np.where(ames['HasPorch']==0, 0, 1)

# create totsf that includes basement and garage
ames['TotSF'] = ames.GrLivArea + ames.TotalBsmtSF + ames.GarageArea

# create Price per sqft feature
ames['PriceSqft'] = ames.SalePrice / ames.GrLivArea

# Save csv
ames.to_csv("data\housing_clean2.csv",index_label = False)
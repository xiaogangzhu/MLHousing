
def clean(df):
    import pandas as pd
    import numpy as np
    select_column = ["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtHalfBath","BsmtFullBath", "MasVnrArea"]
    df[select_column] = df[select_column].fillna(0)

    # Electric 
    df.Electrical = df.Electrical.fillna(df.Electrical.mode()[0])

    # Lotfrontage average
    df.LotFrontage = df.LotFrontage.fillna(np.mean(df.LotFrontage))
    # GarageCars need to impute 0
    df.GarageCars = df.GarageCars.fillna(0)
    # GarageArea with 0
    df.GarageArea = df.GarageArea.fillna(0)

    # # Impute median for yr built houses without garages
    df.GarageYrBlt = df.GarageYrBlt.astype(float).fillna(df.GarageYrBlt.median())

    # Fill others with None
    df = df.fillna("None")

    df.drop('PID', axis=1, inplace = True)

    # Transform ordinal categorical to numerical
    df_rank = df.filter(regex='Qual$|QC$|Qu$|Cond$').drop(['OverallQual','OverallCond'], axis=1)
    df_rank.replace({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    df[df_rank.columns] = df_rank

    # Change month to str to make categorical
    df.MoSold = df.MoSold.astype(str)

    return df

def clean_category_var(df):
    import pandas as pd
    import numpy as np
    df.BldgType = np.where(df.BldgType == '1Fam','1',np.where(df.BldgType == 'TwnhsE','2','3'))
    df.BsmtExposure = np.where(df.BsmtExposure == 'Gd',"1",np.where(df.BsmtExposure == 'None',"2","3")) 
    df.BsmtFinType1 = np.where(df.BsmtFinType1 == 'GLQ',"1",np.where(df.BsmtFinType1 == 'None',"2","3"))
    df.Condition1 = np.where(df.Condition1.isin(['PosN','PosA','RRAn','RRNn']) ,"1","2")
    df.Electrical = np.where(df.Electrical == 'SBrkr',"1","2")
    df.Foundation = np.where(df.Foundation == 'PConc',"1","2")
    df.Functional = np.where(df.Functional == 'Typ',"1","2")
    df.GarageFinish = np.where(df.GarageFinish.isin(['Rfn','Fin']),"1","2")
    df.GarageType = np.where(df.GarageType.isin(['Attchd','BuiltIn']),"1","2")
    df.HouseStyle = np.where(df.HouseStyle.isin(['1.5Fin','1.5Unf','1Story']),"1",np.where(df.HouseStyle.isin(['2Story','2.5Unf','2.5Fin']),"2","3"))
    df.LandSlope = np.where(df.LandSlope == 'Gtl',"1","2")
    df.LotShape = np.where(df.LotShape == 'Reg',"1","2")
    df.MSZoning = np.where(df.MSZoning.isin(['RL','FV']),"1","2")
    df.PavedDrive = np.where(df.PavedDrive == 'Y',"1","2")
    df.SaleCondition = np.where(df.SaleCondition == 'Partial',"1","2")
    df['Neighborhood'] = np.where(df['Neighborhood'].isin(['GrnHill', 'Greens', 'NridgHt', 'StoneBr', 'Veenker', 'Somerst',
       'Timber', 'CollgCr', 'Blmngtn']), "Tier_1", np.where(df['Neighborhood'].isin(['Blmngtn', 'NoRidge', 'Mitchel', 'ClearCr',
       'Blueste', 'Sawyer', 'Crawfor', 'SawyerW', 'Gilbert', 'NPkVill']), "Tier_2", "Tier_3"))
    
    #drop columns
    drop_columns = ["Alley","BsmtFinType2","Condition2","Exterior2nd","Fence","Heating","LandContour","LotConfig",
                    "MiscFeature","MoSold","RoofMatl","RoofStyle","SaleType"]
    df = df.drop(drop_columns,axis=1)
    
    return df
    

def j_model_clean(df):
    import pandas as pd
    import numpy as np
    df['HouseAge'] = df.YrSold - df.YearBuilt
    df['TotalArea'] = df['GrLivArea'] + df['TotalBsmtSF'] + df['GarageArea']
    df['TotalBath'] = df['HalfBath']/2 + df['BsmtFullBath'] + df['BsmtHalfBath']/2 + df['FullBath']
    df['MultStory'] = np.where(df['HouseStyle']=='1Story', 0, 1)
    df['PartialSale'] = np.where(df['SaleCondition']=='Partial', 1, 0)
    df['PavedDrive'] = np.where(df['PavedDrive']=='Y', 1, 0)
    df['Neighborhood'] = np.where(df['Neighborhood'].isin(['GrnHill', 'Greens', 'NridgHt', 'StoneBr', 'Veenker', 'Somerst',
       'Timber', 'CollgCr', 'Blmngtn']), "Tier_1", np.where(df['Neighborhood'].isin(['Blmngtn', 'NoRidge', 'Mitchel', 'ClearCr',
       'Blueste', 'Sawyer', 'Crawfor', 'SawyerW', 'Gilbert', 'NPkVill']), "Tier_2", "Tier_3"))
    column_names = ['TotalArea','Neighborhood','BldgType','ExterQual','BsmtQual','HeatingQC','CentralAir','BedroomAbvGr',
                    'KitchenQual','Fireplaces','GarageQual','PavedDrive','HouseAge','TotalBath', 'MultStory', 'PartialSale']
    df = df[column_names].copy()
    df['TotalArea'] = np.log(df['TotalArea'].copy())

    return df
    
def final_model(df):
    import numpy as np
    df['TotalBath'] = df['HalfBath']/2 + df['BsmtFullBath'] + df['BsmtHalfBath']/2 + df['FullBath']
    df['PartialSale'] = np.where(df['SaleCondition']=='Partial', 1, 0)
    df['PavedDrive'] = np.where(df['PavedDrive']=='Y', 1, 0)
    df['Neighborhood'] = np.where(df['Neighborhood'].isin(['GrnHill', 'Greens', 'NridgHt', 'StoneBr', 'Veenker', 'Somerst',
       'Timber', 'CollgCr', 'Blmngtn']), "Tier_1", np.where(df['Neighborhood'].isin(['Blmngtn', 'NoRidge', 'Mitchel', 'ClearCr',
       'Blueste', 'Sawyer', 'Crawfor', 'SawyerW', 'Gilbert', 'NPkVill']), "Tier_2", "Tier_3"))
    column_names = ['ExterQual','BsmtQual','KitchenQual','Fireplaces','GarageQual','PavedDrive',
                    'TotalBath','PartialSale','TotalBsmtSF','1stFlrSF','2ndFlrSF','GarageArea','WoodDeckSF']
    df = df[column_names].copy()

    return df

def transform_age(df):
    df['HouseAge'] = df.YrSold - df.YearBuilt
    df['RemodelAge'] = df.YrSold - df.YearRemodAdd
    
    return df

def j_model_clean(df):
    import pandas as pd
    import numpy as np
    df['HouseAge'] = df.YrSold - df.YearBuilt
    df['TotalArea'] = df['GrLivArea'] + df['TotalBsmtSF'] + df['GarageArea']
    df['TotalBath'] = df['HalfBath']/2 + df['BsmtFullBath'] + df['BsmtHalfBath']/2 + df['FullBath']
    df['MultStory'] = np.where(df['HouseStyle']=='1Story', 0, 1)
    df['PartialSale'] = np.where(df['SaleCondition']=='Partial', 1, 0)
    df['PavedDrive'] = np.where(df['PavedDrive']=='Y', 1, 0)
    df['Neighborhood'] = np.where(df['Neighborhood'].isin(['GrnHill', 'Greens', 'NridgHt', 'StoneBr', 'Veenker', 'Somerst',
       'Timber', 'CollgCr', 'Blmngtn']), "Tier_1", np.where(df['Neighborhood'].isin(['Blmngtn', 'NoRidge', 'Mitchel', 'ClearCr',
       'Blueste', 'Sawyer', 'Crawfor', 'SawyerW', 'Gilbert', 'NPkVill']), "Tier_2", "Tier_3"))
    column_names = ['TotalArea','Neighborhood','BldgType','ExterQual','BsmtQual','HeatingQC','CentralAir','BedroomAbvGr',
                    'KitchenQual','Fireplaces','GarageQual','PavedDrive','HouseAge','TotalBath', 'MultStory', 'PartialSale']
    df = df[column_names].copy()
    df['TotalArea'] = np.log(df['TotalArea'].copy())

    return df
    
def final_clean1(df):
    import numpy as np
    df['HouseAge'] = df.YrSold - df.YearBuilt
    df['RemodelAge'] = df.YrSold - df.YearRemodAdd
    df['TotalBath'] = df['HalfBath']/2 + df['BsmtFullBath'] + df['BsmtHalfBath']/2 + df['FullBath']
    df['MultStory'] = np.where(df['HouseStyle']=='1Story', 0, 1)
    df['PartialSale'] = np.where(df['SaleCondition']=='Partial', 1, 0)
    df['PavedDrive'] = np.where(df['PavedDrive']=='Y', 1, 0)
    df['HasDeck'] = np.where(df['WoodDeckSF']==0, 0, 1)
    df['NoPorch'] = df.OpenPorchSF + df.EnclosedPorch + df['3SsnPorch'] + df.ScreenPorch
    df['NoPorch'] = np.where(df['NoPorch']==0, 1, 0)
    df['Neighborhood'] = np.where(df['Neighborhood'].isin(['GrnHill', 'Greens', 'NridgHt', 'StoneBr', 'Veenker', 'Somerst',
       'Timber', 'CollgCr', 'Blmngtn']), "Tier_1", np.where(df['Neighborhood'].isin(['Blmngtn', 'NoRidge', 'Mitchel', 'ClearCr',
       'Blueste', 'Sawyer', 'Crawfor', 'SawyerW', 'Gilbert', 'NPkVill']), "Tier_2", "Tier_3"))
    column_names = ['Neighborhood','BldgType','ExterQual','BsmtQual','HeatingQC','CentralAir','BedroomAbvGr',
                    'KitchenQual','Fireplaces','GarageQual','PavedDrive','HouseAge','TotalBath', 'MultStory',
                    'PartialSale','TotalBsmtSF','1stFlrSF','2ndFlrSF','GarageArea','WoodDeckSF','OpenPorchSF']
    df = df[column_names].copy()

    return df

def final_choose(df):
    
    column_names = ['Neighborhood','BldgType','ExterQual','BsmtQual','CentralAir','BedroomAbvGr',
                    'KitchenQual','Fireplaces','GarageQual','PavedDrive','HouseAge','TotalBath',
                    'PartialSale','TotalBsmtSF','1stFlrSF','2ndFlrSF','GarageArea','WoodDeckSF']
    df = df[column_names].copy()
    
    return df



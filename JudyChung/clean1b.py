
def clean(df):
    import pandas as pd
    import numpy as np
    select_column = ["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtHalfBath","BsmtFullBath", "MasVnrArea"]
    df[select_column] = df[select_column].fillna(0)

    # Electric 
    df.Electrical = df.Electrical.fillna(df.Electrical.mode()[0])

    # Lotfrontage average
    df.LotFrontage = df.LotFrontage.fillna(np.mean(df.LotFrontage)).copy()
    # GarageCars need to impute 0
    df.GarageCars = df.GarageCars.fillna(0).copy()
    # GarageArea with 0
    df.GarageArea = df.GarageArea.fillna(0).copy()

    # # Impute median for yr built houses without garages
    df.GarageYrBlt = df.GarageYrBlt.astype(float).fillna(df.GarageYrBlt.median()).copy()

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

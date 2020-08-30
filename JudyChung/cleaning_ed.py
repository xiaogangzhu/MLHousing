import pandas as pd


Ames = pd.read_csv("data/Ames_HousePrice.csv", index_col=0)
Ames = Ames.reset_index(drop = True)
# Garage
Ames = Ames.drop([433,531])
# Basement
Ames.loc[[813,1201],"BsmtExposure"] = "No"
select_column = ["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtHalfBath","BsmtFullBath"]
Ames[select_column] = Ames[select_column].fillna(0)
# Masonry Veneer
Ames.MasVnrArea = Ames.MasVnrArea.fillna(0)
# Electric 
Ames.Electrical = Ames.Electrical.fillna(Ames.Electrical.mode()[0])
# Fill others with None
Ames = Ames.fillna("None")
# drop columns and repeative rows
Ames = Ames.drop(1816,axis=0)
drop_columns = ["Alley", "PID", "MSSubClass","MSZoning","LotFrontage","LotArea","Street","LotShape","LandContour",
                      "LotConfig","LandSlope","Neighborhood","Condition1","Condition2","SaleType","SaleCondition"]

Ames = Ames.drop(drop_columns,axis=1)
# Save csv
Ames = Ames.reset_index(drop = True)
Ames.to_csv("data\housing_clean.csv",index_label = False)

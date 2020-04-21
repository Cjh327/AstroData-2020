# In[0]
import zipfile
import pandas as pd

# In[1]
zFile = zipfile.ZipFile("/mnt/data3/caojh/dataset/AstroData/validation/compressed/val_sets_v1_full.zip", "r")
#ZipFile.namelist(): 获取ZIP文档内所有文件的名称列表
# In[2]
pd.read_csv(zFile.open("val_sets_v1.csv"), header=None)
print(pd)
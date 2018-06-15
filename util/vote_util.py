import pandas as pd
import numpy as np
from glob import glob


ending={}
files=glob("*.csv")


for i in range(len(files)):
	print(files[i])
	f=files[i]
	end=pd.read_csv(f,header=None,sep=' ')[0].get_values()
	for x in end:
		if(str(x) in ending):
			ending[str(x)]+=1
		else:
			ending[str(x)]=1



endingg=[]
for key,value in ending.items():
	if value>=len(files)/2:
		endingg.append(key)

print(len(endingg))
pd.DataFrame(np.array(endingg).reshape(-1,1))[0].to_csv("fuse_6_4.csv",index=False)
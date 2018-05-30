import os
import sys
sys.path.append("../../")
from tqdm import tqdm
import numpy as np
import pandas as pd
from active_users.util.load_data import load_data
from active_users.util.vectorizer import vec

def gen_vec_data():

    if os.path.exists('../original_data/x_30.npy'):
        x=np.load('../original_data/x_30.npy')
        ids=np.load('../original_data/id_30.npy')
        print(np.shape(x))
        return x,ids
    else:
        r, l, c, a = load_data()
        x=[]
        author_id = list(a['author_id'].get_values())
        for index in tqdm(r.index):
            user_id = r.loc[index]['user_id']
            v = vec(
                r.loc[index].get_values(),
                l.loc[l.user_id == user_id].get_values(),
                c.loc[c.user_id == user_id].get_values(),
                a.loc[a.user_id == user_id].get_values(),
                author_id,
                31)
            x.append(v)
        x=np.array(x)
        ids=r['user_id'].get_values()
        np.save('../original_data/x_30',x)
        np.save('../original_data/id_30',ids)
        print(np.shape(x))
        return x,ids

    # print(r.info())
    # print(l.info())
    # print(c.info())
    # print(a.info())


def xgb_predict(X):
    import xgboost as xgb
    bst= xgb.Booster() #init model  
    bst.load_model("../model/529xgb.model") # load data  
    D = xgb.DMatrix(X)
    result=bst.predict(D)
    print(len(result))
    return result

if __name__ == '__main__':
	x,ids=gen_vec_data()
	result=xgb_predict(x)
	ending=[]
	for i in range (len(result)):
		if result[i]>=0.5:
			ending.append(ids[i])
	pd.DataFrame(np.array(ending).reshape(-1,1))[0].to_csv('529xgb.csv',index=False)


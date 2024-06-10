import os
import numpy as np
import pandas as pd

def preprocess_episode(data_path, horizon, hop):
    print("Preprocessing episodes..")
    episode_list = []
    for file in os.listdir(data_path):
        if file.endswith("csv"):
            df = pd.read_csv(os.path.join(data_path, file)).sort_values(by='Date').reset_index(drop=True)
            df['Volume'] = df['Volume'].replace({0: df['Volume'].mean()})
            df['Volume'] = df['Volume'].apply(lambda x: np.log(x))
            df['Volume'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            for idx in range((df.shape[0]-horizon)//hop):
                episode_df = df.iloc[idx*hop:idx*hop+horizon]
                episode_base_price = episode_df['Open'].iloc[0]
                for price_key in ['Open', 'High', 'Low', 'Close']:
                    episode_df[price_key] = (episode_df[price_key] - episode_base_price) / episode_base_price * 100
                episode_list.append(episode_df.to_numpy())
    
    print(f'Completed preprocessing episodes..\nepisode_list len: {len(episode_list)}')
    
    return np.stack(episode_list, axis=0)
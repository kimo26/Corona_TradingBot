import pandas as pd
import numpy as np
import random
from glob import glob
from collections import deque
from sklearn import preprocessing
import pickle
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
global X,y
scaler = MinMaxScaler()

X=[]
y=[]
def classify(current,future):
    pc = 100*(future-current)/current
    if abs(pc) > 1:
        if current<future:
            return 1
        return 0
dis = ['rsi']
def preprocess_df(df):
    global X,y
    for col in df.columns:
        if col != 'class':
            print(col)
            scaler = MinMaxScaler()
            if col not in dis:
                df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] =  scaler.fit_transform(np.array(df[col]).reshape(-1,1))
            joblib.dump(scaler, f'scaler_{col}.pkl') 
    df.dropna(inplace=True)
    

    sequential_data = []
    prev_days = deque(maxlen=60)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == 60:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)
    
    buys = []
    sells = []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])


    random.shuffle(buys)
    random.shuffle(sells)
    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells
    random.shuffle(sequential_data) 

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
        
def rsi(price, n=14):
    ''' rsi indicator '''
    gain = (price-price.shift(1)).fillna(0) # calculate price gain with previous day, first row nan is filled with 0

    def rsiCalc(p):
        # subfunction for calculating rsi for one lookback period
        avgGain = p[p>0].sum()/n
        avgLoss = -p[p<0].sum()/n 
        rs = avgGain/avgLoss
        return 100 - 100/(1+rs)

    # run for all periods with rolling_apply
    return gain.rolling(n).apply(rsiCalc)         

def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean()
    
    # Create MACD
    dataset['26ema'] = dataset['Close'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Close'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])
    dataset['signal'] = dataset['MACD'].rolling(window=9).mean()
    dataset['MACD_signal'] = df[['MACD', 'signal']].apply(lambda row: (row.iloc[0]-row.iloc[1])/row.iloc[0], axis=1)

    # Create Bollinger Bands
    dataset['20sd'] = dataset['Close'].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    dataset['upper_band'] = df[['Close', 'upper_band']].apply(lambda row: (row.iloc[0]-row.iloc[1])/row.iloc[0], axis=1)
    dataset['lower_band'] = df[['Close', 'lower_band']].apply(lambda row: (row.iloc[0]-row.iloc[1])/row.iloc[0], axis=1)
    # Create Exponential moving average
    dataset['ema7'] = dataset['Close'].ewm(span=7,adjust=False).mean()
    dataset['ema21'] = dataset['Close'].ewm(span=21,adjust=False).mean()

    #RSI
    dataset['rsi'] = rsi(dataset['Close'])
    
    return dataset
#for file in glob('stock/^GSPC.csv'):
for i in range(1):
    file = 'stock/^GSPC (1).csv'
    df = pd.read_csv(file)
    df['Volume'] =df['Volume'].replace(0,np.nan)
    df=df.drop('Adj Close',1)
    df = get_technical_indicators(df)
    df.dropna(inplace=True)
    df=df.drop('Date',1)
    
    df['future'] = df['Close'].shift(-3)
    df['class'] = list(map(classify,df['Close'],df['future']))
    df = df.drop('future',1)
    df.fillna(method='ffill')
    df.dropna(inplace=True)
    preprocess_df(df)
X=np.array(X)
y=np.array(y)
with open('data.pickle','wb') as f:
    pickle.dump([X,y],f)
print(X.shape)


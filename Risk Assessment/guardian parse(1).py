import requests
import bs4 as bs
import urllib
import pandas as pd
import datetime as dt
import yfinance as yf
stock = yf.Ticker('^GSPC')
inv=[5,6]
def soups(url):
    req = requests.get(url, proxies=urllib.request.getproxies())
    soup = bs.BeautifulSoup(req.text,'lxml')
    return soup
def date_clean(date):
    months = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}
    date=date.split('-')
    day = int(date[0])
    month=months[date[1]]
    year = int(date[-1])
    return dt.date(year,month,day)
def vola(date):
    weekday = date.weekday()
    if weekday == 5:
        date += dt.timedelta(days=-1)
    elif weekday == 6:
        date+=dt.timedelta(days=-2)
    tomorrow = date + dt.timedelta(days=1)
    hist = stock.history(interval='60m',start=date,end=tomorrow)
    return hist['Close'].std()
def classify(vol):
    if 10<vol:
        return 1
    return 0
data = {}
for i in range(1,350):
    soup = soups(f'https://www.theguardian.com/us-news/us-politics?page={i}')
    for batch in soup.findAll('section'):
        try:
            date=date_clean(batch['data-component'])
            data[date]=[]
            data[date].append(vola(date))
            titles=[]
            for article in batch.findAll('h3'):
                title = article.text.replace('\n','')
                titles.append(title)
            data[date].append(' '.join(titles))
        except:pass
past = []
cleaned=[]
for date in data:
    if date.weekday() == 4:
        past.append(float(data[date][0]))
        past.append(data[date][1])
    elif date.weekday() == 5:
        past.append(data[date][1])
    elif date.weekday() == 6:
        past.append(data[date][1])
        if len(past) == 4:
            titles=[]
            full= ' '.join([i for i in past if type(i) != float])
            volat = data[date][0]
            cleaned.append([volat,full])
        past=[]
    else:
        cleaned.append(data[date])
    
df = pd.DataFrame()
df['vola']=[i[0] for i in cleaned]
df['titles']= [i[1] for i in cleaned]
df.fillna(method='ffill')
df['future_vola']= df['vola'].shift(-1)
df.dropna(inplace=True)
df['target'] = list(map(classify, df['future_vola']))

df = df.drop('vola',1)
df = df.drop('future_vola',1)
good = []
bad=[]
for i,j in zip(df['titles'],df['target']):
    if j==0:
        bad.append(i)
    else:
        good.append(i)

size = min(len(good),len(bad))
good = good[:size]
bad = bad[:size]
gen = good + bad
targets = [1 for _ in range(size)]+[0 for _ in range(size)]
df = pd.DataFrame()
df['target'] = targets
df['titles']=gen
print(df.head(n=10))
df.to_csv('news_data.csv')

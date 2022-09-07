import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm # 폰트매니저 라이브러리
import matplotlib.pyplot as plt

from pykrx import stock

from tqdm import tqdm # 반복문 진행상황 출력해주는 라이브러리
from scipy.spatial.distance import euclidean # euclidean거리 구하는 라이브러리
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer # 정규화 라이브러리
from collections import Counter # 클러스터 내 종목 수 카운트 해주는 라이브러리

def open_json(year):
    year = str(year)
    with open('./datasets/names_'+year+'.json', 'r', encoding='utf-8') as f:
        codes = json.load(f) # 종목코드 불러오기 
    return codes

def open_csv(year):
    year = str(year)
    df = pd.read_csv('./datasets/'+year+'close_clu.csv', dtype={'code': str}, index_col=0) # 주가데이터 불러오기
    return df

def make_array(df, codes):
    # 일일주가 데이터에서 클러스터링에 필요한 벡터 추출한다.
    a = len(df[df.code == '005930']) -19
    X = []
    names = []
    w = 20 # 이동평균기간(ex.20일 평균)
    n = a + w - 1 # 해당 주식의 총기간(ex. 각 종목당 139일 주가데이터가 있다.)
    for code_item in tqdm(codes):
        code = code_item['code'] # 코드 불러온다
        _df = df[df['code'] == code].dropna() # 주가데이터에서 불러온 코드에 결측치가 있으면 제거한다.
        if len(_df) < n: # 데이터 프레임의 길이가 n보다 작으면 다음 단계로 넘어간다.
            continue
        _df = _df.sort_values(by='date', ascending=True) # 데이터 프레임을 날짜기준 오름차순 정렬한다.(과거->현재)
        _df['x'] = _df['close'] / _df.iloc[0]['close'] - 1 # 초기값 대비 수익률을 구한다.
        _df = _df.dropna() # 결측치 제거한다.
        if len(_df) < n: # 데이터 프레임의 길이가 n보다 작으면 다음 단계로 넘어간다.
            continue
        _df['x'] = pd.Series(_df['x']).rolling(w).mean() #  수익률의 w일 이동평균을 구한다.
        X.append(_df['x'].tolist()[w-1:]) # w일 이동평균 값을 리스트화 한다.
        names.append(code_item['name']) # 종목명 리스트를 만든다.
    X = np.array(X) # 이동평균값 배열을 만든다.
    names = np.array(names) # 종목명 배열을 만든다.
    return X, names
    # X : 2320종목의 각 종목 당 120일 동안의 20일 이동평균
    # names : 2300종목의 종목명

# 클러스터 결과 시각화 함수
def plot_clusters(com, model, k, X, names): # model = 사용할 분류모델(kmeans), k = 클러스터 개수
    if com == 'window':   
        # 한글 폰트 가져오기(Window용)
        fm.get_fontconfig_fonts()
        font_location = 'C:\\Windows\\Fonts\\NanumGothic.ttf'
        font_name = fm.FontProperties(fname=font_location).get_name()
        matplotlib.rc('font', family=font_name)
        matplotlib.rc('axes', unicode_minus=False)
    elif com == 'mac':
        # 한극 폰트 가져오기(Mac용)
        matplotlib.rcParams['font.family'] = 'AppleGothic'
        matplotlib.rcParams['axes.unicode_minus'] = False

    # subplot '1번 컬럼' 부분 시각화
    c = 5 
    fig, axes = plt.subplots(nrows=k, ncols=c+1, figsize=(c*3, k*2))
    for label in range(k): # 기준이 되는 클러스터 지정(k개), 첫 번째 컬럼 그래프 그리기, 학습된 클러스터 중심점
        x = model.cluster_centers_[label] # x = 클러스터 센터 / model.cluster_centers : 클러스터를 특정할 벡터(중심점)
        axes[label][0].plot(np.arange(len(x)), x) # x축 - 길이:120, y축 - 클러스터를 특정하는 벡터(중심점)
        axes[label][0].set_title(f'{label}: Cent.') # 타이틀
        axes[label][0].set_ylim([min(x.min() - 0.01, -0.1), max(x.max() + 0.01, 0.1)]) # y축 범위 지정

        # 2번  ~ 6번 컬럼 시각화
        n = 0
        _X = X[model.labels_ == label] # model.labels_ : 클러스터의 번호 / label = 클러스터 번호
        _names = names[model.labels_ == label] # 클러스터 번호에 해당하는 종목가져 불러오기
        dists = [euclidean(x, _x) for _x in _X] # 클러스터의 중심( x = model.cluster_centers_)과 X 데이터 간의 'Euclidean'거리 구하기
        idxs = np.argsort(dists)[:c] # argsort 배열을 오름차순으로 정렬하는 함수
        for x, name in zip(_X[idxs], _names[idxs]): # 각 클러스터에 해당하는 종목들 시각화 
            n += 1
            axes[label][n].plot(np.arange(len(x)), x) # x축 - 길이:120, y축 - 클러스터를 특정하는 벡터(중심점)
            axes[label][n].set_title(f'{label}: {name}') # 타이틀
            axes[label][n].set_ylim([min(x.min() - 0.01, -0.1), max(x.max() + 0.01, 0.1)]) # y축 범위 지정
            if n >= c:
                break
    plt.tight_layout()
    plt.show()

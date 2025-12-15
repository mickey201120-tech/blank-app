import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import datetime

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="AI 股價分析", page_icon="📈", layout="wide")
st.title("📈 AI 智能股價趨勢分析")
st.markdown("針對 **通訊/AI 所** 書審設計：結合時間序列與線性回歸的展示專案")

# --- 2. 側邊欄輸入 ---
st.sidebar.header("設定")
# 預設台積電，讓它一定有資料抓
default_ticker = "2330.TW"
ticker = st.sidebar.text_input("輸入股票代號 (台股請加 .TW)", value=default_ticker)

# 日期設定
start_date = st.sidebar.date_input("開始日期", datetime.date.today() - datetime.timedelta(days=365))
end_date = st.sidebar.date_input("結束日期", datetime.date.today())

# --- 3. 抓取資料函數 (加強錯誤處理) ---
@st.cache_data
def load_data(symbol, start, end):
    try:
        # 下載資料
        df = yf.download(symbol, start=start, end=end)
        
        # 檢查資料是否為空
        if df.empty:
            return None
        
        # --- 關鍵修正：處理 yfinance 可能的多層索引問題 ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        return None

# 執行抓取
data = load_data(ticker, start_date, end_date)

# --- 4. 主要邏輯區 ---
if data is None or data.empty:
    st.error(f"⚠️ 找不到股票代號 `{ticker}` 的資料，請確認代號是否正確 (例如台股要打 `2330.TW`)，或檢查網路連線。")
else:
    # 確保 Date 欄位是 datetime 格式
    data['Date'] = pd.to_datetime(data['Date'])
    
    # 簡單的特徵工程 (Feature Engineering)
    # 計算均線 (Moving Average) -> 訊號處理觀念
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()

    # --- AI 核心：線性回歸預測 ---
    # 準備訓練資料 (X=時間序列, y=股價)
    data['Date_ID'] = data['Date'].apply(lambda x: x.toordinal())
    
    # 移除有空值 (NaN) 的資料以免報錯
    train_data = data.dropna(subset=['Close', 'Date_ID'])
    
    X = train_data[['Date_ID']].values
    y = train_data['Close'].values
    
    if len(X) > 0:
        model = LinearRegression()
        model.fit(X, y)
        
        # 產生預測線 (Trend Line)
        trend_pred = model.predict(X)
        slope = model.coef_[0] # 斜率
    else:
        trend_pred = []
        slope = 0

    # --- 5. 顯示指標 ---
    latest_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    diff = latest_price - prev_price
    diff_pct = (diff / prev_price) * 100

    col1, col2 = st.columns(2)
    col1.metric("最新收盤價", f"{latest_price:.2f}", f"{diff:.2f} ({diff_pct:.2f}%)")
    
    trend_str = "📈 長期看漲" if slope > 0 else "📉 長期看跌"
    col2.metric("AI 趨勢判讀 (線性回歸)", trend_str, f"斜率: {slope:.4f}")

    # --- 6. 畫圖 (Plotly) ---
    st.subheader(f"📊 {ticker} 股價走勢圖")
    
    fig = go.Figure()

    # K線圖
    fig.add_trace(go.Candlestick(
        x=data['Date'],
        open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'],
        name='K線'
    ))

    # 均線
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA5'], line=dict(color='orange', width=1), name='MA 5 (短線)'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA20'], line=dict(color='blue', width=1), name='MA 20 (長線)'))

    # 趨勢線
    if len(trend_pred) > 0:
        fig.add_trace(go.Scatter(x=train_data['Date'], y=trend_pred, line=dict(color='red', width=2, dash='dash'), name='AI 趨勢線'))

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 顯示原始資料
    with st.expander("查看詳細數據"):
        st.dataframe(data.sort_values('Date', ascending=False))

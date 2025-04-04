import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from GoogleNews import GoogleNews
from datetime import datetime
import pytz


st.set_page_config(
    page_title = "Stock Analytics Dashboard",
    page_icon=":bar_chart:",
    # layout="wide"
)

# Sidebar Header
st.sidebar.header("Filter Selection")

# Sidebar Stock Selection
stocks = ["NVDA", "META", "PLTR", "GOOGL", "SHOP", "TSLA", "AMZN"]
selected_stock = st.sidebar.selectbox("Stock Ticker Selection", stocks)

# App Title
st.title(f"{selected_stock} Stock Analytics Dashboard")


# Sidebar Data Selection
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

@st.cache_data
def fetch_stock_data(stock, start, end):
    # Calculate the extended start date (300 days before the selected start date)
    extended_start = pd.to_datetime(start) - pd.Timedelta(days=300)
    # Fetch data from the extended date
    df = yf.download(stock, start=extended_start, end=end)
    
    # Get stock info and financials
    ticker = yf.Ticker(stock)
    stock_info = ticker.info

    print(f'stock_info: {stock_info}')

    # Get specific financial data we need
    financial_data = {
        'financials': ticker.financials.loc['Total Revenue'].to_dict() if not ticker.financials.loc['Total Revenue'].empty else {},
        'quarterly_financials': ticker.quarterly_financials.loc['Total Revenue'].to_dict() if not ticker.quarterly_financials.loc['Total Revenue'].empty else {},
        'earnings': ticker.financials.loc['Net Income'].to_dict() if not ticker.financials.loc['Net Income'].empty else {},
        'quarterly_earnings': ticker.quarterly_financials.loc['Net Income'].to_dict() if not ticker.quarterly_financials.loc['Net Income'].empty else {}
    }
    # Calculate moving averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Trim the dataframe to only show from the user-selected start date
    df = df[start:end]
    return df, stock_info, financial_data

def get_ny_time_status():
    # NY time
    ny_time_zone = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_time_zone)
    

    # Check if NY is weekday
    # This should be 0 - 4 to be weekday
    is_weekday = ny_time.weekday() < 5

    # Cehck if time is 9:30 - 4:00 pm NY time
    pre_market_start = ny_time.replace(hour=4, minute=0, second=0) # 4:00 AM ET
    market_open_time = ny_time.replace(hour=9, minute=30, second=0) # 9:30 AM ET
    market_close_time = ny_time.replace(hour=16, minute=0, second=0) # 4:00 PM ET
    is_market_hours = market_open_time <= ny_time <= market_close_time

    # Time format
    time_str = ny_time.strftime("%I:%M:%S %p Eastern Time")

    # Identify the market status
    if not is_weekday: # Market closed still shows post-market price
        status = "Market Closed (Weekend)"
        color = "red"
    elif pre_market_start <= ny_time < market_open_time:
        status = "Pre-Market"
        color = "skyblue"
    elif market_open_time <= ny_time < market_close_time:
        status = "Market Open"
        color = "green"
    elif market_close_time <= ny_time or ny_time < pre_market_start:
        status = "After Hours"
        color = "skyblue"
    return time_str, status, color

df, stock_info, financial_data = fetch_stock_data(selected_stock, start_date, end_date)


# Display market time and status
col1, col2 = st.columns(2)
time_str, market_status, color = get_ny_time_status()

with col1:
    st.markdown(f"<h3 style='margin-bottom: 0px'>{time_str}</h3>", unsafe_allow_html=True)

with col2:
    icons = {
        "Market Closed (Weekend)": "🔒",
        "Market Open": "🔆",
        "After Hours": "🌙",
        "Pre-Market": "🌅"
    }
    icon = icons.get(market_status, "")

    st.markdown(f"<h5 style='color: {color}; margin-bottom: 0px; margin-top: 12px;'>{market_status} {icon}</h5>", unsafe_allow_html=True)


# Annual and Quarter Financials
financials_dict = financial_data['financials']

financials = pd.DataFrame(list(financial_data['financials'].items()), columns=["Date", "Revenue"]).set_index('Date')
quarter_financials = pd.DataFrame(list(financial_data['quarterly_financials'].items()), columns=["Date", "QuarterRevenue"]).set_index('Date')
earnings = pd.DataFrame(list(financial_data['earnings'].items()), columns=["Date", "Earnings"]).set_index('Date')
quarter_earnings = pd.DataFrame(list(financial_data['quarterly_earnings'].items()), columns=["Date", "QuarterEarnings"]).set_index('Date')

# print(f'stock_info: {stock_info}')

# Create columns for metrics
col1, col2, col3, col4 = st.columns(4)
col5, col6, col7 = st.columns(3)
col8, col9, col10, col11 = st.columns(4)

# Current and Post Price and Change
latest_price = float(df['Close'].iloc[-1][selected_stock])
price_change = ((df['Close'].iloc[-1][selected_stock] - df['Close'].iloc[-2][selected_stock]) / df['Close'].iloc[-2][selected_stock]) * 100

# Stock Price
with col1:
    st.metric(label="Stock Price", value=round(latest_price, 2), delta=f"{price_change:.2f}%")

# Post Stock Price
with col2:
    # Only show this information if the market is in After hour
    if market_status == 'After Hours':
        post_latest_price = stock_info.get('postMarketPrice', 'N/A')
        post_price_change = stock_info.get('postMarketChange', 'N/A')
        if post_latest_price is not None and post_price_change is not None:

            st.metric(label="Post Stock Price", value=round(post_latest_price, 2), delta=f"{post_price_change:.2f}%")
        else:
            st.metric(label="Post Stock Price", value='N/A', delta=None)
    elif market_status == 'Market Closed (Weekend)':
        post_latest_price = stock_info.get('postMarketPrice', 'N/A')
        post_price_change = stock_info.get('postMarketChange', 'N/A')
        if post_latest_price is not None and post_price_change is not None:

            st.metric(label="Post Stock Price", value=round(post_latest_price, 2), delta=f"{post_price_change:.2f}%")
        else:
            st.metric(label="Post Stock Price", value='N/A', delta=None)
    elif market_status == 'Pre-Market':
        pre_latest_price = stock_info.get('preMarketPrice', 'N/A')
        pre_price_change = stock_info.get('preMarketChange', 'N/A')
        if pre_latest_price is not None and pre_price_change is not None:

            st.metric(label="Pre Stock Price", value=round(pre_latest_price, 2), delta=f"{pre_price_change:.2f}%")
        else:
            st.metric(label="Pre Stock Price", value='N/A', delta=None)
    elif market_status == 'Market Open':
        # Post-Pre Stock Price will not show up during market open
        st.metric(label="Post Stock Price", value='N/A', delta=None)

# 50-day MA comparison
with col3:
    ma50_latest = float(df['MA50'].iloc[-1])
    ma50_diff = ((latest_price - ma50_latest) / ma50_latest) * 100
    st.metric(label="50-Day MA", value=f"${ma50_latest:.2f}", delta=f"{ma50_diff:.2f}%")

# 200-day MA comparison
with col4:
    ma200_latest = float(df['MA200'].iloc[-1])
    ma200_diff = ((latest_price - ma200_latest) / ma200_latest) * 100
    st.metric(label="200-Day MA", value=f"${ma200_latest:.2f}", delta=f"{ma200_diff:.2f}%")

# PE Ratio
with col5:
    pe_ratio = stock_info.get('trailingPE', 'N/A')
    if pe_ratio != 'N/A':
        pe_ratio = f"{pe_ratio:.2f}"
    st.metric(label="P/E Ratio", value=pe_ratio)

# EPS
with col6:
    # trailing twelve months
    eps = stock_info.get('trailingEps', 'N/A')
    if eps != 'N/A':
        eps = f"${eps:.2f}"
    st.metric(label="EPS (TTM)", value=eps)

# Beta
with col7:
    beta = stock_info.get('beta', 'N/A')
    st.metric(label="Beta", value=beta)

# Annual Revenue Growth Comparison
with col8:
    latest_year_revenue = financials.iloc[0].values[0]
    previous_year_revenue = financials.iloc[1].values[0]
    annual_revenue_growth = ((latest_year_revenue - previous_year_revenue) / previous_year_revenue) * 100

    st.metric(label="Annual Revenue Growth", value=f"{annual_revenue_growth:.2f}%")

# Quarterly Revenue Growth Comparison
with col9:
    latest_quarter_revenue = quarter_financials.iloc[0].values[0]
    previous_quarter_revenue = quarter_financials.iloc[1].values[0]
    quarterly_revenue_growth = ((latest_quarter_revenue - previous_quarter_revenue) / previous_quarter_revenue) * 100
    st.metric(label="Quarter Revenue Growth", value=f"{quarterly_revenue_growth:.2f}%")

# Annual Earning Growth Comparison
with col10:
    latest_yearly_earning = earnings.iloc[0].values[0]
    previous_yearly_earning = earnings.iloc[1].values[0]
    annual_earning_growth = ((latest_yearly_earning - previous_yearly_earning) / previous_yearly_earning) * 100
    st.metric(label="Annual Earning Growth", value=f"{annual_earning_growth:.2f}%")

# Quarterly Earning Growth Comparison
with col11:
    latest_quarterly_earning = quarter_earnings.iloc[0].values[0]
    previous_quarterly_earning = quarter_earnings.iloc[1].values[0]
    quarterly_earning_growth = ((latest_quarterly_earning - previous_quarterly_earning) / previous_quarterly_earning) * 100
    st.metric(label="Quarterly Earning Growth", value=f"{quarterly_earning_growth:.2f}%")


df['MA50'] = df['MA50'].squeeze()
df['MA200'] = df['MA200'].squeeze()

# Present Data & Plot
st.write("")
st.write(f"**{selected_stock} Stock Price Trend**")
chart_data = pd.DataFrame({
    'Price': df['Close'].squeeze(),
    '200 Day MA': df['MA200'],
    '50 Day MA': df['MA50']
})
st.line_chart(chart_data)
st.caption("Daily stock price in $USD")

st.write("---")
st.write(f"**Stock Volume**")
st.line_chart(df['Volume'])
st.caption("Daily stock volume")

# Stock Comparison Section
st.write("---")
st.write("**Stock Performance Comparison**")

# Create two columns for stock selection
comp_col1, comp_col2, comp_col3_index = st.columns(3)

with comp_col1:
    compare_stock1 = st.selectbox("Select First Stock", stocks, index=stocks.index(selected_stock), key='comp1')

with comp_col2:
    # Remove the first selected stock from options to prevent duplicate selection
    remaining_stocks = [s for s in stocks if s != compare_stock1]
    compare_stock2 = st.selectbox("Select Second Stock", remaining_stocks, key='comp2')

index_stocks = ["S&P 500", "NASDAQ 100"]
with comp_col3_index:
    # Benchmark index to choose
    compare_index = st.selectbox("Select Benchmark Index", index_stocks, key='index')

@st.cache_data
def fetch_comparison_data(stock1, stock2, indexBenchmark, start, end):
    # indexBench Dict
    index_bench_dict = {"S&P 500": "^GSPC", "NASDAQ 100": "^NDX"}

    # Fetch data for both stocks
    df1 = yf.download(stock1, start=start, end=end)
    df2 = yf.download(stock2, start=start, end=end)
    df3 = yf.download(index_bench_dict[indexBenchmark], start=start, end=end)
    
    # Calculate percentage change from first day
    df1_pct = ((df1['Close'] - df1['Close'].iloc[0]) / df1['Close'].iloc[0]) * 100
    df2_pct = ((df2['Close'] - df2['Close'].iloc[0]) / df2['Close'].iloc[0]) * 100
    df3_pct = ((df3['Close'] - df3['Close'].iloc[0]) / df3['Close'].iloc[0]) * 100

    # Combine into a single dataframe with the date index
    comparison_df = pd.DataFrame({
        f'{stock1}': df1_pct.squeeze(),
        f'{stock2}': df2_pct.squeeze(),
        f'{indexBenchmark}': df3_pct.squeeze()
    }, index=df1.index) 
    
    return comparison_df

# Only start the graph if two stocks are chosen from options
if compare_stock1 and compare_stock2:
    comparison_df = fetch_comparison_data(compare_stock1, compare_stock2, compare_index, start_date, end_date)
    
    # Create comparison chart
    st.line_chart(comparison_df)
    st.caption("Price percentage change from start date (%)")


# News Section
st.write("---")
st.write(f"**Recent News for {selected_stock}**")

@st.cache_data(ttl=3600)  # Cache news for 1 hour
def fetch_stock_news(ticker):
    stock = yf.Ticker(ticker)
    return stock.news

news_results = fetch_stock_news(selected_stock)

# Display news in expandable containers
for news in news_results[:5]:  # Display top 5 news items
    news_content = news['content']
    if news_content != None:
        with st.expander(f"{news_content['title']}", expanded=False):
            st.write(f"**Date:** {news_content['pubDate']}")
            st.write(f"**Source:** {news_content['provider']['displayName']}")
            st.write(f"**Description:** {news_content.get('summary', 'No description available')}")
            if news_content['clickThroughUrl'] != None:
                st.write(f"[Read more]({news_content['clickThroughUrl']['url']})")


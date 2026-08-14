# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Helper Functions -----------------------------

def get_stock_data(tickers, period="5y"):
    """
    Download adjusted closing prices for a list of tickers.
    tickers: list of strings (e.g., ['AAPL', 'MSFT'])
    period: '1y', '3y', '5y', '10y', etc.
    Returns a pandas DataFrame with dates as index and tickers as columns.
    """
    data = yf.download(tickers, period=period, progress=False)['Close']
    return data.dropna()

def calculate_portfolio_value(prices, weights_dict):
    """
    Calculate the cumulative value of a portfolio from daily prices.
    prices: DataFrame of daily closing prices
    weights_dict: dict like {'AAPL': 3000, 'MSFT': 4000} (dollar amounts)
    Returns a pandas Series of portfolio value over time.
    """
    # Extract tickers in order
    tickers = list(weights_dict.keys())
    weights = np.array([weights_dict[t] for t in tickers])
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    # Daily returns for each stock
    returns = prices[tickers].pct_change().dropna()
    # Weighted portfolio return (sum across columns)
    port_returns = (returns * weights).sum(axis=1)
    # Compound returns: start at 1, multiply daily
    portfolio_value = (1 + port_returns).cumprod()
    return portfolio_value

def portfolio_metrics(port_value_series):
    """
    Calculate annualized return and volatility from a daily value series.
    """
    daily_returns = port_value_series.pct_change().dropna()
    annual_return = daily_returns.mean() * 252
    annual_vol = daily_returns.std() * np.sqrt(252)
    return annual_return, annual_vol

def monte_carlo_simulation(initial_value, annual_return, annual_vol, years,
                           num_simulations=1000, trading_days=252):
    """
    Simulate future portfolio value using a log-normal random walk.
    initial_value: starting portfolio value
    annual_return: expected annual return (decimal)
    annual_vol: annual volatility (decimal)
    years: projection horizon
    num_simulations: number of simulated price paths
    trading_days: assumed trading days per year (252 for stocks)
    Returns a 2D numpy array: rows = days, columns = simulations
    """
    dt = 1 / trading_days
    num_days = int(trading_days * years)
    
    # Generate random daily shocks for all simulations at once
    shocks = np.random.normal(
        (annual_return - 0.5 * annual_vol**2) * dt,
        annual_vol * np.sqrt(dt),
        (num_days, num_simulations)
    )
    
    # Cumulatively sum and exponentiate to get price paths
    price_paths = initial_value * np.exp(np.cumsum(shocks, axis=0))
    return price_paths

# ----------------------------- Streamlit App -----------------------------

# Page configuration
st.set_page_config(page_title="FutureWealth", layout="wide")
st.title("📈 FutureWealth: Portfolio Simulator & Monte Carlo Projection")
st.markdown("Build a portfolio, backtest it, and see where it could be in 10 years.")

# ---- Sidebar: User Inputs ----
st.sidebar.header("Your Portfolio")

# Text inputs for tickers and amounts
tickers_input = st.sidebar.text_input(
    "Enter tickers separated by commas",
    "AAPL, MSFT, GOOGL, JPM"
)
amounts_input = st.sidebar.text_input(
    "Enter dollar amounts (same order)",
    "3000, 4000, 2000, 1000"
)

# Historical data period
period = st.sidebar.selectbox(
    "Historical data period",
    ["1y", "3y", "5y", "10y"],
    index=2  # default "5y"
)

# Monte Carlo settings
st.sidebar.header("Monte Carlo Settings")
mc_years = st.sidebar.slider("Projection horizon (years)", 1, 30, 10)
mc_sims = st.sidebar.slider("Number of simulations", 100, 5000, 1000, step=100)

# ---- Parse user input ----
try:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    amounts = [float(a.strip()) for a in amounts_input.split(",") if a.strip()]
    
    if len(tickers) != len(amounts):
        st.error("Number of tickers and amounts must match.")
        st.stop()
    
    total_investment = sum(amounts)
    weights_dict = {t: amt for t, amt in zip(tickers, amounts)}
except:
    st.error("Please enter valid tickers and numeric amounts.")
    st.stop()

# ---- Fetch data ----
with st.spinner("Downloading market data..."):
    prices = get_stock_data(tickers, period=period)

if prices.empty:
    st.error("Could not download data. Check ticker symbols.")
    st.stop()

# ---- Backtest portfolio ----
port_value = calculate_portfolio_value(prices, weights_dict)
# Scale so the start equals the initial investment
port_value = port_value * total_investment

# ---- Benchmark: S&P 500 ----
bench_prices = get_stock_data(["SPY"], period=period)
bench_returns = (1 + bench_prices.pct_change().dropna()).cumprod()
bench_value = bench_returns * total_investment

# Combine for plotting
comparison = pd.DataFrame({
    "Your Portfolio": port_value,
    "S&P 500 (SPY)": bench_value.squeeze()
}).dropna()

# ---- Main dashboard ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Backtest Performance")
    st.line_chart(comparison)

with col2:
    st.subheader("Portfolio Metrics")
    ann_ret, ann_vol = portfolio_metrics(port_value)
    st.metric("Annualized Return", f"{ann_ret:.2%}")
    st.metric("Annualized Volatility (Risk)", f"{ann_vol:.2%}")
    
    current_value = port_value.iloc[-1]
    st.metric("Current Value", f"${current_value:,.2f}")
    st.caption(f"Initial investment: ${total_investment:,.2f}")

# ---- Monte Carlo section ----
st.header("🔮 Monte Carlo Future Projection")
st.markdown(
    f"Simulating **{mc_sims}** possible paths over **{mc_years}** years "
    f"using historical return ({ann_ret:.2%}) and volatility ({ann_vol:.2%})."
)

# Run the simulation
sim_paths = monte_carlo_simulation(
    initial_value=current_value,
    annual_return=ann_ret,
    annual_vol=ann_vol,
    years=mc_years,
    num_simulations=mc_sims
)

# Plot a subset of paths (max 200) for clarity
fig, ax = plt.subplots(figsize=(10, 5))
n_show = min(200, mc_sims)
ax.plot(sim_paths[:, :n_show], lw=0.5, alpha=0.1, color='blue')
ax.axhline(current_value, color='black', linestyle='--', label='Current Value')
ax.set_title(f"Monte Carlo Simulation: Portfolio Value after {mc_years} Years")
ax.set_xlabel("Trading Days")
ax.set_ylabel("Portfolio Value ($)")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# ---- Statistics ----
ending_values = sim_paths[-1, :]

st.subheader("Projection Statistics")
stat_col1, stat_col2, stat_col3 = st.columns(3)
stat_col1.metric("Median Final Value", f"${np.median(ending_values):,.2f}")
stat_col2.metric("5th Percentile (worst-case)", f"${np.percentile(ending_values, 5):,.2f}")
stat_col3.metric("95th Percentile (best-case)", f"${np.percentile(ending_values, 95):,.2f}")

# Probability of exceeding a target value
target = st.number_input("Target future value ($)", value=50000, step=1000)
prob = (ending_values >= target).mean() * 100
st.metric(f"Probability of ≥ ${target:,.0f}", f"{prob:.1f}%")

st.caption("Monte Carlo uses historical estimates; actual results may vary. Not financial advice.")
# dca_dashboard.py  — v1.8 (TestFol.io in Tabs & UI Refinements)
# -----------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import requests 
import json 

# --- THIS MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="DCA & Rebalance Helper", layout="wide", initial_sidebar_state="expanded")
# --- END FIRST STREAMLIT COMMAND ---

# Conditional import of yfinance only if needed
_yfinance_imported = False
def import_yfinance_if_needed():
    global _yfinance_imported
    if not _yfinance_imported:
        global yf
        import yfinance as yf
        _yfinance_imported = True

# ---------- Constants & Version Info ----------
EQ_WEIGHT_METHOD = "Equal weight"
EQ_RISK_METHOD = "Equal risk (inverse vol)" 
DEFAULT_TILT_TICKERS_LIST = ["MSFT", "GOOGL", "AMZN", "NVDA", "TSM", "SNPS", "MU", "CRWD"]
DEFAULT_TILT_TICKERS_STR = ",".join(DEFAULT_TILT_TICKERS_LIST)
TESTFOLIO_API_URL = "https://testfol.io/api/backtest"
DECIMAL_PLACES_FOR_API = 4 # Increased precision for TestFolio API

# --- Sidebar ---
with st.sidebar:
    st.title("🛠️ Settings")

    st.header("Core Portfolio Allocation")
    vti_pct_input = st.slider("VTI (%)", 0, 100, 60, key="vti_slider")
    remaining_for_vxus = 100 - vti_pct_input
    vxus_pct_input = st.slider("VXUS (%)", 0, remaining_for_vxus, min(20, remaining_for_vxus), key="vxus_slider")
    
    st.header("Tech Tilt Sleeve")
    tilt_pct_derived = 100 - vti_pct_input - vxus_pct_input
    st.metric(label="Calculated Tilt Sleeve Percentage", value=f"{tilt_pct_derived}%")

    tickers_text_input = st.text_area("Tilt tickers (comma‑separated)", DEFAULT_TILT_TICKERS_STR, height=120, key="tilt_tickers_input")
    tilt_method_input = st.radio("Tilt weighting", (EQ_WEIGHT_METHOD, EQ_RISK_METHOD), key="tilt_method_radio", horizontal=True)

    st.header("DCA & Holdings")
    dca_amount_input = st.number_input("Recurring DCA amount ($)", 0, value=5000, step=100, key="dca_amount_input")
    
    # Holdings Entry - must be dynamic based on tickers
    tilt_tickers_list_for_holdings = [t.strip().upper() for t in tickers_text_input.split(",") if t.strip()]
    base_holdings_dict_for_editor = {"VTI": 0.0, "VXUS": 0.0, **{t: 0.0 for t in tilt_tickers_list_for_holdings}}
    
    if 'holdings_df' not in st.session_state:
        st.session_state.holdings_df = pd.DataFrame(base_holdings_dict_for_editor, index=["Value"]).T
    else:
        current_data = st.session_state.holdings_df['Value'].to_dict()
        new_data = {ticker: current_data.get(ticker, 0.0) for ticker in base_holdings_dict_for_editor.keys()}
        # Remove tickers from session state that are no longer in the list
        for old_ticker in list(current_data.keys()):
            if old_ticker not in base_holdings_dict_for_editor:
                new_data.pop(old_ticker, None)
        st.session_state.holdings_df = pd.DataFrame(new_data, index=["Value"]).T.reindex(list(base_holdings_dict_for_editor.keys()), fill_value=0.0)


    edited_hold_df = st.data_editor(st.session_state.holdings_df, num_rows="dynamic", use_container_width=True, key="hold_edit")
    current_holdings_input = edited_hold_df["Value"].astype(float).to_dict()
    
    st.header("Backtest Settings (TestFol.io)")
    default_start_date = date.today() - timedelta(days=5*365) # 5 years ago
    api_start_date_input = st.date_input("Start Date", value=default_start_date, max_value=date.today() - timedelta(days=1))
    api_end_date_input = st.date_input("End Date", value=date.today(), min_value=api_start_date_input + timedelta(days=1), max_value=date.today())
    api_start_val_input = st.number_input("Starting Value ($)", 100, 1_000_000_000, 10000, 100)
    api_rebalance_freq_input = st.selectbox("Rebalance Frequency", ["Monthly", "Quarterly", "Yearly", "Never"], index=1)

    run_button = st.button("🚀 Run Calculations & Backtest", use_container_width=True, type="primary")

# ---------- Helper Functions ----------
def _norm(d):
    total = sum(d.values()); return {k: v / total for k, v in d.items()} if total else d

@st.cache_data(ttl=3600)
def yf_prices_for_inv_vol(tickers_tuple, lookback_days=252):
    import_yfinance_if_needed()
    if not tickers_tuple: return pd.DataFrame()
    end_date = date.today(); start_date = end_date - timedelta(days=int(lookback_days * 1.5))
    try: raw_data = yf.download(list(tickers_tuple), start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=False, actions=False)
    except Exception: return pd.DataFrame()
    if raw_data.empty: return pd.DataFrame()
    price_series_list = []
    for ticker in tickers_tuple:
        df_ticker = None
        if isinstance(raw_data.columns, pd.MultiIndex) and ticker in raw_data.columns.levels[0]: df_ticker = raw_data[ticker]
        elif len(tickers_tuple) == 1 and ticker in raw_data.columns : df_ticker = raw_data
        elif len(tickers_tuple) == 1 and not isinstance(raw_data.columns, pd.MultiIndex): df_ticker = raw_data
        if df_ticker is not None:
            col_to_use = 'Adj Close' if 'Adj Close' in df_ticker.columns and not df_ticker['Adj Close'].isnull().all() else 'Close'
            if col_to_use in df_ticker.columns and not df_ticker[col_to_use].isnull().all(): price_series_list.append(df_ticker[col_to_use].rename(ticker))
    if not price_series_list: return pd.DataFrame()
    return pd.concat(price_series_list, axis=1).ffill()

@st.cache_data(ttl=3600)
def calculate_inv_vol_weights(tickers_list, lookback_trading_days=252):
    if not tickers_list: return {}
    prices_df = yf_prices_for_inv_vol(tuple(tickers_list), lookback_trading_days)
    min_data_points = max(2, int(lookback_trading_days * 0.2))
    if prices_df.empty or prices_df.shape[0] < min_data_points: return {t: 1.0 / len(tickers_list) for t in tickers_list}
    returns = prices_df.pct_change().dropna()
    if returns.empty or len(returns) < min_data_points: return {t: 1.0 / len(tickers_list) for t in tickers_list}
    std_devs = returns.std(); inv_vols = {}; valid_count = 0
    for ticker in tickers_list:
        if ticker not in std_devs or pd.isna(std_devs[ticker]): inv_vols[ticker] = 1e-9
        elif std_devs[ticker] == 0: inv_vols[ticker] = 1.0 / 1e-9; valid_count+=1
        else: inv_vols[ticker] = 1.0 / std_devs[ticker]; valid_count+=1
    if valid_count == 0: return {t: 1.0 / len(tickers_list) for t in tickers_list}
    return _norm(inv_vols)

def dca_split(total_amount, target_weights_dict):
    normalized_weights = _norm(target_weights_dict)
    return {k: round(total_amount * normalized_weights[k], 2) for k in normalized_weights}

def rebalance_trades(target_weights_dict, current_holdings_dict):
    normalized_weights = _norm(target_weights_dict); portfolio_value = sum(current_holdings_dict.values()); trades = {}
    for ticker, target_w in normalized_weights.items(): trades[ticker] = round(portfolio_value * target_w - current_holdings_dict.get(ticker, 0.0), 2)
    for ticker, current_val in current_holdings_dict.items():
        if ticker not in normalized_weights and current_val > 0: trades[ticker] = round(0 - current_val, 2)
    return trades

@st.cache_data(ttl=600)
def run_testfolio_backtest_api(payload):
    try:
        response = requests.post(TESTFOLIO_API_URL, json=payload, timeout=45)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try: st.error(f"API Response: {e.response.json()}")
            except json.JSONDecodeError: st.error(f"API Raw Response: {e.response.text}")
        return None
    except json.JSONDecodeError:
        st.error("Failed to decode API JSON response."); return None

def format_api_allocation(vti_p, vxus_p, tilt_overall_p, tilt_components, tilt_method_choice):
    # This function now takes all necessary percentages as direct inputs
    alloc_raw = {}
    if vti_p > 0: alloc_raw["VTI"] = float(vti_p)
    if vxus_p > 0: alloc_raw["VXUS"] = float(vxus_p)

    actual_tilt_tickers = [t.strip().upper() for t in tilt_components.split(",") if t.strip()]
    
    if tilt_overall_p > 0 and actual_tilt_tickers:
        component_weights = {}
        if tilt_method_choice == EQ_RISK_METHOD:
            component_weights = calculate_inv_vol_weights(actual_tilt_tickers)
        else: # EQ_WEIGHT_METHOD
            component_weights = {t: 1.0 / len(actual_tilt_tickers) for t in actual_tilt_tickers}
        
        for ticker, weight in component_weights.items():
            alloc_raw[ticker] = tilt_overall_p * weight
    
    # Normalize raw sum to 100
    raw_sum = sum(alloc_raw.values())
    if raw_sum > 0 and abs(raw_sum - 100.0) > 1e-9 : # If sum isn't 100 (e.g. from slider derivation)
        factor = 100.0 / raw_sum
        alloc_raw = {t: w * factor for t, w in alloc_raw.items()}

    alloc_rounded = {t: round(w, DECIMAL_PLACES_FOR_API) for t, w in alloc_raw.items()}
    alloc_final = {t: w for t, w in alloc_rounded.items() if w > (10**-(DECIMAL_PLACES_FOR_API + 1))} # Filter very small/zero after rounding

    if not alloc_final: return {} # Return empty if nothing is left

    # Ensure sum is exactly 100.00
    current_sum = sum(alloc_final.values())
    difference = 100.0 - current_sum
    if abs(difference) > (10**-(DECIMAL_PLACES_FOR_API + 1)): # If diff is significant
        sorted_alloc = sorted(alloc_final.items(), key=lambda item: item[1], reverse=True)
        if sorted_alloc:
            largest_ticker = sorted_alloc[0][0]
            adjusted_val = alloc_final[largest_ticker] + difference
            alloc_final[largest_ticker] = max(0.0, round(adjusted_val, DECIMAL_PLACES_FOR_API))
            alloc_final = {t: w for t, w in alloc_final.items() if w > (10**-(DECIMAL_PLACES_FOR_API + 1))} # Re-filter
    
    # Final check and re-normalization if filtering changed sum
    final_sum_check = sum(alloc_final.values())
    if alloc_final and abs(final_sum_check - 100.0) > (10**-(DECIMAL_PLACES_FOR_API)):
        factor = 100.0 / final_sum_check
        alloc_final = {t: round(w * factor, DECIMAL_PLACES_FOR_API) for t, w in alloc_final.items()}
        current_sum = sum(alloc_final.values()); difference = 100.0 - current_sum
        if abs(difference) > (10**-(DECIMAL_PLACES_FOR_API+1)) and alloc_final:
            sorted_alloc = sorted(alloc_final.items(), key=lambda item: item[1], reverse=True)
            if sorted_alloc:
                largest_ticker = sorted_alloc[0][0]
                alloc_final[largest_ticker] = round(alloc_final[largest_ticker] + difference, DECIMAL_PLACES_FOR_API)

    return {str(t).strip(): w for t, w in alloc_final.items() if str(t).strip() and w > 0}


# ---------- Main Application UI (Tabs) ----------
st.title("📈 Portfolio Management Dashboard")

if not run_button:
    st.info("👈 Adjust settings in the sidebar and click **Run Calculations & Backtest**.")
    st.stop()

# --- Calculate allocations based on current sidebar values ---
# For TestFolio API (needs to sum to 100)
custom_portfolio_api_alloc = format_api_allocation(
    vti_pct_input, vxus_pct_input, tilt_pct_derived, tickers_text_input, tilt_method_input
)

# For internal DCA/Rebalance calculations (normalized to sum to 1)
tilt_tickers_internal = [t.strip().upper() for t in tickers_text_input.split(",") if t.strip()]
tilt_component_weights_internal = {}
if tilt_pct_derived > 0 and tilt_tickers_internal:
    if tilt_method_input == EQ_RISK_METHOD:
        tilt_component_weights_internal = calculate_inv_vol_weights(tilt_tickers_internal)
    else:
        tilt_component_weights_internal = {t: 1.0 / len(tilt_tickers_internal) for t in tilt_tickers_internal}

internal_weights_raw = {}
if vti_pct_input > 0: internal_weights_raw["VTI"] = vti_pct_input / 100.0
if vxus_pct_input > 0: internal_weights_raw["VXUS"] = vxus_pct_input / 100.0
for ticker, weight in tilt_component_weights_internal.items():
    internal_weights_raw[ticker] = (tilt_pct_derived / 100.0) * weight

internal_final_normalized_weights = _norm({k:v for k,v in internal_weights_raw.items() if v > 1e-9})


tab1, tab2 = st.tabs(["💰 Allocation & Trades", "📊 Backtest Performance (TestFol.io)"])

with tab1:
    st.header("Portfolio Allocation & Orders")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Target Allocation")
        if internal_final_normalized_weights:
            weights_df = pd.Series(internal_final_normalized_weights, name="Target %").sort_values(ascending=False)
            st.dataframe(weights_df.to_frame().style.format("{:.2%}"), use_container_width=True, height=min(35 * (len(weights_df) + 1), 400))
        else: st.write("No assets selected.")
    with col2:
        st.subheader(f"💸 DCA Order")
        st.metric(label="Amount to Invest", value=f"${dca_amount_input:,.0f}")
        if dca_amount_input > 0 and internal_final_normalized_weights:
            dca_orders = dca_split(dca_amount_input, internal_final_normalized_weights)
            dca_df = pd.Series(dca_orders, name="Buy $").sort_values(ascending=False)[lambda x: x > 0.005]
            if not dca_df.empty: st.dataframe(dca_df.to_frame().style.format("${:,.2f}"), use_container_width=True, height=min(35 * (len(dca_df) + 1), 365))
            else: st.write("No DCA orders (check weights/amount).")
        elif dca_amount_input == 0: st.write("DCA amount is $0.")
        else: st.write("Cannot calculate DCA orders.")

    st.subheader("🔄 Rebalance Trades")
    portfolio_value = sum(current_holdings_input.values())
    st.metric(label="Current Portfolio Value", value=f"${portfolio_value:,.2f}")
    if portfolio_value > 0 and internal_final_normalized_weights:
        rebalance_order_dict = rebalance_trades(internal_final_normalized_weights, current_holdings_input)
        rebalance_df = pd.Series(rebalance_order_dict, name="Trade $").sort_values()[lambda x: abs(x) > 0.005]
        if not rebalance_df.empty: st.dataframe(rebalance_df.to_frame().style.format("${:,.2f}"), use_container_width=True, height=min(35 * (len(rebalance_df) + 1), 400))
        else: st.success("🎉 Portfolio is balanced!")
    elif portfolio_value == 0: st.write("Current portfolio value $0. No rebalancing applicable.")
    else: st.write("Cannot calculate rebalance trades.")

with tab2:
    st.header("Backtest Performance (via TestFol.io)")

    if not custom_portfolio_api_alloc:
        st.warning("Custom portfolio for API is empty (all weights zero or not defined). Cannot run backtest.")
    else:
        backtests_payload_list = []
        # Custom Portfolio
        backtests_payload_list.append({
            "name": "Custom Portfolio", # Name for easier identification later
            "invest_dividends": True, "rebalance_freq": api_rebalance_freq_input,
            "allocation": custom_portfolio_api_alloc,
            "drag": 0, "absolute_dev": 0, "relative_dev": 0
        })
        # VTI Benchmark (always include for comparison if not 100% VTI)
        if not ("VTI" in custom_portfolio_api_alloc and len(custom_portfolio_api_alloc) == 1 and custom_portfolio_api_alloc["VTI"] == 100):
            backtests_payload_list.append({
                "name": "VTI Benchmark",
                "invest_dividends": True, "rebalance_freq": api_rebalance_freq_input, # Match custom rebal freq
                "allocation": {"VTI": 100.00}, # Ensure VTI is exactly 100
                "drag": 0, "absolute_dev": 0, "relative_dev": 0
            })
        
        api_payload_final = {
            "start_date": api_start_date_input.isoformat(), "end_date": api_end_date_input.isoformat(),
            "start_val": api_start_val_input, "adj_inflation": False, "cashflow": 0,
            "cashflow_freq": "Yearly", "rolling_window": 60, "backtests": backtests_payload_list
        }

        st.markdown("##### API Request:")
        with st.expander("View Payload sent to TestFol.io"): st.json(api_payload_final)

        with st.spinner("⏳ Running TestFol.io backtest... This may take a moment."):
            api_result_data = run_testfolio_backtest_api(api_payload_final)

        if api_result_data:
            if api_result_data.get("errors") and api_result_data["errors"]:
                st.error("TestFol.io API returned errors:")
                for error_msg in api_result_data["errors"]: st.error(f"- {error_msg}")
            else:
                st.success("✅ Backtest complete!")

            charts_api = api_result_data.get("charts", {})
            history_api = charts_api.get("history")
            drawdown_api = charts_api.get("drawdown")

            portfolio_names_from_payload = [bt.get("name", f"Portfolio {i+1}") for i, bt in enumerate(backtests_payload_list)]

            if history_api and len(history_api) > 1:
                st.subheader("📈 Portfolio Value History")
                timestamps = pd.to_datetime(history_api[0], unit='s')
                df_history_plot = pd.DataFrame(index=timestamps)
                for i, name in enumerate(portfolio_names_from_payload):
                    if i+1 < len(history_api): # history_api[0] is dates, history_api[1] is first portfolio etc.
                        df_history_plot[name] = history_api[i+1]
                st.line_chart(df_history_plot)
            else: st.warning("Could not plot portfolio history from API.")

            if drawdown_api and len(drawdown_api) > 1:
                st.subheader("📉 Portfolio Drawdowns")
                if history_api and history_api[0]: # Use same timestamps if available
                    timestamps_dd = pd.to_datetime(drawdown_api[0], unit='s')
                    df_drawdown_plot = pd.DataFrame(index=timestamps_dd)
                    for i, name in enumerate(portfolio_names_from_payload):
                        if i+1 < len(drawdown_api):
                             df_drawdown_plot[f"{name} DD (%)"] = drawdown_api[i+1]
                    st.line_chart(df_drawdown_plot)
                else: st.warning("Missing timestamps for drawdown data from API.")
            else: st.warning("Could not plot portfolio drawdowns from API.")

            stats_list_api = api_result_data.get("stats")
            annual_returns_api = api_result_data.get("annual_returns")

            if stats_list_api:
                st.subheader("📊 Key Performance Statistics")
                df_stats_display_list = []
                for i, stats_item in enumerate(stats_list_api):
                    if i < len(portfolio_names_from_payload):
                        name = portfolio_names_from_payload[i]
                        beta_val = stats_item.get('beta')
                        df_stats_display_list.append({
                            "Portfolio": name,
                            "CAGR (%)": f"{stats_item.get('cagr', 0):.2f}",
                            "Max DD (%)": f"{stats_item.get('max_drawdown', 0):.2f}",
                            "Volatility (%)": f"{stats_item.get('std', 0):.2f}",
                            "Sharpe": f"{stats_item.get('sharpe', 0):.2f}",
                            "Sortino": f"{stats_item.get('sortino', 0):.2f}",
                            "Beta": f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else "N/A",
                            "End Value ($)": f"{stats_item.get('end_val', 0):,.0f}", # No cents for end val
                        })
                if df_stats_display_list:
                    st.dataframe(pd.DataFrame(df_stats_display_list).set_index("Portfolio"), use_container_width=True)
            
            if annual_returns_api:
                st.subheader("🗓️ Annual Returns")
                df_annual_display_data = []
                for row in annual_returns_api: # [year, p1_ret, p1_val, p2_ret, p2_val, ...]
                    year_val = row[0]
                    entry = {"Year": year_val}
                    for i, name in enumerate(portfolio_names_from_payload):
                        ret_idx = 1 + (i * 2) # Return is at 1, 3, 5...
                        if ret_idx < len(row):
                            entry[f"{name} (%)"] = f"{row[ret_idx]:.2f}"
                    df_annual_display_data.append(entry)
                if df_annual_display_data:
                    st.dataframe(pd.DataFrame(df_annual_display_data).set_index("Year"), use_container_width=True)
        else:
            st.error("❌ Failed to get a valid response from TestFol.io API.")
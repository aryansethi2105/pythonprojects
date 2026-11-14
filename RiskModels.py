import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats, integrate
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import colorsys
warnings.filterwarnings('ignore')

st.set_page_config(page_title="VaR & ES Calculator", layout="wide")

st.title("Value at Risk (VaR) & Expected Shortfall (ES) Calculator", text_alignment="center")
st.markdown("---")

tab1, tab2 = st.tabs(["Introduction", "Calculator"])

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data"""
    ticker = yf.Ticker(symbol)
    stock_data = ticker.history(
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        auto_adjust=False
    )
    return stock_data

@st.cache_data
def calculate_var_es(portfolio_value, mu_daily=0, sigma_daily=0, 
                    daily_returns=None, confidence=0.95, horizon_days=1,
                    method='parametric', lognormal=False):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) with time horizon
    
    Parameters:
    - portfolio_value: Portfolio value ($)
    - mu_daily: Daily mean return (decimal) - for parametric/monte_carlo
    - sigma_daily: Daily volatility (decimal) - for parametric/monte_carlo
    - daily_returns: Array of historical daily returns (decimal) - for historical method
    - confidence: Confidence level (e.g., 0.95 for 95%)
    - horizon_days: Time horizon in days
    - method: 'parametric', 'historical', or 'monte_carlo'
    - lognormal: For parametric, use lognormal (True) or normal (False)
    
    Returns:
    - var, es (both positive numbers representing losses)
    
    Examples:
    >>> calculate_var_es(1000000, mu_daily=0.0005, sigma_daily=0.02,
    ...                  confidence=0.95, horizon_days=10)
    >>> calculate_var_es(1000000, daily_returns=returns_array,
    ...                  confidence=0.95, horizon_days=10, method='historical')
    """
    
    alpha = 1 - confidence
    
    if method == 'parametric':
        # Scale for horizon
        mu_horizon = mu_daily * horizon_days
        sigma_horizon = sigma_daily * np.sqrt(horizon_days)
        
        if lognormal:
            # Lognormal distribution
            z_alpha = stats.norm.ppf(alpha)
            var = portfolio_value * (1 - np.exp(mu_horizon + z_alpha * sigma_horizon))
            es = portfolio_value * (1 - np.exp(mu_horizon + 0.5 * sigma_horizon**2) *
                                   stats.norm.cdf(z_alpha - sigma_horizon) / alpha)
        else:
            # Normal distribution
            z_alpha = stats.norm.ppf(alpha)
            var = portfolio_value * (z_alpha * sigma_horizon - mu_horizon)
            phi_z_alpha = stats.norm.pdf(z_alpha)
            es = portfolio_value * (-mu_horizon + sigma_horizon * phi_z_alpha / alpha)
    
    elif method == 'historical':
        if daily_returns is None:
            raise ValueError("daily_returns required for historical method")
        
        # Check if we have enough data
        if len(daily_returns) < horizon_days:
            # If we can't do proper bootstrapping, use simpler approach
            # Generate horizon returns by sampling individual days
            n_simulations = min(10000, len(daily_returns) * 10)
            horizon_returns = []
            for _ in range(n_simulations):
                indices = np.random.choice(len(daily_returns), horizon_days, replace=True)
                selected_returns = daily_returns[indices]
                cum_return = np.prod(1 + selected_returns) - 1
                horizon_returns.append(cum_return)
        else:
            # Use overlapping periods (block bootstrap)
            n_simulations = min(10000, len(daily_returns) - horizon_days + 1)
            horizon_returns = []
            possible_starts = len(daily_returns) - horizon_days + 1
            
            # Use overlapping periods
            for _ in range(n_simulations):
                start_idx = np.random.randint(0, possible_starts)
                block_returns = daily_returns[start_idx:start_idx + horizon_days]
                cum_return = np.prod(1 + block_returns) - 1
                horizon_returns.append(cum_return)
        
        horizon_returns = np.array(horizon_returns)
        pnl = portfolio_value * horizon_returns
        sorted_pnl = np.sort(pnl)
        
        # Calculate VaR and ES
        n = len(sorted_pnl)
        var_index = int(alpha * n)
        var_index = max(0, min(var_index, n - 1))
        
        var = -sorted_pnl[var_index]
        tail_losses = -sorted_pnl[:var_index + 1]
        es = np.mean(tail_losses) if len(tail_losses) > 0 else var
    
    elif method == 'monte_carlo':
        n_simulations = 10000
        
        if daily_returns is not None and not lognormal:
            # Use empirical distribution from historical data if available
            # Bootstrap from historical returns
            horizon_returns = []
            for _ in range(n_simulations):
                indices = np.random.choice(len(daily_returns), horizon_days, replace=True)
                selected_returns = daily_returns[indices]
                cum_return = np.prod(1 + selected_returns) - 1
                horizon_returns.append(cum_return)
            horizon_returns = np.array(horizon_returns)
        else:
            # Use parametric simulation
            # Scale for horizon
            mu_horizon = mu_daily * horizon_days
            sigma_horizon = sigma_daily * np.sqrt(horizon_days)
            
            if lognormal:
                # Simulate log returns
                horizon_log_returns = np.random.normal(mu_horizon, sigma_horizon, n_simulations)
                horizon_returns = np.exp(horizon_log_returns) - 1
            else:
                # Simulate simple returns
                horizon_returns = np.random.normal(mu_horizon, sigma_horizon, n_simulations)
        
        # Calculate VaR and ES
        pnl = portfolio_value * horizon_returns
        sorted_pnl = np.sort(pnl)
        n = len(sorted_pnl)
        var_index = int(alpha * n)
        var_index = max(0, min(var_index, n - 1))
        
        var = -sorted_pnl[var_index]
        tail_losses = -sorted_pnl[:var_index + 1]
        es = np.mean(tail_losses) if len(tail_losses) > 0 else var
    
    else:
        raise ValueError("Method must be 'parametric', 'historical', or 'monte_carlo'")
    
    return abs(var), abs(es)
                        
def simulate_monte_carlo_returns(investment, mu_daily, sigma_daily, horizon_days, n_simulations, use_log_returns=True):
    """Simulate Monte Carlo returns for visualization purposes"""
    # Scale for horizon
    mu_horizon = mu_daily * horizon_days
    sigma_horizon = sigma_daily * np.sqrt(horizon_days)
    
    if use_log_returns:
        # Simulate log returns
        horizon_log_returns = np.random.normal(mu_horizon, sigma_horizon, n_simulations)
        horizon_returns = np.exp(horizon_log_returns) - 1
    else:
        # Simulate simple returns
        horizon_returns = np.random.normal(mu_horizon, sigma_horizon, n_simulations)
    
    # Calculate P&L
    portfolio_returns = investment * horizon_returns
    
    # Simulate daily returns for path visualization
    if use_log_returns:
        daily_returns = np.random.normal(mu_daily, sigma_daily, (n_simulations, horizon_days))
    else:
        daily_returns = np.random.normal(mu_daily, sigma_daily, (n_simulations, horizon_days))
    
    return portfolio_returns, daily_returns
                        
def generate_distinct_colors(n):
    """Generate n visually distinct colors"""
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.5 + 0.2 * np.sin(2 * np.pi * i / n)
        saturation = 0.7 + 0.2 * np.cos(2 * np.pi * i / n)
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.3)')
    return colors

with tab1:
    st.header("A Brief Introduction of VaR & ES")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### What is Value at Risk (VaR)?")
        st.markdown("""
        **Value at Risk (VaR)** is a statistical measure that quantifies the level of financial risk 
        within a firm, portfolio, or position over a specific time frame. It estimates the maximum 
        potential loss with a given confidence level.
        
        **Example:** A 1-day 95% VaR of \$1 million means there is a 95% confidence that losses 
        will not exceed \$1 million in one day (or 5% chance of exceeding it).
        """)
        
        st.markdown("#### Formulas:")
        st.latex(r"VaR = P \times (Z_{\alpha} \sigma \sqrt{T} - \mu T)")
        st.latex(r"VaR_{\text{lognormal}} = P \times \left(1 - e^{\mu T + Z_{\alpha} \sigma \sqrt{T}}\right)")
        
        st.markdown("**Where:**")
        st.markdown(r"""
        - $P$ = Initial investment
        - $\mu$ = Mean daily return
        - $T$ = Time horizon in days
        - $Z_{\alpha}$ = Z-score for tail probability
        - $\sigma$ = Daily standard deviation
        """)
    
    with col2:
        st.markdown("### What is Expected Shortfall (ES)?")
        st.markdown("""
        **Expected Shortfall (ES)**, also known as Conditional VaR (CVaR), measures the average loss 
        that occurs in the worst-case scenarios beyond the VaR level. It provides a more 
        comprehensive view of tail risk.
        
        **Example:** If 95% VaR is \$1M, the ES might be \$1.5M, meaning the average loss 
        in the worst 5% of cases is \$1.5M.
        """)
        
        st.markdown("#### Formulas:")
        st.latex(r"ES = P \times \left(-\mu T + \sigma \sqrt{T} \frac{\phi(Z_{\alpha})}{\alpha}\right)")
        st.latex(r"ES_{\text{lognormal}} = P \times \left(1 - e^{\mu T + \frac{\sigma^2 T}{2}} \times \frac{\Phi(Z_{\alpha} - \sigma \sqrt{T})}{\alpha}\right)")
        
        st.markdown("**Where:**")
        st.markdown(r"""
        - $\phi(Z_{\alpha})$ = Standard normal PDF at $Z_{\alpha}$
        - $\alpha$ = Tail probability (e.g., 0.05 for 95% confidence)
        """)

with tab2:
    use_custom_params = st.checkbox(
        "Use Custom Statistical Parameters",
        help="Override automatic calculation from historical data",
        key="use_custom_params"
    )
    
    # Initialize custom parameters
    custom_mean_return = 0.05 / 100  # 0.05%
    custom_std_dev = 1.5 / 100  # 1.5%
    
    if use_custom_params:
        st.markdown("#### Custom Statistical Parameters")
        
        st.info("""
        **Important:** These parameters represent the mean and standard deviation of 
        **daily returns**.
        """)
        
        custom_col1, custom_col2 = st.columns(2)
        with custom_col1:
            custom_mean_return = st.number_input(
                "Daily Mean Return (%)",
                value=0.05,
                step=0.01,
                format="%.2f",
                help="Expected average daily return",
                key="custom_mean_return"
            ) / 100
        
        with custom_col2:
            custom_std_dev = st.number_input(
                "Daily Standard Deviation (%)",
                value=1.5,
                step=0.1,
                format="%.2f",
                help="Daily volatility (standard deviation of returns)",
                key="custom_std_dev"
            ) / 100
    
    with st.form("input_form"):
        st.subheader("Input Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Parameters")
            investment = st.number_input(
                "Initial Investment Amount (USD)",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
                help="The initial portfolio value"
            )
            
            if not use_custom_params:
                stock_symbol = st.text_input(
                    "Stock Symbol (e.g., AAPL, MSFT, GOOGL)", 
                    "AAPL",
                    help="Enter a valid stock ticker symbol"
                ).upper()
            else:
                stock_symbol = "CUSTOM_PARAMS"
            
            st.markdown("#### Date Range")
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                disabled=use_custom_params,
                help="Ignored when using custom parameters" if use_custom_params else "Start date for historical data"
            ) 
            
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                disabled=use_custom_params,
                help="Ignored when using custom parameters" if use_custom_params else "End date for historical data"
            )
        
        with col2:
            st.markdown("#### Risk Parameters")
            forecast_days = st.number_input(
                "Time Horizon (Days)",
                min_value=1,
                value=10,
                step=1,
                help="Number of days for VaR/ES calculation"
            )
            
            confidence_level = st.slider(
                "Confidence Level (%)",
                min_value=0,
                max_value=100,
                value=95,
                step=1,
                help="Confidence level for risk calculation"
            )
            
            st.markdown("#### Simulation Settings")
            n_simulations = st.number_input(
                "Number of Monte Carlo Simulations",
                min_value=1,
                value=1000,
                step=1000,
                help="More simulations = more accurate but slower computation"
            )

            use_log_returns = st.checkbox(
                "Use Log Returns (Recommended)",
                value=True,
                help="""Use log returns for more accurate multi-period calculations."""
            )
            
        calculate_button = st.form_submit_button("Calculate VaR & ES", use_container_width=True)
    
    if calculate_button:
        try:
            with st.spinner(f"Calculating {forecast_days}-day VaR & ES..."):
                # Initialize variables
                mean_return = None
                std_dev = None
                stock_data = None
                returns = None
        
                if use_custom_params:
                    # Use custom parameters
                    mean_return = custom_mean_return
                    std_dev = custom_std_dev
                    stock_symbol = "CUSTOM_PARAMS"
                    st.success(f"✓ Using custom parameters: Mean={mean_return*100:.2f}%, Std Dev={std_dev*100:.2f}%")
                    
                    # Create empty returns for Data Points metric
                    returns = pd.Series([])
                    
                else:
                    # Fetch historical data
                    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
                    
                    if stock_data.empty:
                        st.error(f"No data found for {stock_symbol}. Please check the symbol and date range.")
                        st.stop()
                    
                    st.success(f"✓ Retrieved {len(stock_data)} trading days of data for {stock_symbol}")
                    
                    # Calculate returns from historical data
                    price_series = stock_data['Adj Close'] if 'Adj Close' in stock_data.columns else stock_data['Close']
                    price_series_clean = price_series.dropna()
                    
                    if len(price_series_clean) == 0:
                        st.error("No valid price data after removing NaN values. Please try a different date range.")
                        st.stop()
                    
                    if use_log_returns:
                        returns = np.log(price_series_clean / price_series_clean.shift(1)).dropna()
                        st.info("Using log returns for more accurate multi-period calculations")
                    else:
                        returns = price_series_clean.pct_change().dropna()
                        st.info("Using simple returns")
                
                    mean_return = float(returns.mean())
                    std_dev = float(returns.std())
                
                # Validate calculations
                if mean_return is None or std_dev is None:
                    st.error("Could not determine mean or standard deviation. Please check your inputs.")
                    st.stop()
                
                if pd.isna(mean_return) or pd.isna(std_dev):
                    st.error("Could not calculate mean or standard deviation. Please check your data.")
                    st.stop()
                
                if std_dev == 0:
                    st.warning("Standard deviation is zero. This may indicate insufficient price variation.")
                
                # Display statistical summary
                st.markdown("---")
                st.subheader("Statistical Summary")
                
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Daily Mean", f"{mean_return*100:.2f}%")
                with stats_col2:
                    st.metric("Daily Std Dev", f"{std_dev*100:.2f}%")
                with stats_col3:
                    annualized_vol = std_dev * np.sqrt(252)
                    st.metric("Annual Volatility", f"{annualized_vol*100:.2f}%")
                with stats_col4:
                    if use_custom_params:
                        st.metric("Data Points", "N/A (Custom)")
                    else:
                        st.metric("Data Points", f"{len(returns)}")
                
                # Time-scaled parameters
                st.markdown(f"#### {forecast_days}-Day Scaled Parameters")
                scaled_mean = mean_return * forecast_days
                scaled_vol = std_dev * np.sqrt(forecast_days)
                
                scale_col1, scale_col2 = st.columns(2)
                with scale_col1:
                    st.metric(f"{forecast_days}-Day Mean Return", f"{scaled_mean*100:.2f}%")
                with scale_col2:
                    st.metric(f"{forecast_days}-Day Volatility", f"{scaled_vol*100:.2f}%")
                
                # Use appropriate methods based on data source
                if use_custom_params:
                    # For custom parameters: Use Parametric and Monte Carlo
                    # Calculate Parametric VaR & ES
                    var_method1, es_method1 = calculate_var_es(
                        portfolio_value=investment,
                        mu_daily=mean_return,
                        sigma_daily=std_dev,
                        confidence=confidence_level/100,
                        horizon_days=forecast_days,
                        method='parametric',
                        lognormal=use_log_returns  # Use lognormal if log returns are selected
                    )
                    
                    method1_name = "Parametric"
                    
                else:
                    # For historical data: Use Historical and Monte Carlo
                    # Calculate Historical VaR & ES
                    var_method1, es_method1 = calculate_var_es(
                        portfolio_value=investment,
                        daily_returns=returns.values if returns is not None else None,
                        confidence=confidence_level/100,
                        horizon_days=forecast_days,
                        method='historical'
                    )
                    
                    method1_name = "Historical"

                # Always calculate Monte Carlo (common to both approaches)
                var_monte_carlo, es_monte_carlo = calculate_var_es(
                    portfolio_value=investment,
                    mu_daily=mean_return,
                    sigma_daily=std_dev,
                    daily_returns=returns.values if (returns is not None and not use_custom_params) else None,
                    confidence=confidence_level/100,
                    horizon_days=forecast_days,
                    method='monte_carlo',
                    lognormal=use_log_returns
                )
                
                # Get z-score for display (only relevant for parametric method)
                alpha = 1 - (confidence_level/100)
                z_score = stats.norm.ppf(alpha)
                
                # Simulate returns for visualization
                portfolio_returns, daily_returns = simulate_monte_carlo_returns(
                    investment=investment,
                    mu_daily=mean_return,
                    sigma_daily=std_dev,
                    horizon_days=forecast_days,
                    n_simulations=n_simulations,
                    use_log_returns=use_log_returns
                )
                
                # Display results
                st.markdown("---")
                st.subheader(f"{forecast_days}-Day Risk Metrics")
                
                results_col1, results_col2 = st.columns(2)
                with results_col1:
                    st.markdown(f"##### {method1_name} Method")
                    st.metric(f"Value at Risk ({confidence_level}%)", f"${var_method1:,.2f}")
                    st.metric("Expected Shortfall", f"${es_method1:,.2f}")
                    if method1_name == "Parametric":
                        st.caption(f"Z-score: {z_score:.4f}")
                    else:
                        st.caption(f"Based on {len(returns) if returns is not None else 0} historical returns")
                
                with results_col2:
                    st.markdown("##### Monte Carlo Simulation")
                    st.metric(f"Value at Risk ({confidence_level}%)", f"${var_monte_carlo:,.2f}")
                    st.metric("Expected Shortfall", f"${es_monte_carlo:,.2f}")
                    st.caption(f"Based on {n_simulations:,} simulations")
                
                # Percentage of investment
                st.markdown("#### As Percentage of Investment")
                percent_col1, percent_col2 = st.columns(2)
                with percent_col1:
                    var_percent_method1 = var_method1 / investment * 100
                    es_percent_method1 = es_method1 / investment * 100
                    st.metric(f"{method1_name} VaR", f"{var_percent_method1:.2f}%")
                    st.metric(f"{method1_name} ES", f"{es_percent_method1:.2f}%")
                
                with percent_col2:
                    var_percent_mc = var_monte_carlo / investment * 100
                    es_percent_mc = es_monte_carlo / investment * 100
                    st.metric("Monte Carlo VaR", f"{var_percent_mc:.2f}%")
                    st.metric("Monte Carlo ES", f"{es_percent_mc:.2f}%")
                
                # Visualizations
                st.markdown("---")
                st.subheader("Visualizations")
                
                viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                    f"{forecast_days}-Day Return Distribution", 
                    "Monte Carlo Simulation", 
                    "Time Horizon Analysis"
                ])
                
                with viz_tab1:
                    hist, bin_edges = np.histogram(portfolio_returns, bins=100)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    fig1 = go.Figure()
                    
                    # Legend items with consistent formatting
                    legend_items = [
                        ('Simulated Returns', 'rgba(31, 119, 180, 0.7)'),
                        ('Tail Risk Region', 'rgba(214, 39, 40, 0.3)'),
                        (f'VaR ({confidence_level}%): ${var_monte_carlo:,.2f}', '#d62728'),
                        (f'Expected Shortfall: ${es_monte_carlo:,.2f}', '#ff7f0e')
                    ]
                    
                    for name, color in legend_items:
                        fig1.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(symbol='square', size=12, color=color, line=dict(width=1, color='white')),
                            name=name,
                            showlegend=True
                        ))
                    
                    # Actual data traces (hidden from legend)
                    fig1.add_trace(go.Bar(
                        x=bin_centers, y=hist,
                        width=[(bin_edges[i+1] - bin_edges[i]) * 0.9 for i in range(len(bin_edges)-1)],
                        name='_Simulated Returns Data',
                        marker_color='rgba(31, 119, 180, 0.7)',
                        marker_line=dict(width=1, color='white'),
                        opacity=0.7,
                        hovertemplate='<b>Return</b>: $%{x:,.2f}<br><b>Frequency</b>: %{y}<extra></extra>',
                        showlegend=False
                    ))
                    
                    # Add vertical lines - VaR and ES are shown as NEGATIVE returns (losses)
                    fig1.add_vline(x=-var_monte_carlo, line_dash="dash", line_color="#d62728", line_width=2.5, showlegend=False)
                    fig1.add_vline(x=-es_monte_carlo, line_dash="dot", line_color="#ff7f0e", line_width=2.5, showlegend=False)
                    
                    # Update layout
                    fig1.update_layout(
                        title=dict(text=f"{forecast_days}-Day Return Distribution", font=dict(size=16), x=0.5, xanchor='center'),
                        xaxis_title=dict(text=f"{forecast_days}-Day Return ($)", font=dict(size=12)),
                        yaxis_title=dict(text="Frequency", font=dict(size=12)),
                        xaxis=dict(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        yaxis=dict(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        template="plotly_white",
                        height=500,
                        hovermode="x unified",
                        legend=dict(
                            yanchor="top", y=0.99, xanchor="right", x=1.15,
                            bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="rgba(0, 0, 0, 0.2)",
                            borderwidth=1, font=dict(size=10, color="black")
                        ),
                        margin=dict(l=50, r=50, t=60, b=50),
                        plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                        bargap=0.1, bargroupgap=0.1
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Distribution statistics
                    tail_mask = portfolio_returns <= -var_monte_carlo
                    tail_returns = portfolio_returns[tail_mask]
                    
                    st.caption("Distribution Statistics")
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    with summary_col1:
                        st.metric("Simulation Mean", f"${portfolio_returns.mean():,.2f}")
                    with summary_col2:
                        st.metric("Simulation Median", f"${np.median(portfolio_returns):,.2f}")
                    with summary_col3:
                        st.metric("Simulation Std Dev", f"${portfolio_returns.std():,.2f}")
                    with summary_col4:
                        st.metric("Tail Probability", f"{len(tail_returns)/len(portfolio_returns)*100:.2f}%")
                
                with viz_tab2:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text(f"Generating {n_simulations:,} simulations...")
                    
                    colors = generate_distinct_colors(n_simulations)
                    fig2 = go.Figure()
                    
                    for i in range(n_simulations):
                        
                        cumulative_return = np.cumprod(1 + daily_returns[i])
                        portfolio_path = investment * cumulative_return
                        
                        fig2.add_trace(go.Scatter(
                            x=list(range(forecast_days)), y=portfolio_path,
                            mode='lines', line=dict(width=1, color=colors[i]),
                            name=f'Path {i+1}', showlegend=False,
                            hovertemplate=f'<b>Path {i+1}</b><br>Day %{{x}}<br>Value: $%{{y:,.2f}}<extra></extra>'
                        ))
                        
                        if i % 10 == 0:
                            progress_bar.progress((i + 1) / n_simulations)
                    
                    # Add reference lines
                    mean_path = investment * np.cumprod(1 + daily_returns.mean(axis=0))
                    median_path = investment * np.cumprod(1 + np.median(daily_returns, axis=0))
                    var_portfolio_value = investment - var_monte_carlo  # Subtract loss from investment
                    
                    fig2.add_trace(go.Scatter(
                        x=list(range(forecast_days)), y=mean_path,
                        mode='lines', line=dict(width=3, color='#000000'),
                        name='Mean Path',
                        hovertemplate='<b>Mean Path</b><br>Day %{x}<br>Value: $%{y:,.2f}<extra></extra>'
                    ))
                    
                    fig2.add_trace(go.Scatter(
                        x=list(range(forecast_days)), y=median_path,
                        mode='lines', line=dict(width=2, color='#ff7f0e', dash='dash'),
                        name='Median Path',
                        hovertemplate='<b>Median Path</b><br>Day %{x}<br>Value: $%{y:,.2f}<extra></extra>'
                    ))
                    
                    fig2.add_hline(y=investment, line_dash="dash", line_color="#2ca02c", line_width=2,
                        annotation=dict(text=f"Initial: ${investment:,.2f}", font=dict(size=10, color="#2ca02c"),
                        bgcolor="rgba(255,255,255,0.9)", borderwidth=1, bordercolor="#2ca02c",
                        yanchor="bottom", y=1.02, xanchor="left", x=0.02, showarrow=False))
                    
                    fig2.add_hline(y=var_portfolio_value, line_dash="dot", line_color="#d62728", line_width=2,
                        annotation=dict(text=f"VaR ({confidence_level}%): ${var_portfolio_value:,.2f}", font=dict(size=10, color="#d62728"),
                        bgcolor="rgba(255,255,255,0.9)", borderwidth=1, bordercolor="#d62728",
                        yanchor="top", y=-0.02, xanchor="left", x=0.02, showarrow=False))
                    
                    fig2.update_layout(
                        title=dict(text=f"Monte Carlo Simulation ({n_simulations:,} Paths) for {forecast_days} Days", font=dict(size=16), x=0.5, xanchor='center'),
                        xaxis_title=dict(text="Days", font=dict(size=12)),
                        yaxis_title=dict(text="Portfolio Value ($)", font=dict(size=12)),
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        template="plotly_white", height=600, hovermode="closest",
                        legend=dict(
                            yanchor="top", y=0.99, xanchor="left", x=1.02,
                            bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="rgba(0, 0, 0, 0.2)",
                            borderwidth=1, font=dict(size=10, color="black")
                        ),
                        margin=dict(l=50, r=50, t=60, b=50),
                        plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)'
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    status_text.success(f"✓ Generated {n_simulations:,} paths!")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Path statistics
                    st.caption("Path Statistics")
                    path_col1, path_col2, path_col3, path_col4 = st.columns(4)
                    with path_col1:
                        st.metric("Final Mean Value", f"${mean_path[-1]:,.2f}")
                    with path_col2:
                        max_final = investment * np.max(np.prod(1 + daily_returns, axis=1))
                        st.metric("Max Final Value", f"${max_final:,.2f}")
                    with path_col3:
                        min_final = investment * np.min(np.prod(1 + daily_returns, axis=1))
                        st.metric("Min Final Value", f"${min_final:,.2f}")
                    with path_col4:
                        median_final = investment * np.median(np.prod(1 + daily_returns, axis=1))
                        st.metric("Median Final", f"${median_final:,.2f}")
                
                with viz_tab3:
                    horizons_to_analyze = [1, 5, 10, 20, 30, forecast_days]
                    horizons_to_analyze = sorted(set(horizons_to_analyze))
                    
                    method_vars, method_es_list = [], []
                    monte_carlo_vars, monte_carlo_es_list = [], []
                    
                    progress_bar2 = st.progress(0)
                    status_text2 = st.empty()
                    
                    for idx, horizon in enumerate(horizons_to_analyze):
                        status_text2.text(f"Calculating for {horizon}-day horizon...")

                        if use_custom_params:
                            # For custom parameters: use parametric method
                            var_method, es_method = calculate_var_es(
                                portfolio_value=investment,
                                mu_daily=mean_return,
                                sigma_daily=std_dev,
                                confidence=confidence_level/100,
                                horizon_days=horizon,
                                method='parametric',
                                lognormal=use_log_returns
                            )
                        else:
                            # For historical data: use historical method
                            var_method, es_method = calculate_var_es(
                                portfolio_value=investment,
                                daily_returns=returns.values if returns is not None else None,
                                confidence=confidence_level/100,
                                horizon_days=horizon,
                                method='historical'
                            )
                        
                        method_vars.append(var_method)
                        method_es_list.append(es_method)
                        
                        var_mc, es_mc = calculate_var_es(
                            portfolio_value=investment,
                            mu_daily=mean_return,
                            sigma_daily=std_dev,
                            daily_returns=returns.values if (returns is not None and not use_custom_params) else None,
                            confidence=confidence_level/100,
                            horizon_days=horizon,
                            method='monte_carlo',
                            lognormal=use_log_returns
                        )
                        monte_carlo_vars.append(var_mc)
                        monte_carlo_es_list.append(es_mc)
                        
                        progress_bar2.progress((idx + 1) / len(horizons_to_analyze))
                    
                    fig3 = go.Figure()
                    
                    # Setting appropriate labels based on method
                    method1_label = 'Parametric VaR' if use_custom_params else 'Historical VaR'
                    method1_es_label = 'Parametric ES' if use_custom_params else 'Historical ES'
                    
                    fig3.add_trace(go.Scatter(x=horizons_to_analyze, y=method_vars, mode='lines+markers',
                        name=method1_label, line=dict(color='#1f77b4', width=2.5), marker=dict(size=8, symbol='circle')))
                    fig3.add_trace(go.Scatter(x=horizons_to_analyze, y=monte_carlo_vars, mode='lines+markers',
                        name='Monte Carlo VaR', line=dict(color='#ff7f0e', width=2.5, dash='dash'), marker=dict(size=8, symbol='square')))
                    fig3.add_trace(go.Scatter(x=horizons_to_analyze, y=method_es_list, mode='lines+markers',
                        name=method1_es_label, line=dict(color='#2ca02c', width=2.5), marker=dict(size=8, symbol='diamond')))
                    fig3.add_trace(go.Scatter(x=horizons_to_analyze, y=monte_carlo_es_list, mode='lines+markers',
                        name='Monte Carlo ES', line=dict(color='#d62728', width=2.5, dash='dot'), marker=dict(size=8, symbol='cross')))
                    
                    sqrt_scaling = [method_vars[0] * np.sqrt(h) for h in horizons_to_analyze]
                    fig3.add_trace(go.Scatter(x=horizons_to_analyze, y=sqrt_scaling, mode='lines',
                        name='√n Scaling', line=dict(color='#7f7f7f', width=1.5, dash='dot'), opacity=0.6))
                    
                    fig3.update_layout(
                        title=dict(text=f"Risk Metric Scaling with Time Horizon ({confidence_level}% Confidence)", font=dict(size=16), x=0.5, xanchor='center'),
                        xaxis_title=dict(text="Time Horizon (Days)", font=dict(size=12)),
                        yaxis_title=dict(text="Risk Metric ($)", font=dict(size=12)),
                        xaxis=dict(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        yaxis=dict(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        template="plotly_white", height=500, hovermode="x unified",
                        legend=dict(
                            yanchor="top", y=0.99, xanchor="left", x=1.02,
                            bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="rgba(0, 0, 0, 0.2)",
                            borderwidth=1, font=dict(size=10, color="black")
                        ),
                        margin=dict(l=50, r=50, t=60, b=50),
                        plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)'
                    )
                    
                    progress_bar2.empty()
                    status_text2.empty()
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Export results
                st.markdown("---")
                st.subheader("Export Results")
                
                method1_display = "Parametric" if use_custom_params else "Historical"
                
                results_df = pd.DataFrame({
                    'Parameter': [
                        'Stock Symbol', 'Initial Investment ($)', 'Time Horizon (Days)', 'Confidence Level (%)',
                        'Daily Mean Return (%)', 'Daily Standard Deviation (%)', f'{forecast_days}-Day Mean Return (%)',
                        f'{forecast_days}-Day Volatility (%)', f'{method1_display} VaR ($)', f'{method1_display} VaR (%)',
                        f'{method1_display} ES ($)', f'{method1_display} ES (%)', 'Monte Carlo VaR ($)', 'Monte Carlo VaR (%)',
                        'Monte Carlo ES ($)', 'Monte Carlo ES (%)', 'Z-score', 'Number of Simulations'
                    ],
                    'Value': [
                        stock_symbol, f'{investment:,.2f}', forecast_days, f'{confidence_level}',
                        f'{mean_return*100:.2f}', f'{std_dev*100:.2f}', f'{scaled_mean*100:.2f}',
                        f'{scaled_vol*100:.2f}', f'{var_method1:,.2f}', f'{var_method1/investment*100:.2f}',
                        f'{es_method1:,.2f}', f'{es_method1/investment*100:.2f}', f'{var_monte_carlo:,.2f}',
                        f'{var_monte_carlo/investment*100:.2f}', f'{es_monte_carlo:,.2f}', f'{es_monte_carlo/investment*100:.2f}',
                        f'{z_score:.4f}', f'{n_simulations:,}'
                    ]
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Complete Results as CSV",
                    data=csv,
                    file_name=f"var_es_{stock_symbol}_{forecast_days}d_{confidence_level}pc_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        st.markdown("---")
        st.info("Enter your parameters above and click 'Calculate VaR & ES' to begin analysis.")

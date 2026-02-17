import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes Formula Functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price, d1, d2

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price, d1, d2

st.title("Black-Scholes Option Pricing Calculator", text_alignment="center")

tab1, tab2 = st.tabs(["Home", "Calculator"])

# Home Page
with tab1:
    st.subheader("Introduction")
    st.write("""
    Welcome to the **Black-Scholes Option Pricing Calculator**! This professional tool allows you to:
    
    - Calculate theoretical prices for European call and put options  
    - Understand the Black-Scholes model in detail  
    - Visualise option payoffs and Greeks  
    - Make informed financial decisions  
    
    Options are financial derivatives that give the holder the right, but not the obligation, 
    to buy or sell an underlying asset at a specified price (strike price) on or before a certain date (expiration).
    """)
    
    st.subheader("About the Black-Scholes Model")
    
    st.write("""
    The **Black-Scholes model**, developed by Fischer Black and Myron Scholes in 1973, is a groundbreaking 
    mathematical model for pricing European-style options. It revolutionized the field of quantitative finance 
    and earned Scholes and Robert Merton the 1997 Nobel Prize in Economics.
    """)
    
    st.write("""
    The model is based on these key assumptions:
    
    1. **European Exercise**: Options can only be exercised at expiration.
    2. **No Dividends**: No dividends are paid during the option's life.
    3. **Efficient Markets**: No arbitrage opportunities exist.
    4. **Constant Parameters**: Risk-free rate and volatility are constant.
    5. **Lognormal Returns**: Stock returns follow a log-normal distribution.
    6. **Frictionless Markets**: No transaction costs or taxes.
    7. **Continuous Trading**: Trading occurs continuously.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Call Option Formula")
        st.latex(r"C = S_0 N(d_1) - K e^{-rT} N(d_2)")
        
        with st.expander("📐 Variables Explained", expanded=True):
            st.write("""
            Where:
            - $C$ = Call option price
            - $S_0$ = Current stock price
            - $K$ = Strike price
            - $T$ = Time to expiration (years)
            - $r$ = Risk-free interest rate
            - $N(·)$ = Cumulative standard normal distribution
            - $d_1 = \dfrac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$
            - $d_2 = d_1 - \sigma\sqrt{T}$
            """)
    
    with col2:
        st.subheader("Put Option Formula")
        st.latex(r"P = K e^{-rT} N(-d_2) - S_0 N(-d_1)")
        
        with st.expander("📐 Variables Explained", expanded=True):
            st.write("""
            Where:
            - $P$ = Put option price
            - $S_0$ = Current stock price
            - $K$ = Strike price
            - $T$ = Time to expiration (years)
            - $r$ = Risk-free interest rate
            - $N(·)$ = Cumulative standard normal distribution
            - $d_1 = \dfrac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$
            - $d_2 = d_1 - \sigma\sqrt{T}$
            """)

    st.subheader("Understanding Option Moneyness")
    
    st.write("**Moneyness** indicates whether an option would be profitable if exercised immediately:")

    moneyness_table = """
    | Status | Call Condition | Put Condition | Description |
    |--------|----------------|---------------|-------------|
    | *In-The-Money (ITM)* | $S > K$ | $S < K$ | Option has intrinsic value |
    | *At-The-Money (ATM)* | $S = K$ | $S = K$ | Stock price equals strike price |
    | *Out-of-The-Money (OTM)* | $S < K$ | $S > K$ | Option has no intrinsic value |
    """
    
    st.markdown(moneyness_table, text_alignment="center")
    
    st.subheader("**Intrinsic Value Formulas:**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Call Option:**")
        st.latex(r"C_{\text{IV}} = \max(0, S - K)")
        
    with col2:
        st.write("**Put Option:**")
        st.latex(r"P_{\text{IV}} = \max(0, K - S)")
    
    st.write("""
    **Where:**
    - **$S$** = Current Stock Price
    - **$K$** = Strike Price
    - **$C_{\t{IV}}$** = Call Intrinsic Value
    - **$P_{\t{IV}}$** = Put Intrinsic Value
    """)

    st.subheader("Option Greeks")
    st.write('The "Greeks" measure the sensitivity of the option price to various factors:')
    
    greek_cols = st.columns(5)
    
    greeks = [
        ("Δ (Delta)", r"\frac{\partial V}{\partial S}", "Price sensitivity to underlying asset"),
        ("Γ (Gamma)", r"\frac{\partial^2 V}{\partial S^2}", "Delta's sensitivity to price changes"),
        ("Θ (Theta)", r"\frac{\partial V}{\partial t}", "Time decay of option value"),
        ("ν (Vega)", r"\frac{\partial V}{\partial \sigma}", "Sensitivity to volatility changes"),
        ("ρ (Rho)", r"\frac{\partial V}{\partial r}", "Sensitivity to interest rate changes")
    ]
    
    for col, (name, formula, desc) in zip(greek_cols, greeks):
        with col:
            st.write("")
            st.write(f"**{name}**")
            st.latex(formula)
            st.caption(desc, text_alignment="center")
            st.write("")


# Calculator Page
with tab2:
    
    st.write("**Instructions:** Fill in all parameters in the sidebar and click 'Calculate Option Price' to compute the theoretical option value.")
    
    with st.form("Option_Pricing_Calculator"):
        st.subheader("Input Parameters")
        
        # Option Type
        option_type = st.selectbox(
            "**Option Type**",
            ["Call", "Put"],
            help="Select 'Call' for right to buy, 'Put' for right to sell."
        )
        
        # Stock Price
        S = st.number_input(
            "**Asset Price (S)**",
            min_value=0.01,
            value=100.0,
            step=1.0,
            help="Current market price of the underlying asset."
        )
        
        # Strike Price
        K = st.number_input(
            "**Strike Price (K)**",
            min_value=0.01,
            value=100.0,
            step=1.0,
            help="Price at which the option can be exercised."
        )
        
        # Time to Expiration
        T = st.number_input(
            "**Time to Expiration (T in years)**",
            min_value=0.01,
            value=1.0,
            step=0.25,
            help="Time remaining until the option expires."
        )
        
        # Risk-Free Rate
        r = st.number_input(
            "**Risk-Free Rate (r)**",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Annual risk-free interest rate (e.g., 0.05 for 5%)."
        )
        
        # Volatility
        sigma = st.slider(
            "**Volatility (σ)**",
            min_value=0.01,
            max_value=1.0,
            value=0.2,
            step=0.01,
            help="Annualised volatility of the underlying asset (e.g., 0.2 for 20%)."
        )
        
        calculate_button = st.form_submit_button(
            "Calculate Option Price",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if calculate_button:
        calc_col1, calc_col2 = st.columns(2)
        
        with calc_col1:
            st.header("Calculation Results")
            
        with calc_col2:
            # Calculate option price
            if option_type == "Call":
                price, d1, d2 = black_scholes_call(S, K, T, r, sigma)
                option_formula = r"C = S_0 N(d_1) - K e^{-rT} N(d_2)"
            else:
                price, d1, d2 = black_scholes_put(S, K, T, r, sigma)
                option_formula = r"P = K e^{-rT} N(-d_2) - S_0 N(-d_1)"
            
            # Display result
            st.metric(
                label=f"{option_type} Option Price", 
                value=f"${price:.2f}",
            )
            st.caption("*Theoretical Price based on the Black-Scholes Model*")
        
        st.write("---")
        
        # Display Greeks
        st.header("Option Greeks")
        
        # Calculate Greeks
        if option_type == "Call":
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01
        rho = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type == "Call" else norm.cdf(-d2)) * 0.01
        
        greek_cols = st.columns(5)
        greeks_data = [
            ("Δ (Delta)", f"{delta:.2f}", "Price sensitivity"),
            ("Γ (Gamma)", f"{gamma:.2f}", "Delta sensitivity"),
            ("Θ (Theta)", f"{theta:.2f}", "Daily time decay"),
            ("ν (Vega)", f"{vega:.2f}", "Volatility sensitivity"),
            ("ρ (Rho)", f"{rho:.2f}", "Interest rate sensitivity")
        ]
        
        for col, (name, value, desc) in zip(greek_cols, greeks_data):
            with col:
                st.metric(label=name, value=value)
                st.caption(desc)
        
        st.write("---")
        
        st.header("Payoff Visualisation")
        
        # Create payoff diagram
        spot_prices = np.linspace(S * 0.5, S * 1.5, 200)
        
        # Calculate payoffs for call and put options
        if option_type == "Call":
            intrinsic_value = np.maximum(spot_prices - K, 0)
            option_color = "green"
            # Breakeven points
            breakeven_long = K + price
            breakeven_short = K + price
            # Moneyness
            moneyness = "In-The-Money (ITM)" if S > K else "Out-of-The-Money (OTM)" if S < K else "At-The-Money (ATM)"
            intrinsic_val = max(S - K, 0)
        else:
            intrinsic_value = np.maximum(K - spot_prices, 0)
            option_color = "red"
            # Breakeven points
            breakeven_long = K - price
            breakeven_short = K - price
            # Moneyness
            moneyness = "In-The-Money (ITM)" if S < K else "Out-of-The-Money (OTM)" if S > K else "At-The-Money (ATM)"
            intrinsic_val = max(K - S, 0)
        
        time_val = price - intrinsic_val
        
        # Calculate profit/loss for long and short positions
        profit_loss_long = intrinsic_value - price  # Long: payoff - premium
        profit_loss_short = price - intrinsic_value  # Short: premium - payoff
        
        # Long Position Analysis
        st.subheader("**Long Position (Buyer)**")
        
        fig_long, (ax1_long, ax2_long) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Long Payoff Diagram
        ax1_long.plot(spot_prices, intrinsic_value, color=option_color, linewidth=3, 
                     label=f'Long {option_type} Payoff', alpha=0.8)
        ax1_long.fill_between(spot_prices, intrinsic_value, alpha=0.2, color=option_color)
        ax1_long.axvline(x=S, color='black', linestyle='--', alpha=0.7, linewidth=2, 
                        label=f'Current Price (${S:.2f})')
        ax1_long.axvline(x=K, color='gray', linestyle='--', alpha=0.7, linewidth=2, 
                        label=f'Strike Price (${K:.2f})')
        ax1_long.axvline(x=breakeven_long, color='purple', linestyle=':', alpha=0.7, 
                        linewidth=1.5, label=f'Breakeven (${breakeven_long:.2f})')
        ax1_long.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        ax1_long.set_xlabel('Stock Price at Expiration ($)', fontsize=12)
        ax1_long.set_ylabel('Option Payoff ($)', fontsize=12)
        ax1_long.set_title(f'Long {option_type} Option: Payoff Diagram', fontsize=14, fontweight='bold')
        ax1_long.legend(loc='best')
        ax1_long.grid(True, alpha=0.3)
        
        # Plot 2: Long Profit/Loss Diagram
        ax2_long.plot(spot_prices, profit_loss_long, color='purple', linewidth=3, 
                     label='Net Profit/Loss', alpha=0.8)
        
        # Fill profit and loss zones for long position
        profit_zone_long = profit_loss_long >= 0
        loss_zone_long = profit_loss_long < 0
        
        ax2_long.fill_between(spot_prices, profit_loss_long, where=profit_zone_long, 
                             alpha=0.2, color='green', label='Profit Zone')
        ax2_long.fill_between(spot_prices, profit_loss_long, where=loss_zone_long, 
                             alpha=0.2, color='red', label='Loss Zone')
        
        # Key reference lines
        ax2_long.axvline(x=S, color='black', linestyle='--', alpha=0.7, linewidth=2, 
                        label=f'Current Price (${S:.2f})')
        ax2_long.axvline(x=K, color='gray', linestyle='--', alpha=0.7, linewidth=2, 
                        label=f'Strike Price (${K:.2f})')
        ax2_long.axvline(x=breakeven_long, color='purple', linestyle=':', alpha=0.7, 
                        linewidth=1.5, label=f'Breakeven (${breakeven_long:.2f})')
        
        ax2_long.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax2_long.axhline(y=-price, color='red', linestyle=':', alpha=0.5, linewidth=1.5, 
                        label=f'Max Loss: -${price:.2f}')
        
        # Maximum profit for long position
        if option_type == "Call":
            ax2_long.axhline(y=spot_prices[-1] - K - price, color='darkgreen', 
                           linestyle=':', alpha=0.3, linewidth=1,
                           label='Max Profit: Unlimited')
        else:
            max_profit_long = K - price
            ax2_long.axhline(y=max_profit_long, color='darkgreen', linestyle=':', 
                           alpha=0.3, linewidth=1,
                           label=f'Max Profit: ${max_profit_long:.2f}')
        
        ax2_long.set_xlabel('Stock Price at Expiration ($)', fontsize=12)
        ax2_long.set_ylabel('Net Profit/Loss ($)', fontsize=12)
        ax2_long.set_title(f'Long {option_type} Option: Profit/Loss Diagram', fontsize=14, fontweight='bold')
        ax2_long.legend(loc='best')
        ax2_long.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_long)
        
        col_long1, col_long2 = st.columns(2)
        
        with col_long1:
            st.write("**Long Position Summary:**")
            st.write(f"- **Position**: Buyer of {option_type.lower()} option")
            st.write(f"- **Moneyness**: {moneyness}")
            st.write(f"- **Premium Paid**: ${price:.2f}")
            st.write(f"- **Breakeven**: ${breakeven_long:.2f}")
            st.write("- **Intrinsic Value**: ${:,.2f}".format(intrinsic_val))
            st.write("- **Time Value**: ${:,.2f}".format(time_val))
        
        with col_long2:
            if option_type == "Call":
                st.write("**For Long Call:**")
                st.write("- **Profit When**: Stock > ${:,.2f}".format(breakeven_long))
                st.write("- **Loss When**: Stock ≤ ${:,.2f}".format(breakeven_long))
                st.write("- **Maximum Loss**: -\\${:,.2f} (premium paid)".format(price))
                st.write("- **Maximum Profit**: Unlimited (stock can rise indefinitely)")
                st.write("- **Strategy**: Bullish - Expect stock to rise")
                
            else:
                max_profit_long = (K - price) * 100
                st.write("**For Long Put:**")
                st.write("- **Profit When**: Stock < ${:,.2f}".format(breakeven_long))
                st.write("- **Loss When**: Stock ≥ ${:,.2f}".format(breakeven_long))
                st.write("- **Maximum Loss**: -\\${:,.2f} (premium paid)".format(price))
                st.write("- **Maximum Profit**: \\${:,.2f} (if stock goes to $0)".format(max_profit_long))
                st.write("- **Strategy**: Bearish - Expect stock to fall")
    
        st.write("---")
        
        # Short Position Analysis
        st.subheader("**Short Position (Seller/Writer)**")
        
        fig_short, (ax1_short, ax2_short) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Short Payoff Diagram (mirror image of long)
        short_payoff_color = "blue" if option_type == "Call" else "orange"
        ax1_short.plot(spot_prices, -intrinsic_value, color=short_payoff_color, linewidth=3, 
                      label=f'Short {option_type} Payoff', alpha=0.8)
        ax1_short.fill_between(spot_prices, -intrinsic_value, alpha=0.2, color=short_payoff_color)
        ax1_short.axvline(x=S, color='black', linestyle='--', alpha=0.7, linewidth=2, 
                         label=f'Current Price (${S:.2f})')
        ax1_short.axvline(x=K, color='gray', linestyle='--', alpha=0.7, linewidth=2, 
                         label=f'Strike Price (${K:.2f})')
        ax1_short.axvline(x=breakeven_short, color='purple', linestyle=':', alpha=0.7, 
                         linewidth=1.5, label=f'Breakeven (${breakeven_short:.2f})')
        ax1_short.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        ax1_short.set_xlabel('Stock Price at Expiration ($)', fontsize=12)
        ax1_short.set_ylabel('Option Payoff ($)', fontsize=12)
        ax1_short.set_title(f'Short {option_type} Option: Payoff Diagram', fontsize=14, fontweight='bold')
        ax1_short.legend(loc='best')
        ax1_short.grid(True, alpha=0.3)
        
        # Plot 2: Short Profit/Loss Diagram
        ax2_short.plot(spot_prices, profit_loss_short, color='purple', linewidth=3, 
                      label='Net Profit/Loss', alpha=0.8)
        
        # Fill profit and loss zones for short position
        profit_zone_short = profit_loss_short >= 0
        loss_zone_short = profit_loss_short < 0
        
        ax2_short.fill_between(spot_prices, profit_loss_short, where=profit_zone_short, 
                              alpha=0.2, color='green', label='Profit Zone')
        ax2_short.fill_between(spot_prices, profit_loss_short, where=loss_zone_short, 
                              alpha=0.2, color='red', label='Loss Zone')
        
        # Key reference lines
        ax2_short.axvline(x=S, color='black', linestyle='--', alpha=0.7, linewidth=2, 
                         label=f'Current Price (${S:.2f})')
        ax2_short.axvline(x=K, color='gray', linestyle='--', alpha=0.7, linewidth=2, 
                         label=f'Strike Price (${K:.2f})')
        ax2_short.axvline(x=breakeven_short, color='purple', linestyle=':', alpha=0.7, 
                         linewidth=1.5, label=f'Breakeven (${breakeven_short:.2f})')
        
        ax2_short.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax2_short.axhline(y=price, color='darkgreen', linestyle=':', alpha=0.5, linewidth=1.5, 
                         label=f'Max Profit: ${price:.2f}')
        
        # Maximum loss for short position
        if option_type == "Call":
            max_loss_short = -(spot_prices[-1] - K) + price
            ax2_short.axhline(y=max_loss_short, color='red', linestyle=':', alpha=0.3, 
                            linewidth=1, label='Max Loss: Unlimited')
        else:
            max_loss_short = -K + price
            ax2_short.axhline(y=max_loss_short, color='red', linestyle=':', alpha=0.3, 
                            linewidth=1, label=f'Max Loss: ${max_loss_short:.2f}')
        
        ax2_short.set_xlabel('Stock Price at Expiration ($)', fontsize=12)
        ax2_short.set_ylabel('Net Profit/Loss ($)', fontsize=12)
        ax2_short.set_title(f'Short {option_type} Option: Profit/Loss Diagram', fontsize=14, fontweight='bold')
        ax2_short.legend(loc='best')
        ax2_short.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_short)
        
        # Short position analysis
        col_short1, col_short2 = st.columns(2)
        with col_short1:
            st.write("**Short Position Summary:**")
            st.write(f"- **Position**: Seller/writer of {option_type.lower()} option")
            st.write(f"- **Moneyness**: {moneyness}")
            st.write(f"- **Premium Received**: ${price:.2f}")
            st.write(f"- **Breakeven**: ${breakeven_short:.2f}")
            st.write("- **Intrinsic Value**: ${:,.2f}".format(intrinsic_val))
            st.write("- **Time Value**: ${:,.2f}".format(time_val))
        
        with col_short2:
            if option_type == "Call":
                st.write("**For Short Call:**")
                st.write("- **Profit When**: Stock < ${:,.2f}".format(breakeven_short))
                st.write("- **Loss When**: Stock ≥ ${:,.2f}".format(breakeven_short))
                st.write("- **Maximum Profit**: \\${:,.2f} (premium received)".format(price))
                st.write("- **Maximum Loss**: Unlimited (stock can rise indefinitely)")
                st.write("- **Strategy**: Neutral to bearish - Expect stock to stay flat or fall")
                st.write("- **Risk**: Naked call - Must deliver shares if exercised")
            else:
                max_loss_short = (K - price) * 100
                st.write("**For Short Put:**")
                st.write("- **Profit When**: Stock > ${:,.2f}".format(breakeven_short))
                st.write("- **Loss When**: Stock ≤ ${:,.2f}".format(breakeven_short))
                st.write("- **Maximum Profit**: \\${:,.2f} (premium received)".format(price))
                st.write("- **Maximum Loss**: -\\${:,.2f} (if stock goes to $0)".format(max_loss_short))
                st.write("- **Strategy**: Neutral to bullish - Expect stock to stay flat or rise")
                st.write("- **Risk**: Must buy shares at strike price if exercised")

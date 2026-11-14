import streamlit as st
import math
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Bond Price Calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling for dark mode support
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Light mode colors */
    .main-header {
        color: #1E3A8A;
    }
    
    /* Sub-header styling - visible in both modes */
    .sub-header {
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Light mode colors for sub-headers */
    .sub-header {
        color: #374151;
    }
    
    /* Dark mode colors */
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #60A5FA;
        }
        .sub-header {
            color: #F3F4F6 !important;
        }
    }
    
    /* Force dark mode styling */
    [data-theme="dark"] .main-header {
        color: #60A5FA !important;
    }
    
    [data-theme="dark"] .sub-header {
        color: #F3F4F6 !important;
    }
    
    /* Result box styling */
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        background-color: #f8f9fa;
    }
    
    /* Dark mode override for result box */
    @media (prefers-color-scheme: dark) {
        .result-box {
            background-color: #1a1a1a !important;
            border-color: #444 !important;
        }
        .result-box h4 {
            color: #fafafa !important;
        }
    }
    
    [data-theme="dark"] .result-box {
        background-color: #1a1a1a !important;
        border-color: #444 !important;
    }
    
    .result-box h4 {
        color: #262730;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    [data-theme="dark"] .result-box h4 {
        color: #fafafa !important;
    }
    
    /* Price difference styling */
    .price-difference {
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .positive-diff {
        color: #10B981;
    }
    
    .negative-diff {
        color: #EF4444;
    }
    
    /* Table styling for negative values */
    .negative-value {
        color: #EF4444;
    }
</style>
""", unsafe_allow_html=True)

# Constants
FREQUENCY_OPTIONS = {
    "Annual": 1,
    "Semi-annual": 2,
    "Quarterly": 4,
    "Monthly": 12,
    "Daily": 365
}

class BondCalculator:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'bond_type' not in st.session_state:
            st.session_state.bond_type = "Zero Coupon Bond"
        if 'calculation_done' not in st.session_state:
            st.session_state.calculation_done = False
    
    def calculate_discrete_price(self, params):
        """Calculate bond price using discrete compounding"""
        if params['bond_type'] == "Zero Coupon Bond":
            r_per_period = params['interest_rate'] / params['compounding_periods']
            n_periods = params['maturity'] * params['compounding_periods']
            price = params['principal'] / ((1 + r_per_period) ** n_periods)
            return price
        else:
            r_per_period = params['interest_rate'] / params['compounding_periods']
            n_periods = params['maturity'] * params['compounding_periods']
            coupon_payment = (params['coupon_rate'] * params['principal']) / params['payments_per_year']
            
            pv_coupons = 0
            for k in range(1, int(n_periods) + 1):
                pv_coupons += coupon_payment / ((1 + r_per_period) ** k)
            
            pv_principal = params['principal'] / ((1 + r_per_period) ** n_periods)
            
            price = pv_coupons + pv_principal
            return price
    
    def calculate_continuous_price(self, params):
        """Calculate bond price using continuous compounding"""
        if params['bond_type'] == "Zero Coupon Bond":
            price = params['principal'] * math.exp(-params['interest_rate'] * params['maturity'])
            return price
        else:
            price = 0
            coupon_payment = (params['coupon_rate'] * params['principal']) / params['payments_per_year']
            
            for i in range(1, int(params['maturity'] * params['payments_per_year']) + 1):
                t = i / params['payments_per_year']
                price += coupon_payment * math.exp(-params['interest_rate'] * t)
            
            price += params['principal'] * math.exp(-params['interest_rate'] * params['maturity'])
            return price
    
    def calculate_macaulay_duration(self, params, discrete_price):
        """Calculate Macaulay Duration for the bond"""
        if params['bond_type'] == "Zero Coupon Bond":
            return params['maturity']
        else:
            r_per_period = params['interest_rate'] / params['compounding_periods']
            n_periods = params['maturity'] * params['compounding_periods']
            coupon_payment = (params['coupon_rate'] * params['principal']) / params['payments_per_year']
            
            weighted_sum = 0
            for k in range(1, int(n_periods) + 1):
                time_years = k / params['compounding_periods']
                cash_flow = coupon_payment
                present_value = cash_flow / ((1 + r_per_period) ** k)
                weighted_sum += time_years * present_value
            
            time_years = params['maturity']
            cash_flow = params['principal']
            present_value = cash_flow / ((1 + r_per_period) ** n_periods)
            weighted_sum += time_years * present_value
            
            return weighted_sum / discrete_price
    
    def calculate_modified_duration(self, params, macaulay_duration):
        """Calculate Modified Duration for the bond"""
        r_per_period = params['interest_rate'] / params['compounding_periods']
        return macaulay_duration / (1 + r_per_period)
    
    def calculate_convexity(self, params, discrete_price):
        """Calculate Convexity for the bond"""
        if params['bond_type'] == "Zero Coupon Bond":
            r_per_period = params['interest_rate'] / params['compounding_periods']
            n_periods = params['maturity'] * params['compounding_periods']
            convexity = (n_periods * (n_periods + 1)) / ((1 + r_per_period) ** 2)
            return convexity / (params['compounding_periods'] ** 2)
        else:
            r_per_period = params['interest_rate'] / params['compounding_periods']
            n_periods = params['maturity'] * params['compounding_periods']
            coupon_payment = (params['coupon_rate'] * params['principal']) / params['payments_per_year']
            
            convexity_sum = 0
            for k in range(1, int(n_periods) + 1):
                time_years = k / params['compounding_periods']
                cash_flow = coupon_payment
                present_value = cash_flow / ((1 + r_per_period) ** k)
                convexity_sum += time_years * (time_years + 1/params['compounding_periods']) * present_value
            
            time_years = params['maturity']
            cash_flow = params['principal']
            present_value = cash_flow / ((1 + r_per_period) ** n_periods)
            convexity_sum += time_years * (time_years + 1/params['compounding_periods']) * present_value
            
            return convexity_sum / (discrete_price * ((1 + r_per_period) ** 2))
    
    def create_input_section(self):
        """Create the input section in sidebar"""
        st.sidebar.header("Bond Parameters")
        
        bond_type = st.sidebar.selectbox(
            "Bond Type",
            ["Zero Coupon Bond", "Coupon Bond"],
            index=0 if st.session_state.bond_type == "Zero Coupon Bond" else 1,
            key="bond_type_select"
        )
        st.session_state.bond_type = bond_type
        
        principal = st.sidebar.number_input(
            "Principal/Face Value ($)",
            min_value=0.01,
            value=1000.0,
            step=100.0,
            format="%.2f"
        )
        
        maturity = st.sidebar.number_input(
            "Time to Maturity (years)",
            min_value=1.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
            format="%.1f"
        )
        
        interest_rate = st.sidebar.number_input(
            "Annual Interest Rate (Yield)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
            format="%.1f") / 100
        
        coupon_rate = 0.0
        if bond_type == "Coupon Bond":
            coupon_rate = st.sidebar.number_input(
                "Annual Coupon Rate",
                min_value=0.0,
                max_value=100.0,
                value=3.5,
                step=0.1,
                format="%.1f") / 100
        
        frequency = st.sidebar.selectbox(
            "Compounding/Payment Frequency",
            list(FREQUENCY_OPTIONS.keys()),
            index=0
        )
        compounding_periods = FREQUENCY_OPTIONS[frequency]
        payments_per_year = FREQUENCY_OPTIONS[frequency] if bond_type == "Coupon Bond" else 1
        
        if st.sidebar.button("Calculate Bond Price", type="primary", use_container_width=True):
            params = {
                'bond_type': bond_type,
                'principal': principal,
                'maturity': maturity,
                'interest_rate': interest_rate,
                'coupon_rate': coupon_rate,
                'compounding_periods': compounding_periods,
                'payments_per_year': payments_per_year,
                'frequency_name': frequency
            }
            st.session_state.calculation_params = params
            st.session_state.calculation_done = True
        
        if st.sidebar.button("Reset", use_container_width=True):
            st.session_state.calculation_done = False
    
    def format_currency(self, value):
        """Format currency with negative sign before dollar sign"""
        if value < 0:
            return f"-${abs(value):,.2f}"
        return f"${value:,.2f}"
    
    def display_results(self):
        """Display calculation results"""
        params = st.session_state.calculation_params
        
        discrete_price = self.calculate_discrete_price(params)
        continuous_price = self.calculate_continuous_price(params)
        
        price_diff = continuous_price - discrete_price
        price_diff_pct = (price_diff / discrete_price) * 100 if discrete_price != 0 else 0
        
        st.title("Bond Price Calculator", text_alignment="center")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Bond Information")
            
            info_data = {
                "Parameter": ["Bond Type", "Principal/Face Value", "Time to Maturity", 
                             "Annual Interest Rate", "Compounding Frequency"],
                "Value": [
                    params['bond_type'],
                    f"${params['principal']:,.2f}",
                    f"{params['maturity']:.1f} years",
                    f"{params['interest_rate']*100:.2f}%",
                    params['frequency_name']
                ]
            }
            
            if params['bond_type'] == "Coupon Bond":
                info_data["Parameter"].insert(4, "Annual Coupon Rate")
                info_data["Value"].insert(4, f"{params['coupon_rate']*100:.2f}%")
            
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Price Results")
            
            cols = st.columns(2)
            with cols[0]:
                st.metric(
                    label="Discrete Model",
                    value=f"${discrete_price:,.2f}",
                    delta=None
                )
            with cols[1]:
                st.metric(
                    label="Continuous Model",
                    value=f"${continuous_price:,.2f}",
                    delta=None
                )
        
            formatted_price_diff = self.format_currency(price_diff)
            diff_sign_pct = "+" if price_diff_pct >= 0 else ""
            
            diff_color = "positive-diff" if price_diff >= 0 else "negative-diff"
            
            st.markdown(
                f'<div class="result-box">'
                f'<h4>Price Difference</h4>'
                f'<p class="price-difference {diff_color}">'
                f'Continuous - Discrete: {formatted_price_diff} '
                f'({diff_sign_pct}{price_diff_pct:.2f}%)'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        tab1, tab2 = st.tabs(["Detailed Analysis", "Explanation"])
        
        with tab1:
            self.display_detailed_analysis(params, discrete_price, continuous_price)
            self.display_duration_analysis(params, discrete_price)
            
        with tab2:
            self.display_explanation(params)
    
    def display_detailed_analysis(self, params, discrete_price, continuous_price):
        """Display detailed analysis including cash flows"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Payment Schedule")
            st.write("(Coupon Bonds Only)")
            if params['bond_type'] == "Coupon Bond":
                num_payments = int(params['maturity'] * params['payments_per_year'])
                coupon_payment = (params['coupon_rate'] * params['principal']) / params['payments_per_year']
                
                payments = []
                for i in range(1, num_payments + 1):
                    payment_time = i / params['payments_per_year']
                    r_per_period = params['interest_rate'] / params['compounding_periods']
                    discount_factor_discrete = 1 / ((1 + r_per_period) ** (i))
                    pv_discrete = coupon_payment * discount_factor_discrete
                    
                    discount_factor_continuous = math.exp(-params['interest_rate'] * payment_time)
                    pv_continuous = coupon_payment * discount_factor_continuous
                    
                    payments.append({
                        "Payment #": i,
                        "Time (years)": f"{payment_time:.2f}",
                        "Coupon Payment": f"${coupon_payment:.2f}",
                        "PV (Discrete)": f"${pv_discrete:.2f}",
                        "PV (Continuous)": f"${pv_continuous:.2f}"
                    })
                
                discount_factor_discrete_principal = 1 / ((1 + params['interest_rate']/params['compounding_periods']) ** 
                                                        (params['maturity'] * params['compounding_periods']))
                discount_factor_continuous_principal = math.exp(-params['interest_rate'] * params['maturity'])
                
                payments.append({
                    "Payment #": "Principal",
                    "Time (years)": f"{params['maturity']:.2f}",
                    "Coupon Payment": f"${params['principal']:.2f}",
                    "PV (Discrete)": f"${params['principal'] * discount_factor_discrete_principal:.2f}",
                    "PV (Continuous)": f"${params['principal'] * discount_factor_continuous_principal:.2f}"
                })
                
                payments_df = pd.DataFrame(payments)
                st.dataframe(payments_df, use_container_width=True, hide_index=True)
            else:
                st.info("Payment schedule is only applicable for Coupon Bonds.")
        
        with col2:
            st.subheader("Price Comparison")
            price_data = pd.DataFrame({
                'Model': ['Discrete', 'Continuous'],
                'Price': [discrete_price, continuous_price]
            })
            
            st.bar_chart(price_data.set_index('Model'))
    
    def display_duration_analysis(self, params, discrete_price):
        """Display duration and risk analysis"""
        st.subheader("Duration & Risk Analysis")
        
        macaulay_duration = self.calculate_macaulay_duration(params, discrete_price)
        modified_duration = self.calculate_modified_duration(params, macaulay_duration)
        convexity = self.calculate_convexity(params, discrete_price)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Macaulay Duration",
                value=f"{macaulay_duration:.2f} years",
                delta=None,
                help="Weighted average time to receive cash flows"
            )
        
        with col2:
            st.metric(
                label="Modified Duration",
                value=f"{modified_duration:.2f} years",
                delta=None,
                help="Price sensitivity to interest rate changes"
            )
        
        with col3:
            st.metric(
                label="Convexity",
                value=f"{convexity:.2f}",
                delta=None,
                help="Curvature of price-yield relationship"
            )
        
        st.subheader("Price Sensitivity Estimates")
        
        col_yield1, col_yield2 = st.columns(2)
        
        with col_yield1:
            yield_change = st.slider(
                "Yield Change (percentage points)",
                min_value=-5.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                format="%.1f%%"
            ) / 100
        
        with col_yield2:
            price_change_duration = -modified_duration * yield_change * discrete_price
            price_change_convexity = 0.5 * convexity * (yield_change ** 2) * discrete_price
            total_price_change = price_change_duration + price_change_convexity
            new_price_estimate = discrete_price + total_price_change
            
            st.metric(
                label="Estimated New Price",
                value=f"${new_price_estimate:,.2f}",
                delta=f"{total_price_change:,.2f}",
                delta_color="inverse"
            )
        
        st.subheader("Detailed Price Change Calculation")

        col_calc_data1, col_calc_data2 = st.columns(2)

        with col_calc_data1:
            calc_data = {
                "Component": [
                    "Current Bond Price",
                    "Modified Duration",
                    "Convexity",
                    "Yield Change",
                    "Duration Effect",
                    "Convexity Effect",
                    "Total Price Change",
                    "Estimated New Price"
                ],
                "Value": [
                    f"${discrete_price:,.2f}",
                    f"{modified_duration:.2f}",
                    f"{convexity:.2f}",
                    f"{yield_change*100:+.2f}%",
                    self.format_currency(price_change_duration),
                    self.format_currency(price_change_convexity),
                    self.format_currency(total_price_change),
                    f"${new_price_estimate:,.2f}"
                ]
            }
            
            calc_df = pd.DataFrame(calc_data)

            st.dataframe(calc_df, use_container_width=True, hide_index=True)

        with col_calc_data2:
            st.subheader("Formulas Used")
            
            st.markdown("""
            - **Modified Duration**: $D_{mod} = \\frac{D_{mac}}{1 + \\frac{y}{m}}$
            
            - **Convexity**: $C = \\frac{\\sum_{t=1}^{T} t(t+\\frac{1}{m}) PV(CF_t)}{P(1+\\frac{y}{m})^2}$
            
            - **Duration Effect**: $\\Delta P_{duration} = -D_{mod} \\times \\Delta y \\times P$
            
            - **Convexity Effect**: $\\Delta P_{convexity} = 0.5 \\times C \\times (\\Delta y)^2 \\times P$
            
            - **Total Price Change**: $\\Delta P_{total} = \\Delta P_{duration} + \\Delta P_{convexity}$
            
            - **New Price**: $P_{new} = P + \\Delta P_{total}$
            
            """)
        
        st.subheader("Key Insights")
        
        insights = [
            f"**Interest Rate Sensitivity**: A 1% increase in yield would decrease the bond price by approximately **{modified_duration:.2f}%**.",
            f"**Cash Flow Timing**: The weighted average time to receive all cash flows is **{macaulay_duration:.2f} years**.",
            f"**Convexity Benefit**: Positive convexity means the bond's price increases more when yields fall than it decreases when yields rise.",
            f"**Accuracy**: For small yield changes (±1%), duration alone provides a good estimate. For larger changes, convexity adjustment is important."
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
    
    def display_explanation(self, params):
        """Display explanation of the calculations with all formulas in LaTeX"""
        st.header("How Bond Prices Are Calculated")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("*Discrete Compounding Model:*")
            if params['bond_type'] == "Zero Coupon Bond":
                st.latex(r'''
                P = \frac{F}{(1 + \frac{r}{n})^{n \cdot t}}
                ''')
                st.markdown(r"""
                Where:
                - $P$ = Price of the bond
                - $F$ = Face value/principal
                - $r$ = Annual interest rate (decimal)
                - $n$ = Compounding periods per year
                - $t$ = Time to maturity in years
                """)
            else:
                st.latex(r'''
                P = \sum_{k=1}^{n \cdot t} \frac{C}{(1 + \frac{r}{n})^{k}} + \frac{F}{(1 + \frac{r}{n})^{n \cdot t}}
                ''')
                st.markdown(r"""
                Where:
                - $P$ = Price of the bond
                - $C$ = Coupon payment per period $= \frac{\text{coupon rate} \times F}{\text{payments per year}}$
                - $F$ = Face value/principal
                - $r$ = Annual interest rate (decimal)
                - $n$ = Compounding periods per year
                - $t$ = Time to maturity in years
                """)
        
        with col2:
            st.subheader("*Continuous Compounding Model:*")
            if params['bond_type'] == "Zero Coupon Bond":
                st.latex(r'''
                P = F \cdot e^{-r \cdot t}
                ''')
                st.markdown(r"""
                Where:
                - $P$ = Price of the bond
                - $F$ = Face value/principal
                - $r$ = Annual interest rate (decimal)
                - $t$ = Time to maturity in years
                - $e$ = Euler's number ($\approx 2.71828$)
                """)
            else:
                st.latex(r'''
                P = \sum_{i=1}^{m \cdot t} C \cdot e^{-r \cdot t_i} + F \cdot e^{-r \cdot t}
                ''')
                st.markdown(r"""
                Where:
                - $P$ = Price of the bond
                - $C$ = Coupon payment per period
                - $F$ = Face value/principal
                - $r$ = Annual interest rate (decimal)
                - $t_i$ = Time of $i$-th coupon payment
                - $t$ = Time to maturity in years
                - $m$ = Payments per year
                """)

        st.write("---")
        
        st.subheader("Duration Formulas")
        
        col_dur1, col_dur2, col_dur3 = st.columns(3)
        
        with col_dur1:
            st.subheader("*Macaulay Duration:*")
            st.latex(r'''
            D_{\text{mac}} = \frac{\sum_{t=1}^{T} t \cdot PV(CF_t)}{P}
            ''')
            st.markdown(r"""
            Where:
            - $D_{\text{mac}}$ = Macaulay Duration (years)
            - $t$ = Time of cash flow (years)
            - $PV(CF_t)$ = Present value of cash flow at time $t$
            - $P$ = Bond price
            - $T$ = Maturity
            """)
        
        with col_dur2:
            st.subheader("*Modified Duration:*")
            st.latex(r'''
            D_{\text{mod}} = \frac{D_{\text{mac}}}{1 + \frac{y}{m}}
            ''')
            st.markdown(r"""
            Where:
            - $D_{\text{mod}}$ = Modified Duration (years)
            - $D_{\text{mac}}$ = Macaulay Duration (years)
            - $y$ = Yield to maturity (annual)
            - $m$ = Number of compounding periods per year
            """)
        with col_dur3:
            st.subheader("*Price Sensitivity:*")
            st.latex(r'''
            \frac{\Delta P}{P} \approx -D_{\text{mod}} \cdot \Delta y
            ''')
            st.markdown(r"""
            Where:
            - $\Delta P$ = Change in bond price
            - $\Delta y$ = Change in yield
            """)
        
        st.subheader("Convexity Formula")
        
        col_conv1, col_conv2 = st.columns([2, 1])
        
        with col_conv1:
            st.latex(r'''
            C = \frac{\sum_{t=1}^{T} t(t+1) \cdot PV(CF_t)}{P \cdot (1 + \frac{y}{m})^2}
            ''')
            st.markdown(r"""
            Where:
            - $C$ = Convexity
            - $t$ = Time of cash flow (years)
            - $PV(CF_t)$ = Present value of cash flow at time $t$
            - $P$ = Bond price
            - $y$ = Yield to maturity (annual)
            - $m$ = Number of compounding periods per year
            - $T$ = Maturity
            """)
        
        with col_conv2:
            st.latex(r'''
            \Delta P_{\text{convexity}} = \frac{1}{2} \cdot C \cdot (\Delta y)^2 \cdot P
            ''')
            st.markdown(r"""
            Where:
            - $\Delta P_{\text{convexity}}$ = Price change due to convexity
            - $C$ = Convexity
            - $\Delta y$ = Change in yield
            - $P$ = Bond price
            """)
        
        st.subheader("Price Sensitivity with Convexity Adjustment")
        st.latex(r'''
        \Delta P \approx -D_{\text{mod}} \cdot \Delta y \cdot P + \frac{1}{2} \cdot C \cdot (\Delta y)^2 \cdot P
        ''')
        st.latex(r'''
        P_{\text{new}} = P + \Delta P
        ''')
        st.markdown(r"""
        Where:
        - $P_{\text{new}}$ = New estimated bond price
        - $P$ = Current bond price
        - $\Delta P$ = Total estimated price change
        """)
        
        st.write("---")
        
        st.subheader("Understanding Duration & Convexity")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.markdown("""
            #### *Macaulay Duration*
            - **Definition**: Weighted average time to receive cash flows
            - **Interpretation**: 
                - Higher duration = More price sensitivity
                - Zero-coupon bond duration = Maturity
                - Coupon bond duration < Maturity
            """)
        
        with col_info2:
            st.markdown("""
            #### *Modified Duration*
            - **Definition**: Price sensitivity to yield changes
            - **Application**: 
                - For 1% yield increase: Price ↓ by Modified Duration %
                - Key measure for interest rate risk
            """)
        
        with col_info3:
            st.markdown("""
            #### *Convexity*
            - **Definition**: Measures curvature of price-yield curve
            - **Importance**: 
                - Adjusts duration for large yield changes
                - Positive convexity = Price increases more than duration predicts
                - Higher convexity = Better risk-return profile
                - Always positive for non-callable bonds
            """)
        
        st.write("---")
        st.subheader("Key Insights")
        
        insights = [
            "**Interest Rate Sensitivity**: Bond prices move inversely to interest rates.",
            "**Time Value**: Longer maturity bonds are more sensitive to rate changes.",
            "**Compounding Effect**: More frequent compounding leads to slightly lower prices.",
            "**Coupon Effect**: Higher coupon bonds are less sensitive to rate changes.",
            "**Model Difference**: Continuous compounding typically gives slightly different results than discrete."
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
    
    def display_welcome(self):
        """Display welcome message when no calculation has been done"""
        st.title("Bond Price Calculator", text_alignment="center")
        
        st.markdown("""
            ### **How to Use:**
            1. Adjust parameters in the **sidebar**
            2. Click **"Calculate Bond Price"**
            3. Explore results in the **tabs below**
            """)
        
        st.write("---")
        
        st.markdown("""
            This tool helps you calculate bond prices using two different models:
            
            ### 🔷 **Discrete Compounding Model**
            - Uses periodic compounding (annual, semi-annual, etc.)
            - More common in practice
            - Formula:""")
        
        st.latex(r'''
                P = \sum_{k=1}^{n \cdot t} \frac{C}{(1 + \frac{r}{n})^{k}} + \frac{F}{(1 + \frac{r}{n})^{n \cdot t}}
                ''')
        
        st.markdown("""    
            ### 🔶 **Continuous Compounding Model**
            - Assumes continuous compounding
            - Uses Euler's number (e)
            - Often used in theoretical finance
            - Formula:""")  
        
        st.latex(r'''
                P = \sum_{i=1}^{m \cdot t} C \cdot e^{-r \cdot t_i} + F \cdot e^{-r \cdot t}
                ''')
        
        st.write("---")
        
        st.markdown("""    
            ### **Features:**
            1. **Zero Coupon Bonds**: Calculate prices for bonds with no periodic interest payments
            2. **Coupon Bonds**: Calculate prices for bonds with regular interest payments
            3. **Detailed Breakdown**: View payment schedules and present values
            4. **Visual Comparisons**: Charts to understand relationships 
            5. **Duration Analysis**: Calculate Macaulay and Modified Duration
            6. **Risk Assessment**: Measure interest rate sensitivity
            7. **Convexity**: Account for non-linear price-yield relationship
            8. **Price Sensitivity**: Estimate price changes for yield shifts
            """)
        
        st.write("---")    
        
        st.subheader("Bond Pricing Concepts")
        
        concepts_col1, concepts_col2, concepts_col3 = st.columns(3)
        
        with concepts_col1:
            st.markdown("""
            #### **Zero Coupon Bonds**
            - No periodic interest payments
            - Issued at a discount to face value
            - Price = Present value of face value
            - Higher price sensitivity to rate changes
            - Duration = Maturity
            """)
        
        with concepts_col2:
            st.markdown("""
            #### **Coupon Bonds**
            - Regular interest payments
            - Face value repaid at maturity
            - Price = PV of coupons + PV of face value
            - Lower duration than zero-coupon bonds
            - Duration < Maturity
            """)
        
        with concepts_col3:
            st.markdown("""
            #### **Duration Concepts**
            1. **Macaulay Duration**: Weighted average cash flow time
            2. **Modified Duration**: Price sensitivity to yield
            3. **Convexity**: Curvature adjustment
            """)
    
    def run(self):
        """Main application runner"""
        with st.sidebar:
            self.create_input_section()
        
        if st.session_state.get('calculation_done', False):
            self.display_results()
        else:
            self.display_welcome()

if __name__ == "__main__":
    calculator = BondCalculator()
    calculator.run()

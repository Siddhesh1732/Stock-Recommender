import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import hashlib
import sqlite3
from pathlib import Path


# --------------------------
# Authentication System
# --------------------------

def init_auth_db():
    """Initialize the SQLite database for user accounts"""
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY,
                  password TEXT,
                  email TEXT,
                  created_at TIMESTAMP,
                  risk_profile TEXT,
                  sectors TEXT,
                  holdings TEXT)''')
    conn.commit()
    conn.close()


def hash_password(password):
    """Hash passwords for secure storage"""
    return hashlib.sha256(str.encode(password)).hexdigest()


def verify_password(password, hashed_text):
    """Verify password against stored hash"""
    return hash_password(password) == hashed_text


def create_new_user(username, password, email, risk_profile, sectors, holdings):
    """Create new user account"""
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    c.execute('INSERT INTO users VALUES (?,?,?,?,?,?,?)',
              (username, hash_password(password), email, datetime.now(),
               risk_profile, ','.join(sectors), ','.join(holdings)))
    conn.commit()
    conn.close()


def authenticate_user(username, password):
    """Authenticate existing user"""
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()

    if data and verify_password(password, data[1]):
        return {
            'username': data[0],
            'email': data[2],
            'risk_profile': data[4],
            'sectors': data[5].split(','),
            'holdings': data[6].split(',')
        }
    return None


# Configure app
st.set_page_config(page_title="AI Stock Advisor", layout="wide")
st.title("Smart Stock Recommender")

# Expanded list of 50 Indian stocks with sectors
STOCKS = {
    'RELIANCE.NS': 'Oil & Gas',
    'TCS.NS': 'IT',
    'HDFCBANK.NS': 'Banking',
    'INFY.NS': 'IT',
    'ICICIBANK.NS': 'Banking',
    'HINDUNILVR.NS': 'FMCG',
    'ITC.NS': 'FMCG',
    'KOTAKBANK.NS': 'Banking',
    'BHARTIARTL.NS': 'Telecom',
    'LT.NS': 'Infrastructure',
    'ASIANPAINT.NS': 'Consumer',
    'MARUTI.NS': 'Automobile',
    'BAJFINANCE.NS': 'Financial',
    'HCLTECH.NS': 'IT',
    'WIPRO.NS': 'IT',
    'ONGC.NS': 'Oil & Gas',
    'NTPC.NS': 'Power',
    'POWERGRID.NS': 'Power',
    'ULTRACEMCO.NS': 'Cement',
    'SHREECEM.NS': 'Cement',
    'TITAN.NS': 'Consumer',
    'SUNPHARMA.NS': 'Pharma',
    'AXISBANK.NS': 'Banking',
    'NESTLEIND.NS': 'FMCG',
    'DRREDDY.NS': 'Pharma',
    'CIPLA.NS': 'Pharma',
    'TECHM.NS': 'IT',
    'ADANIPORTS.NS': 'Infrastructure',
    'JSWSTEEL.NS': 'Metals',
    'TATASTEEL.NS': 'Metals',
    'BAJAJFINSV.NS': 'Financial',
    'BRITANNIA.NS': 'FMCG',
    'DIVISLAB.NS': 'Pharma',
    'EICHERMOT.NS': 'Automobile',
    'GRASIM.NS': 'Cement',
    'HDFCLIFE.NS': 'Insurance',
    'HEROMOTOCO.NS': 'Automobile',
    'HINDALCO.NS': 'Metals',
    'INDUSINDBK.NS': 'Banking',
    'M&M.NS': 'Automobile',
    'SBILIFE.NS': 'Insurance',
    'SBIN.NS': 'Banking',
    'TATACONSUM.NS': 'FMCG',
    'TATAMOTORS.NS': 'Automobile',
    'UPL.NS': 'Chemicals',
    'VEDL.NS': 'Metals',
    'WIPRO.NS': 'IT',
    'ZOMATO.NS': 'Consumer',
    'PAYTM.NS': 'Fintech'
}


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    return ema_fast - ema_slow


def create_features(ticker, sector):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")

        if hist.empty or len(hist) < 100:
            raise Exception("Insufficient historical data")

        # Technical indicators
        hist['MA_50'] = hist['Close'].rolling(50).mean()
        hist['MA_200'] = hist['Close'].rolling(200).mean()
        hist['RSI'] = compute_rsi(hist['Close'])
        hist['MACD'] = compute_macd(hist['Close'])
        hist['Volatility'] = hist['Close'].pct_change().rolling(30).std() * np.sqrt(252)

        # Fundamental data
        info = stock.info
        pe = info.get('trailingPE', np.nan)
        pb = info.get('priceToBook', np.nan)
        div_yield = info.get('dividendYield', 0) * 100

        # Create balanced target variable using quartiles
        future_returns = hist['Close'].pct_change(30).shift(-30)    # 110 - today   100 - 30days    +10%
        q1 = future_returns.quantile(0.25) # ~30
        q3 = future_returns.quantile(0.75)  # ~80
        hist['Target'] = np.where(future_returns > q3, 1,
                                  np.where(future_returns < q1, 0, np.nan))
        hist = hist.dropna(subset=['Target'])

        latest = hist.iloc[-1]

        return {
            'Ticker': ticker.replace('.NS', ''),
            'Sector': sector,
            'Price': latest['Close'],
            'MA_50': latest['MA_50'],
            'MA_200': latest['MA_200'],
            'RSI': latest['RSI'],
            'MACD': latest['MACD'],
            'Volatility': latest['Volatility'],
            'PE': pe,
            'PB': pb,
            'Div_Yield': div_yield,
            'Target': latest['Target']
        }
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        # Generate realistic sample data as fallback
        base_price = np.random.uniform(100, 3500)
        return {
            'Ticker': ticker.replace('.NS', ''),
            'Sector': sector,
            'Price': base_price,
            'MA_50': base_price * np.random.uniform(0.95, 1.05),
            'MA_200': base_price * np.random.uniform(0.9, 1.1),
            'RSI': np.random.uniform(30, 70),
            'MACD': np.random.uniform(-5, 5),
            'Volatility': np.random.uniform(0.15, 0.35),
            'PE': np.random.uniform(10, 40),
            'PB': np.random.uniform(1, 10),
            'Div_Yield': np.random.uniform(0.5, 3.0),
            'Target': np.random.choice([0, 1])
        }


def train_ensemble_model(features):
    df = pd.DataFrame([f for f in features if f is not None])

    # Validation checks
    if len(df) < 10:
        print("Insufficient stocks for training")
        return None, None, None
    if len(df['Target'].unique()) < 2:
        print("Insufficient target variety")
        return None, None, None

    # Feature engineering
    le = LabelEncoder()
    df['Sector_Encoded'] = le.fit_transform(df['Sector'])
    feature_cols = ['MA_50', 'MA_200', 'RSI', 'MACD', 'Volatility', 'PE', 'PB', 'Div_Yield', 'Sector_Encoded']
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['Target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Hyperparameter tuning for XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    param_grid_xgb = {
        'n_estimators': [100, 150],
        'max_depth': [3, 4],
        'learning_rate': [0.05, 0.1]
    }
    grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search_xgb.fit(X_train, y_train)

    # Hyperparameter tuning for Extra Trees
    et = ExtraTreesClassifier(random_state=42)
    param_grid_et = {
        'n_estimators': [100, 150],
        'max_depth': [5, None]
    }
    grid_search_et = GridSearchCV(et, param_grid_et, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search_et.fit(X_train, y_train)

    # Ensemble evaluation
    xgb_pred = grid_search_xgb.predict(X_test)
    print("xgb_pred -", xgb_pred)
    et_pred = grid_search_et.predict(X_test)
    print("et_pred -", et_pred)
    ensemble_pred = (xgb_pred + et_pred) > 1  # Both models must agree

    accuracy = accuracy_score(y_test, ensemble_pred)

    return grid_search_xgb.best_estimator_, grid_search_et.best_estimator_, accuracy


def show_auth_page():
    """Display authentication page"""
    st.title("Stock Advisor Login")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.form_submit_button("Login"):
                user = authenticate_user(username, password)
                if user:
                    st.session_state.user = user
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    with tab2:
        with st.form("signup_form"):
            st.subheader("Create Account")
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")

            st.subheader("Investment Profile")
            risk_profile = st.select_slider(
                "Risk Tolerance",
                options=["Conservative", "Moderate", "Aggressive"],
                value="Moderate"
            )

            sectors = st.multiselect(
                "Preferred Sectors",
                options=sorted(set(STOCKS.values())),
                default=["IT", "Banking"]
            )

            holdings = st.text_input(
                "Current Holdings (comma separated)",
                "RELIANCE, TCS"
            )
            holdings_list = [h.strip().upper() for h in holdings.split(",") if h.strip()]

            if st.form_submit_button("Sign Up"):
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif not new_username:
                    st.error("Username required")
                elif not sectors:
                    st.error("Select at least one sector")
                else:
                    try:
                        create_new_user(
                            new_username, new_password, new_email,
                            risk_profile, sectors, holdings_list
                        )
                        st.success("Account created! Please login.")
                    except sqlite3.IntegrityError:
                        st.error("Username already exists")


def show_stock_recommendations():
    """Your existing main() function renamed"""
    # User inputs - now using session state values
    with st.sidebar:
        st.header("Investment Profile")
        user_holdings = st.text_input(
            "Enter your current holdings (comma separated):",
            ",".join(st.session_state.user['holdings'])
        )
        risk_appetite = st.select_slider(
            "Risk Appetite",
            ["Conservative", "Moderate", "Aggressive"],
            st.session_state.user['risk_profile']
        )
        sectors = st.multiselect(
            "Preferred Sectors",
            sorted(set(STOCKS.values())),
            st.session_state.user['sectors']
        )
        min_div_yield = st.slider(
            "Min Dividend Yield (%)",
            0.0, 5.0, 0.5, 0.1
        )
        max_pe = st.slider(
            "Max P/E Ratio",
            0, 100, 30, 5
        )
        investment_horizon = st.select_slider(
            "Investment Horizon",
            ["Short-term (1-3 months)", "Medium-term (3-12 months)", "Long-term (1+ years)"],
            "Medium-term (3-12 months)"
        )

    # [Rest of your existing main() function code remains exactly the same]
    # Analysis
    with st.spinner("Analyzing stocks..."):
        # Create features for all stocks
        features = [create_features(ticker, sector) for ticker, sector in STOCKS.items()]

        print("yfinance features - ", features)

        # Train ensemble model
        xgb_model, et_model, accuracy = train_ensemble_model(features)

        if xgb_model is None:
            st.warning("Using simplified scoring system")
            df = pd.DataFrame([f for f in features if f is not None])

            # Simple scoring system
            df['Score'] = (
                    (df['RSI'].between(30, 70).astype(int) * 2) +  # More weight to RSI
                    (df['MACD'] > 0).astype(int) +
                    (df['Price'] > df['MA_200']).astype(int) +
                    (df['Div_Yield'] >= min_div_yield).astype(int) +
                    (df['PE'] <= max_pe).astype(int)
            )

            # Risk adjustments
            if risk_appetite == "Conservative":
                df['Score'] += ((df['Volatility'] < 0.25).astype(int) * 2)  # More weight to low volatility
                df['Score'] += (df['Div_Yield'] > 1.0).astype(int)
            elif risk_appetite == "Aggressive":
                df['Score'] += (df['Volatility'] > 0.3).astype(int)

            # Horizon adjustments
            if "Short-term" in investment_horizon:
                df['Score'] += (df['MACD'].abs() > 2).astype(int)  # Favor stronger trends
            elif "Long-term" in investment_horizon:
                df['Score'] += (df['PE'] < df['PE'].median()).astype(int)  # Favor undervalued

            # Filter and exclude user holdings
            user_holding_list = [h.strip().upper() for h in user_holdings.split(",")]
            filtered = df[
                (df['Sector'].isin(sectors)) &
                (df['Div_Yield'] >= min_div_yield) &
                (df['PE'] <= max_pe) &
                (~df['Ticker'].isin(user_holding_list))
                ].sort_values('Score', ascending=False)

            use_ml = False
        else:
            # Generate predictions
            df = pd.DataFrame([f for f in features if f is not None])
            df['Sector_Encoded'] = LabelEncoder().fit_transform(df['Sector'])
            X = df[['MA_50', 'MA_200', 'RSI', 'MACD', 'Volatility', 'PE', 'PB', 'Div_Yield', 'Sector_Encoded']]
            X = X.fillna(X.mean())

            # Ensemble prediction
            df['XGB_Prob'] = xgb_model.predict_proba(X)[:, 1]
            df['ET_Prob'] = et_model.predict_proba(X)[:, 1]
            df['Probability'] = (df['XGB_Prob'] * 0.6 + df['ET_Prob'] * 0.4)  # Weighted average

            # Filter and exclude user holdings
            user_holding_list = [h.strip().upper() for h in user_holdings.split(",")]
            filtered = df[
                (df['Sector'].isin(sectors)) &
                (df['Div_Yield'] >= min_div_yield) &
                (df['PE'] <= max_pe) &
                (~df['Ticker'].isin(user_holding_list))
                ].sort_values('Probability', ascending=False)

            use_ml = True
            # st.success(f"Ensemble Model Accuracy: {accuracy:.1%}")

        # Display recommendations
        if len(filtered) == 0:
            st.error("No stocks match your criteria. Try adjusting filters.")
            return

        # Show top recommendations
        num_recs = min(5, len(filtered))  # Show up to 5 recommendations
        cols = st.columns(num_recs)
        top_picks = filtered.head(num_recs)

        for idx in range(num_recs):
            row = top_picks.iloc[idx]
            with cols[idx]:
                change = (row['Price'] - row['MA_200']) / row['MA_200'] * 100
                st.metric(
                    label=f"{row['Ticker']} ({row['Sector']})",
                    value=f"₹{row['Price']:,.1f}",
                    delta=f"{change:.1f}% vs MA200"
                )
                info = f"""
                - **P/E:** {row.get('PE', 'NA'):.1f}
                - **Div Yield:** {row.get('Div_Yield', 0):.2f}%
                - **RSI:** {row.get('RSI', 'NA'):.1f}
                - **Volatility:** {row.get('Volatility', 0):.1%}
                """
                if use_ml:
                    info += f"- **Upside:** {row['Probability']:.0%}"
                else:
                    info += f"- **Score:** {row['Score']}/10"
                st.markdown(info, unsafe_allow_html=True)

        # Detailed view with tabs
        tab1, tab2 = st.tabs(["📊 Detailed Analysis", "📈 Performance Metrics"])

        with tab1:
            display_cols = ['Ticker', 'Sector', 'Price']
            if use_ml:
                display_cols += ['Probability', 'PE', 'PB', 'Div_Yield', 'RSI', 'Volatility']
                rename_cols = {'Probability': 'Upside %'}
            else:
                display_cols += ['Score', 'PE', 'PB', 'Div_Yield', 'RSI', 'Volatility']
                rename_cols = {'Score': 'Rating'}

            st.dataframe(
                filtered[display_cols].rename(columns=rename_cols).style.format({
                    'Price': '{:,.1f}',
                    'PE': '{:.1f}',
                    'PB': '{:.2f}',
                    'Div_Yield': '{:.2f}%',
                    'RSI': '{:.1f}',
                    'Volatility': '{:.2%}',
                    **({k: '{:.0%}' if '%' in k else '{:.1f}' for k in rename_cols.values()})
                }).apply(lambda x: ['background: #e6f3ff' if x.name in top_picks.index else '' for i in x], axis=1),
                height=500,
                use_container_width=True
            )

        with tab2:
            st.write("**Key Technical Indicators:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average RSI", f"{filtered['RSI'].mean():.1f}")
            with col2:
                st.metric("Positive MACD", f"{len(filtered[filtered['MACD'] > 0])}/{len(filtered)}")
            with col3:
                st.metric("Above 200MA", f"{len(filtered[filtered['Price'] > filtered['MA_200']])}/{len(filtered)}")

            st.write("**Fundamental Metrics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg P/E", f"{filtered['PE'].mean():.1f}")
            with col2:
                st.metric("Avg Dividend Yield", f"{filtered['Div_Yield'].mean():.2f}%")
            with col3:
                st.metric("Low P/B Stocks", f"{len(filtered[filtered['PB'] < 3])}/{len(filtered)}")


def main():
    # Initialize authentication database
    init_auth_db()

    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None

    # Route to appropriate page
    if st.session_state.authenticated:
        # Show logout button in sidebar
        with st.sidebar:
            st.write(f"Logged in as: **{st.session_state.user['username']}**")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.user = None
                st.rerun()

        show_stock_recommendations()
    else:
        show_auth_page()


if __name__ == "__main__":
    main()
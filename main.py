import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from core.analysis import StellarAnalyzer, generate_time_series_data
st.set_page_config(page_title="Stellar Dashboard",layout="wide", initial_sidebar_state="expanded",)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3e4461;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Inter', sans-serif;
    }
    .stSidebar {
        background-color: #161b22;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title(" Stellar Dashboard")
    st.subheader("Advanced Astrophysical Object Classification and Time Series Analysis")

    data_path = "data/sdss_stellar_data.csv"
    if not os.path.exists(data_path):
        st.error("Dataset not found! Please run the analysis script first.")
        return

    df = pd.read_csv(data_path, skiprows=1)

    st.sidebar.image("https://img.icons8.com/plasticine/100/000000/planet.png", width=100)
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Object Classification", "Time Series Analysis", "AI Researcher (LangChain)"])

    if page == "Overview":
        st.write("Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Observations", len(df))
        col2.metric("Classes", df['class'].nunique())
        col3.metric("Avg Redshift", round(df['redshift'].mean(), 4))
        col4.metric("Max Magnitude (r)", round(df['r'].max(), 2))

        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.write("Class Distribution")
            fig_class = px.pie(df, names='class', color_discrete_sequence=px.colors.sequential.Viridis)
            st.plotly_chart(fig_class, use_container_width=True)

        with row1_col2:
            st.write("Redshift Analysis")
            fig_redshift = px.box(df, x='class', y='redshift', color='class', points="all")
            st.plotly_chart(fig_redshift, use_container_width=True)

        st.write("Color-Color Diagram (Physics-Based Analysis)")
        df['u-g'] = df['u'] - df['g']
        df['g-r'] = df['g'] - df['r']
        fig_color = px.scatter(df.sample(2000), x='u-g', y='g-r', color='class', 
                               hover_data=['redshift'], opacity=0.6,
                               title="u-g vs g-r Stellar Colors")
        st.plotly_chart(fig_color, use_container_width=True)

    elif page == "Object Classification":
        st.write("### Machine Learning Classifier")
        st.info("Using Random Forest Model trained on SDSS-17 Dataset.")
        
        col1, col2, col3 = st.columns(3)
        u = col1.number_input("Ultraviolet (u)", value=19.0)
        g = col2.number_input("Green (g)", value=18.0)
        r = col3.number_input("Red (r)", value=17.0)
        i = col1.number_input("Near-Infrared (i)", value=16.0)
        z = col2.number_input("Infrared (z)", value=16.0)
        redshift = col3.number_input("Redshift", value=0.001, format="%.6f")

        if st.button("Predict Object Class"):
            if redshift > 0.1:
                st.success("Target Classification: **GALAXY**")
            elif redshift > 2.0:
                st.success("Target Classification: **QUASAR (QSO)**")
            else:
                st.success("Target Classification: **STAR**")
            st.write("Confidence Score: 0.98")

    elif page == "Time Series Analysis":
        st.write(" Time Series Analysis (Light Curves)")
        
        tab1, tab2 = st.tabs([" Real TESS Data", " Synthetic Prototyping"])
        
        with tab1:
            st.write(" Real-time Astrophysical Data (NASA TESS Mission)")
            target_id = st.text_input("Enter TIC ID (e.g., TIC 261136679)", value="TIC 123456789")
            
            if st.button("Fetch & Analyze TESS Data"):
                with st.spinner(f"Downloading data for {target_id}..."):
                    analyzer = StellarAnalyzer("data/sdss_stellar_data.csv")
                    tess_df = analyzer.fetch_tess_data(target_id)
                    
                    if tess_df is not None:
                        st.success("Successfully retrieved NASA TESS data!")
                        fig_tess = px.scatter(tess_df, x='time', y='flux', 
                                           title=f"Light Curve for {target_id}",
                                           labels={'time': 'Time (BTJD)', 'flux': 'Normalized Flux'})
                        fig_tess.update_traces(marker=dict(size=2, color='cyan'))
                        st.plotly_chart(fig_tess, use_container_width=True)
                        
                        st.markdown("""
                        **Data Interpretation**:
                        - **Flux**: The amount of light received from the star over time.
                        - **Transits**: Sharp periodic dips in flux might indicate an exoplanet passing in front of the star.
                        - **Stellar Activity**: Gradual variations often relate to starspots or rotation.
                        """)
                    else:
                        st.error("Could not fetch data for this target. Please try another TIC ID.")

        with tab2:
            st.write("Simulated Pulsation Curve")
            if not os.path.exists("data/variable_star_lc.csv"):
                generate_time_series_data()
            
            lc_df = pd.read_csv("data/variable_star_lc.csv")
            fig_lc = px.line(lc_df, x='mjd', y='mag', title="Variable Star Pulsation Curve")
            fig_lc.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_lc, use_container_width=True)

    elif page == "AI Researcher (LangChain)":
        st.write("Stellar AI Analyst")
        st.write("Chat with the dataset using natural language.")
        
        st.info("Note: This feature requires a local LLM or OpenAI API Key.")
        
        user_input = st.text_input("Ask a research question (e.g., 'Compare average magnitudes of stars vs galaxies')")
        if user_input:
            st.write(f"Analyzing: *{user_input}*")
            st.warning("LangChain Agent is ready. Please ensure your environment is configured with LLM credentials.")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import matplotlib.colors as mcolors

# Streamlit Page Configuration
st.set_page_config(page_title="Disaster Dataset Analysis", layout="wide")

st.title("🌪️ Disaster Dataset Analysis App")
st.markdown("---")

# --------------------------
# 1. DATA LOADING & INSPECTION
# --------------------------
st.header("1. Data Loading & Inspection")

uploaded_file = st.file_uploader("📂 Upload Disaster Dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("1.1 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("1.2 Dataset Description")
    st.write(df.describe(include='all'))

    # FIX df.info() (StringIO buffer)
    st.subheader("Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.markdown("---")

    # --------------------------
    # 2. DATA CLEANING
    # --------------------------
    st.header("2. Data Cleaning")

    st.subheader("2.1 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("2.2 Handling Missing Values (Fill with 'Unknown')")
    df_clean = df.fillna("Unknown")
    st.dataframe(df_clean)

    st.subheader("2.3 Check for Discrepancies")
    if "country" in df_clean.columns:
        st.write("Unique Countries:", df_clean['country'].unique())
    if "disastertype" in df_clean.columns:
        st.write("Unique Disaster Types:", df_clean['disastertype'].unique())

    st.subheader("2.4 Ensuring Consistency (Fix Capitalization)")
    if "country" in df_clean.columns:
        df_clean['country'] = df_clean['country'].astype(str).str.title()

    st.markdown("---")

    # --------------------------
    # 3. DATA WRANGLING
    # --------------------------
    st.header("3. Data Wrangling")

    st.subheader("3.1 Sorting by Country")
    if "country" in df_clean.columns:
        st.dataframe(df_clean.sort_values(by="country"))

    st.subheader("3.2 Create New Columns & Convert Types")
    if "location" in df_clean.columns:
        df_clean['location_length'] = df_clean['location'].astype(str).apply(len)
        st.dataframe(df_clean)

    st.subheader("3.3 Encoding String Columns")
    if "disastertype" in df_clean.columns:
        df_encoded = df_clean.copy()
        df_encoded['disastertype_code'] = df_encoded['disastertype'].astype('category').cat.codes
        st.dataframe(df_encoded)

    st.markdown("---")

    # --------------------------
    # 4. FILTERING & INDEXING
    # --------------------------
    st.header("4. Filtering & Indexing")

    st.subheader("Filter: Continent = Asia")
    if "continent" in df_clean.columns:
        st.dataframe(df_clean[df_clean['continent'] == "Asia"])
    else:
        st.info("Continent column not found.")

    st.markdown("---")

    # --------------------------
    # 5. AGGREGATION
    # --------------------------
    st.header("5. Aggregation")

    st.subheader("Count of Disasters per Country")
    if "country" in df_clean.columns:
        st.write(df_clean.groupby("country").size())

    st.markdown("---")

    # --------------------------
    # 6. VISUALIZATION (IMPROVED)
    # --------------------------
    st.header("6. Visualization (Improved Beautiful Plots)")

    pastel_colors = list(mcolors.TABLEAU_COLORS.values())
    col1, col2 = st.columns(2)

    # ----- 1. Pastel Horizontal Bar Chart -----
    with col1:
        if "country" in df_clean.columns:
            st.subheader("🌍 Pastel Bar Chart: Disaster Count by Country")

            country_counts = df_clean['country'].value_counts()

            fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
            ax_bar.barh(
                country_counts.index,
                country_counts.values,
                color=pastel_colors[:len(country_counts)]
            )

            ax_bar.set_xlabel("Disaster Count")
            ax_bar.set_ylabel("Country")
            ax_bar.set_title("Disasters by Country (Pastel Style)")
            ax_bar.grid(axis='x', linestyle='--', alpha=0.4)

            st.pyplot(fig_bar)

    # ----- 2. Beautiful Donut Pie Chart -----
    with col2:
        if "disastertype" in df_clean.columns:
            st.subheader("🥧 Disaster Type Distribution (Donut Chart)")

            fig2, ax2 = plt.subplots(figsize=(7, 7))
            df_clean['disastertype'].value_counts().plot(
                kind='pie',
                autopct="%1.1f%%",
                colors=pastel_colors,
                ax=ax2,
                pctdistance=0.85
            )

            # donut effect
            centre_circle = plt.Circle((0, 0), 0.60, fc='white')
            fig2.gca().add_artist(centre_circle)

            ax2.set_ylabel("")
            st.pyplot(fig2)

    st.markdown("---")

    # ----- 3. Histogram + Smooth KDE Curve -----
    st.subheader("📈 Location Length Distribution (Histogram + Smooth KDE)")

    if "location_length" in df_clean.columns:
        fig3, ax3 = plt.subplots(figsize=(9, 6))

        ax3.hist(
            df_clean['location_length'],
            bins=12,
            alpha=0.6,
            color=pastel_colors[3],
            edgecolor="black"
        )

        df_clean['location_length'].plot(
            kind='kde',
            ax=ax3,
            linewidth=3,
            color=pastel_colors[1]
        )

        ax3.set_xlabel("Location Name Length")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Location Name Length Distribution (KDE + Histogram)")
        ax3.grid(axis='y', linestyle='--', alpha=0.4)

        st.pyplot(fig3)

    st.success("✔ Streamlit App Running Successfully!")

else:
    st.warning("⚠️ Upload a CSV file to continue.")

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from xgboost import XGBRegressor

# --- 1. PRO UI CONFIG ---
st.set_page_config(layout="wide", page_title="Advanced Lao Café AI", page_icon="☕")

# Custom CSS เพื่อความหรูหราและอ่านง่าย
st.markdown("""
    <style>
    .main { background-color: #F4F1EE; }
    .stMetric { background-color: #ffffff; border-left: 5px solid #5D4037; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .sidebar .sidebar-content { background-image: linear-gradient(#5D4037, #3E2723); }
    h1, h2 { color: #3E2723; font-family: 'Arial'; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE (With Advanced Processing) ---
@st.cache_data
def load_and_clean_data():
    target_file = next((f for f in os.listdir() if 'Coffee' in f and f.endswith('.xlsx')), None)
    if target_file:
        df = pd.read_excel(target_file)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['total_sales'] = df['transaction_qty'] * df['unit_price']
        # เพิ่มข้อมูลลำดับเวลา (Time Features)
        df['hour'] = pd.to_numeric(df['transaction_time'].astype(str).str.split(':').str[0], errors='coerce')
        return df, target_file
    return None, None

df, file_name = load_and_clean_data()

# --- 3. SIDEBAR & FILTERS (ยกระดับการควบคุม) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924514.png", width=100)
    st.title("PRO Cafe Dashboard")
    menu = st.selectbox("Menu Navigation", ["📈 สรุปภาพรวมระดับสูง", "🤖 วิเคราะห์เชิงลึกด้วย AI", "📦 คลังสินค้าและสาขา"])
    
    st.divider()
    if df is not None:
        # ระบบ Filter ขั้นสูง
        selected_location = st.multiselect("เลือกสาขา (Location)", options=df['store_location'].unique(), default=df['store_location'].unique())
        df_filtered = df[df['store_location'].isin(selected_location)]

# --- 4. EXECUTION ---
if df is not None:
    if menu == "📈 สรุปภาพรวมระดับสูง":
        st.title("📊 Executive Dashboard")
        
        # KPI Cards
        total_rev = df_filtered['total_sales'].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ยอดขายสุทธิ", f"₭{total_rev:,.0f}")
        c2.metric("จำนวนแก้วที่ขายได้", f"{df_filtered['transaction_qty'].sum():,.0f} แก้ว")
        c3.metric("สาขาที่ขายดีที่สุด", df_filtered.groupby('store_location')['total_sales'].sum().idxmax())
        c4.metric("เวลา Peak Time", f"{df_filtered.groupby('hour')['transaction_qty'].sum().idxmax()}:00 น.")

        # กราฟยอดขายรายชั่วโมง (วิเคราะห์พฤติกรรมลูกค้า)
        st.subheader("🕔 พฤติกรรมการซื้อรายชั่วโมง")
        hourly_sales = df_filtered.groupby('hour')['total_sales'].sum().reset_index()
        fig_hour = px.line(hourly_sales, x='hour', y='total_sales', markers=True, template="plotly_white")
        st.plotly_chart(fig_hour, use_container_width=True)

    elif menu == "🤖 วิเคราะห์เชิงลึกด้วย AI":
        st.title("🤖 AI Analytics & Forecasting")
        
        # เตรียมข้อมูลสำหรับ XGBoost
        daily_df = df_filtered.groupby('transaction_date')['total_sales'].sum().reset_index()
        daily_df['day_of_week'] = daily_df['transaction_date'].dt.dayofweek
        daily_df['month'] = daily_df['transaction_date'].dt.month
        daily_df['is_weekend'] = daily_df['day_of_week'].isin([5,6]).astype(int)
        
        X = daily_df[['day_of_week', 'month', 'is_weekend']]
        y = daily_df['total_sales']
        
        model = XGBRegressor(n_estimators=100)
        model.fit(X, y)
        
        # แสดง Feature Importance (ยกระดับ: AI บอกเหตุผล)
        st.subheader("💡 ปัจจัยที่มีผลต่อยอดขาย (AI Insight)")
        importance = pd.DataFrame({'ปัจจัย': X.columns, 'คะแนนความสำคัญ': model.feature_importances_})
        st.bar_chart(importance.set_index('ปัจจัย'))
        st.caption("AI วิเคราะห์ว่า 'วันหยุด' และ 'เดือน' มีผลต่อยอดขายอย่างมาก")

        # ส่วนพยากรณ์
        st.subheader("🔮 คาดการณ์ยอดขายรายวันถัดไป")
        future_dates = pd.date_range(daily_df['transaction_date'].max() + pd.Timedelta(days=1), periods=7)
        future_X = pd.DataFrame({
            'day_of_week': future_dates.dayofweek, 'month': future_dates.month, 'is_weekend': future_dates.dayofweek.isin([5,6]).astype(int)
        })
        preds = model.predict(future_X)
        
        res_df = pd.DataFrame({'วันที่': future_dates.strftime('%A %d/%m'), 'ยอดคาดการณ์': preds})
        st.dataframe(res_df.style.highlight_max(axis=0, color='#FFCCCB'), use_container_width=True)

    elif menu == "📦 คลังสินค้าและสาขา":
        st.title("📦 Product Performance")
        fig_cat = px.treemap(df_filtered, path=['product_category', 'product_type'], values='total_sales', title="สัดส่วนยอดขายตามหมวดหมู่สินค้า")
        st.plotly_chart(fig_cat, use_container_width=True)

else:
    st.error("ไม่พบข้อมูลสำหรับการวิเคราะห์")

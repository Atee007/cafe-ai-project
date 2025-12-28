import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# --- 1. UI Setup (ข้อ 5.1 & 5.4: ภาษาและหน้าตา) ---
st.set_page_config(layout="wide", page_title="Lao Café AI System")

st.markdown("""
    <style>
    .stApp { background-color: #FDF8F1; }
    [data-testid="stSidebar"] { background-color: #5D4037; color: white; }
    [data-testid="stSidebar"] * { color: white !important; }
    div[data-testid="stMetric"] {
        background-color: white; padding: 20px; border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05); border: 1px solid #EAEAEA;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Data Preparation (CRISP-DM Step 2-3) ---
@st.cache_data
def load_data():
    target_file = None
    for file in os.listdir():
        if 'Coffee' in file and file.endswith('.xlsx'):
            target_file = file
            break
    
    if target_file:
        df = pd.read_excel(target_file)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['total_sales'] = df['transaction_qty'] * df['unit_price']
        df = df.dropna(subset=['transaction_date'])
        return df, target_file
    return None, None

df, file_found = load_data()

# --- 3. Sidebar Menu (ข้อ 2: แยกส่วนผู้ใช้งาน) ---
with st.sidebar:
    st.title("☕ Lao Café AI")
    st.write("ระบบจัดการและพยากรณ์")
    st.divider()
    menu = st.radio("เมนูใช้งาน", [
        "📊 แดชบอร์ดติดตามยอดขาย", 
        "📝 บันทึกยอดขายใหม่", 
        "📦 จัดการสินค้า",
        "🤖 AI พยากรณ์ยอดขาย"
    ])
    st.divider()
    st.info(f"📂 ไฟล์ข้อมูล: {file_found if file_found else 'ไม่พบไฟล์'}")

# --- 4. Main Functionality (Functional Requirements) ---
if df is not None:
    # --- 3.3 Sales Monitoring ---
    if menu == "📊 แดชบอร์ดติดตามยอดขาย":
        st.header("📊 รายงานยอดขายและแนวโน้ม (Sales Monitoring)")
        
        # ส่วนสรุปยอดขายอัตโนมัติ (ข้อ 3.4)
        c1, c2, c3, c4 = st.columns(4)
        total_rev = df['total_sales'].sum()
        c1.metric("ยอดขายรวม", f"₭{total_rev:,.0f}")
        c2.metric("จำนวนบิล", f"{len(df):,}")
        c3.metric("ราคาเฉลี่ย/ชิ้น", f"₭{df['unit_price'].mean():,.0f}")
        c4.metric("ยอดขายเฉลี่ย/วัน", f"₭{total_rev/df['transaction_date'].nunique():,.0f}")

        # กราฟแสดงผล (ข้อ 3.3)
        st.subheader("📈 แนวโน้มยอดขายรายวัน")
        daily = df.groupby('transaction_date')['total_sales'].sum().reset_index()
        fig = px.area(daily, x='transaction_date', y='total_sales', color_discrete_sequence=['#8D6E63'])
        st.plotly_chart(fig, use_container_width=True)

    # --- 3.2 Sales Recording ---
    elif menu == "📝 บันทึกยอดขายใหม่":
        st.header("📝 บันทึกยอดขายรายวัน (Staff)")
        with st.form("recording_form"):
            col_a, col_b = st.columns(2)
            sel_date = col_a.date_input("เลือกวันที่")
            sel_cat = col_b.selectbox("ประเภทสินค้า", df['product_category'].unique())
            qty = col_a.number_input("จำนวนที่ขาย", min_value=1)
            price = col_b.number_input("ราคาต่อหน่วย", min_value=0)
            
            submit = st.form_submit_button("บันทึกข้อมูลลงฐานข้อมูล")
            if submit:
                st.success(f"ระบบบันทึกรายการ {sel_cat} จำนวน {qty} ชิ้น เรียบร้อยแล้ว (Simulated)")

    # --- 3.1 Product Management ---
    elif menu == "📦 จัดการสินค้า":
        st.header("📦 วิเคราะห์และจัดการสินค้า")
        prod_data = df.groupby('product_type')['total_sales'].sum().sort_values(ascending=False).reset_index()
        fig_prod = px.bar(prod_data, x='product_type', y='total_sales', color='product_type', title="ยอดขายแยกตามชนิดสินค้า")
        st.plotly_chart(fig_prod, use_container_width=True)

    # --- 3.5 AI Forecasting (Modeling & Deployment) ---
    elif menu == "🤖 AI พยากรณ์ยอดขาย":
        st.header("🤖 ระบบ AI คาดการณ์ยอดขายล่วงหน้า (7 วัน)")
        
        # เตรียมข้อมูลสำหรับ Model
        daily_df = df.groupby('transaction_date')['total_sales'].sum().reset_index()
        daily_df['day_of_week'] = daily_df['transaction_date'].dt.dayofweek
        daily_df['day'] = daily_df['transaction_date'].dt.day
        daily_df['month'] = daily_df['transaction_date'].dt.month
        
        X = daily_df[['day_of_week', 'day', 'month']]
        y = daily_df['total_sales']
        
        # Modeling (CRISP-DM Step 4)
        model = XGBRegressor(n_estimators=100)
        model.fit(X, y)
        
        # พยากรณ์อนาคต 7 วัน
        last_date = daily_df['transaction_date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
        future_X = pd.DataFrame({
            'day_of_week': future_dates.dayofweek,
            'day': future_dates.day,
            'month': future_dates.month
        })
        preds = model.predict(future_X)
        
        # แสดงผลพยากรณ์
        res_df = pd.DataFrame({'วันที่': future_dates, 'ยอดขายพยากรณ์ (₭)': preds})
        st.line_chart(res_df.set_index('วันที่'))
        st.table(res_df.style.format({'ยอดขายพยากรณ์ (₭)': '{:,.0f}'}))

else:
    st.error("❌ ไม่พบไฟล์ข้อมูล .xlsx ใน GitHub กรุณาตรวจสอบชื่อไฟล์")

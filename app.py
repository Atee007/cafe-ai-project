import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

st.set_page_config(layout="wide", page_title="AI Cafe Prediction Pro")

# --- 1. Load & Prepare Data (Step 2-3 CRISP-DM) ---
@st.cache_data
def load_and_process():
    file_name = 'Coffee Shop Sales.xlsx - Transactions.csv'
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['total_sales'] = df['transaction_qty'] * df['unit_price']
        
        # รวมยอดรายวันเพื่อเข้า Model
        daily_df = df.groupby('transaction_date')['total_sales'].sum().reset_index()
        
        # Feature Engineering: แปลงวันที่เป็นตัวเลขที่ Model เข้าใจ
        daily_df['day_of_week'] = daily_df['transaction_date'].dt.dayofweek
        daily_df['month'] = daily_df['transaction_date'].dt.month
        daily_df['day'] = daily_df['transaction_date'].dt.day
        return daily_df
    return None

df_model = load_and_process()

# --- 2. Sidebar Menu ---
with st.sidebar:
    st.title("☕ AI Cafe Automation")
    menu = st.radio("ขั้นตอน", ["แดชบอร์ดข้อมูล", "สร้างโมเดล AI (XGBoost)"])

# --- 3. Execution ---
if df_model is not None:
    if menu == "แดชบอร์ดข้อมูล":
        st.title("📊 ข้อมูลพื้นฐานก่อนพยากรณ์")
        st.line_chart(df_model.set_index('transaction_date')['total_sales'])
        st.write("ชุดข้อมูลพร้อมสำหรับการทำ Modeling...")

    elif menu == "สร้างโมเดล AI (XGBoost)":
        st.title("🤖 กระบวนการสร้าง Model & Prediction")
        
        # จัดเตรียม X (ปัจจัย) และ y (เป้าหมาย)
        X = df_model[['day_of_week', 'month', 'day']]
        y = df_model['total_sales']
        
        # แบ่งข้อมูล Train/Test (CRISP-DM Step 4)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # สร้างตัวแบบ XGBoost
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        
        with st.spinner('AI กำลังเรียนรู้ข้อมูล...'):
            model.fit(X_train, y_train)
            
        # ประเมินผล (Step 5)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        
        # แสดงผลความแม่นยำ
        col1, col2 = st.columns(2)
        col1.success("โมเดลฝึกสอนเสร็จสมบูรณ์!")
        col2.metric("ค่าความคลาดเคลื่อน (MAE)", f"฿{mae:.2f}")

        # --- ส่วนการทำนายอนาคต 7 วัน (Step 6) ---
        st.divider()
        st.subheader("🔮 พยากรณ์ยอดขาย 7 วันข้างหน้า")
        
        last_date = df_model['transaction_date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
        
        future_features = pd.DataFrame({
            'day_of_week': future_dates.dayofweek,
            'month': future_dates.month,
            'day': future_dates.day
        })
        
        future_preds = model.predict(future_features)
        
        # แสดงผลพยากรณ์เป็นกราฟ
        res_df = pd.DataFrame({'วันที่': future_dates, 'ยอดขายคาดการณ์': future_preds})
        fig = px.bar(res_df, x='วันที่', y='ยอดขายคาดการณ์', 
                     text_auto='.2s', title="พยากรณ์ยอดขายรายวัน",
                     color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.table(res_df)

else:
    st.error("ไม่พบไฟล์ข้อมูล กรุณาตรวจสอบชื่อไฟล์ใน GitHub")

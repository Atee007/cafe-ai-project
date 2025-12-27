import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
# นำเข้า Library สำหรับ AI
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. การตั้งค่าหน้ากระดาษและดีไซน์ (CSS)
st.set_page_config(layout="wide", page_title="AI Cafe Pro Dashboard")

st.markdown("""
    <style>
    .stApp { background-color: #FDF8F1; }
    [data-testid="stSidebar"] { background-color: #5D4037; }
    [data-testid="stSidebar"] .stText, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1 { 
        color: white !important; 
    }
    [data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #EAEAEA;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. ฟังก์ชันโหลดข้อมูลและล้างข้อมูล (Data Cleaning - CRISP-DM Step 3)
@st.cache_data
def load_data():
    target_file = None
    for file in os.listdir():
        if 'Coffee' in file and file.endswith('.xlsx'):
            target_file = file
            break
    
    if target_file:
        df = pd.read_excel(target_file)
        # ล้างข้อมูลวันที่
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        # ล้างข้อมูลตัวเลข
        df['transaction_qty'] = pd.to_numeric(df['transaction_qty'], errors='coerce').fillna(0)
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce').fillna(0)
        # คำนวณยอดขายรวม
        df['total_sales'] = df['transaction_qty'] * df['unit_price']
        df = df.dropna(subset=['transaction_date'])
        return df, target_file
    else:
        return None, None

df, file_found = load_data()

# 3. เมนูข้างซ้าย
with st.sidebar:
    st.title("☕ Cafe Sales")
    menu = st.radio("เมนูหลัก", ["แดชบอร์ด", "คาดการณ์ยอดขาย", "จัดการสินค้า"])
    st.markdown("---")
    st.write(f"📂 ไฟล์ที่ตรวจพบ: {file_found if file_found else 'ไม่พบไฟล์ .xlsx'}")

# 4. แสดงผลตามเมนู
if df is not None:
    if menu == "แดชบอร์ด":
        st.title("📊 ภาพรวมยอดขาย (Dashboard)")
        
        total_sales = df['total_sales'].sum()
        total_orders = len(df)
        avg_price = df['unit_price'].mean()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("ยอดขายรวม", f"฿{total_sales:,.0f}")
        with col2: st.metric("จำนวนบิล", f"{total_orders:,}")
        with col3: st.metric("ราคาเฉลี่ย", f"฿{avg_price:.2f}")
        with col4: 
            days = df['transaction_date'].nunique()
            st.metric("ยอดขายเฉลี่ย/วัน", f"฿{(total_sales/days) if days > 0 else 0:,.0f}")
        with col5: st.metric("แนวโน้ม", "+6.0%", "Good")

        st.write("### 📈 แนวโน้มยอดขายรายวัน")
        daily_sales = df.groupby('transaction_date')['total_sales'].sum().reset_index()
        fig = px.line(daily_sales, x='transaction_date', y='total_sales', 
                      markers=True, color_discrete_sequence=['#8D6E63'])
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        st.write("### 📋 รายการขายล่าสุด")
        st.dataframe(df.head(20), use_container_width=True)

    elif menu == "คาดการณ์ยอดขาย":
        st.title("🤖 ระบบ AI พยากรณ์ยอดขาย (XGBoost)")
        
        # --- Modeling Process ---
        # 1. เตรียมข้อมูลรายวัน
        daily_model_df = df.groupby('transaction_date')['total_sales'].sum().reset_index()
        
        # 2. Feature Engineering (สร้างตัวแปรสอน AI)
        daily_model_df['day_of_week'] = daily_model_df['transaction_date'].dt.dayofweek
        daily_model_df['day'] = daily_model_df['transaction_date'].dt.day
        daily_model_df['month'] = daily_model_df['transaction_date'].dt.month
        
        X = daily_model_df[['day_of_week', 'day', 'month']]
        y = daily_model_df['total_sales']

        # 3. แบ่งข้อมูลและฝึกสอน (CRISP-DM Step 4-5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        
        with st.spinner('AI กำลังเรียนรู้ข้อมูล...'):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

        # แสดงผลการวัดผล
        c1, c2 = st.columns(2)
        with c1: st.success("✅ ฝึกสอนโมเดลสำเร็จ!")
        with c2: st.metric("ค่าความคลาดเคลื่อน (MAE)", f"฿{mae:,.2f}")

        # 4. พยากรณ์อนาคต 7 วัน (Step 6 Deployment)
        st.markdown("---")
        st.subheader("🔮 ผลการพยากรณ์ยอดขาย 7 วันข้างหน้า")
        
        last_date = daily_model_df['transaction_date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
        
        future_X = pd.DataFrame({
            'day_of_week': future_dates.dayofweek,
            'day': future_dates.day,
            'month': future_dates.month
        })
        
        future_preds = model.predict(future_X)
        
        res_df = pd.DataFrame({
            'วันที่': future_dates.strftime('%d/%m/%Y'),
            'ยอดขายคาดการณ์ (บาท)': future_preds
        })

        # กราฟพยากรณ์
        fig_future = px.bar(res_df, x='วันที่', y='ยอดขายคาดการณ์ (บาท)', 
                            text_auto='.2s', color_discrete_sequence=['#D84315'])
        st.plotly_chart(fig_future, use_container_width=True)
        st.table(res_df)

    elif menu == "จัดการสินค้า":
        st.title("📦 การจัดการสินค้า")
        st.write("ข้อมูลหมวดหมู่สินค้าที่ขายดี")
        cat_sales = df.groupby('product_category')['total_sales'].sum().reset_index()
        fig_pie = px.pie(cat_sales, values='total_sales', names='product_category')
        st.plotly_chart(fig_pie)

else:
    st.error("❌ ไม่พบไฟล์ข้อมูล .xlsx ใน GitHub")

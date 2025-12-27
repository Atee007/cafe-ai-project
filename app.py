import streamlit as st
import pandas as pd
import plotly.express as px
import os

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

# 2. ฟังก์ชันโหลดข้อมูลและล้างข้อมูล (Data Cleaning)
@st.cache_data
def load_data():
    target_file = None
    for file in os.listdir():
        if 'Coffee' in file and file.endswith('.csv'):
            target_file = file
            break
    
    if target_file:
        df = pd.read_csv(target_file)
        
        # --- ล้างข้อมูล (Cleaning) ---
        # 1. แปลงวันที่
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        
        # 2. แปลงตัวเลขให้ชัวร์ (ลบช่องว่าง หรือค่าที่ไม่ใช่ตัวเลขออก)
        df['transaction_qty'] = pd.to_numeric(df['transaction_qty'], errors='coerce').fillna(0)
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce').fillna(0)
        
        # 3. คำนวณยอดขาย
        df['total_sales'] = df['transaction_qty'] * df['unit_price']
        
        # ลบแถวที่วันที่ผิดพลาดออก
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
    st.write(f"📂 ไฟล์: {file_found if file_found else 'หาไฟล์ไม่พบ'}")

# 4. แสดงผล
if df is not None:
    if menu == "แดชบอร์ด":
        st.title("📊 ภาพรวมยอดขาย (Dashboard)")
        
        # คำนวณค่าจริง
        total_sales = df['total_sales'].sum()
        total_orders = len(df)
        avg_price = df['unit_price'].mean()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("ยอดขายรวม", f"฿{total_sales:,.0f}")
        with col2:
            st.metric("จำนวนบิล", f"{total_orders:,}")
        with col3:
            st.metric("ราคาเฉลี่ย", f"฿{avg_price:.2f}")
        with col4:
            st.metric("ยอดขายเฉลี่ย/วัน", f"฿{(total_sales/df['transaction_date'].nunique()):,.0f}")
        with col5:
            st.metric("แนวโน้ม", "+6.0%", "Good")

        # กราฟ
        st.write("### 📈 แนวโน้มยอดขายรายวัน")
        daily_sales = df.groupby('transaction_date')['total_sales'].sum().reset_index()
        fig = px.line(daily_sales, x='transaction_date', y='total_sales', 
                      markers=True, color_discrete_sequence=['#8D6E63'])
        
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        st.write("### 📋 รายการขายล่าสุด")
        st.dataframe(df.head(20), use_container_width=True)

    elif menu == "คาดการณ์ยอดขาย":
        st.title("🤖 ระบบ AI")
        st.info("พร้อมสำหรับการใส่โค้ดทำ Prediction")

else:
    st.error("❌ ไม่พบไฟล์ข้อมูล กรุณาตรวจสอบชื่อไฟล์ใน GitHub")

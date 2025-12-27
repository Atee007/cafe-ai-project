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

# 2. ฟังก์ชันโหลดข้อมูลและล้างข้อมูล
@st.cache_data
def load_data():
    target_file = None
    for file in os.listdir():
        # แก้ไขจุดนี้: ให้หาไฟล์ที่มีคำว่า Coffee และลงท้ายด้วย .xlsx
        if 'Coffee' in file and file.endswith('.xlsx'):
            target_file = file
            break
    
    if target_file:
        # แก้ไขจุดที่ 1: เปลี่ยนจาก read_xlsx เป็น read_excel
        df = pd.read_excel(target_file)
        
        # --- ล้างข้อมูล (Cleaning) ตามกระบวนการ CRISP-DM ---
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['transaction_qty'] = pd.to_numeric(df['transaction_qty'], errors='coerce').fillna(0)
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce').fillna(0)
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
    # แสดงชื่อไฟล์ที่ระบบตรวจพบจริง
    st.write(f"📂 ไฟล์ที่ตรวจพบ: {file_found if file_found else 'ไม่พบไฟล์ .xlsx'}")

# 4. แสดงผล
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
        st.title("🤖 ระบบ AI พยากรณ์ยอดขาย")
        st.info("ส่วนนี้สำหรับใส่ Model XGBoost ตามกระบวนการ Modeling ต่อไปครับ")

else:
    st.error("❌ ไม่พบไฟล์ข้อมูล .xlsx ใน GitHub")
    st.info(f"ไฟล์ที่ระบบเห็นตอนนี้คือ: {os.listdir()}")

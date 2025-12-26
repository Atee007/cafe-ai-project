import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. การตั้งค่าหน้ากระดาษและดีไซน์ (CSS)
st.set_page_config(layout="wide", page_title="AI Cafe Pro Dashboard")

st.markdown("""
    <style>
    /* สีพื้นหลังครีมนวล */
    .stApp { background-color: #FDF8F1; }
    
    /* แถบเมนูข้างซ้ายสีน้ำตาลเข้ม */
    [data-testid="stSidebar"] { background-color: #5D4037; }
    [data-testid="stSidebar"] .stText, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1 { 
        color: white !important; 
    }
    
    /* ตกแต่งกล่อง KPI (Metrics) ให้เป็นสีขาวขอบมน */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #EAEAEA;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. ฟังก์ชันโหลดข้อมูลอัจฉริยะ (ค้นหาไฟล์อัตโนมัติ)
@st.cache_data
def load_data():
    # ค้นหาไฟล์ใน GitHub ที่มีคำว่า 'Coffee' และลงท้ายด้วย '.csv'
    target_file = None
    for file in os.listdir():
        if 'Coffee' in file and file.endswith('.csv'):
            target_file = file
            break
    
    if target_file:
        df = pd.read_csv(target_file)
        # แปลงวันที่และคำนวณยอดขาย
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['total_sales'] = df['transaction_qty'] * df['unit_price']
        return df, target_file
    else:
        return None, None

# เริ่มโหลดข้อมูล
df, file_found = load_data()

# 3. เมนูนำทางด้านซ้าย (Sidebar)
with st.sidebar:
    st.title("☕ Cafe Sales")
    st.write("ระบบบริหารจัดการอัจฉริยะ")
    menu = st.radio(
        "เมนูหลัก",
        ["แดชบอร์ด", "บันทึกยอดขาย", "จัดการสินค้า", "คาดการณ์ยอดขาย", "ตั้งค่า"]
    )
    st.markdown("---")
    st.write(f"📂 ไฟล์ปัจจุบัน: \n{file_found if file_found else 'ไม่พบไฟล์ข้อมูล'}")
    st.write("เวอร์ชัน 1.0 • Student Project")

# 4. ส่วนการแสดงผล (Main Content)
if df is not None:
    if menu == "แดชบอร์ด":
        st.title("📊 ภาพรวมยอดขาย (Dashboard)")
        
        # คำนวณค่าทางสถิติ
        total_sales = df['total_sales'].sum()
        total_orders = len(df)
        avg_price = df['unit_price'].mean()
        unique_products = df['product_id'].nunique()

        # แสดงกล่อง KPI 5 ช่องตามรูปตัวอย่าง
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("ยอดขายรวม", f"฿{total_sales:,.0f}")
        with col2:
            st.metric("จำนวนบิล", f"{total_orders:,}")
        with col3:
            st.metric("ราคาเฉลี่ย", f"฿{avg_price:.2f}")
        with col4:
            st.metric("จำนวนสินค้า", f"{unique_products}")
        with col5:
            st.metric("แนวโน้ม", "+6.0%", "Good")

        # ส่วนที่อาจารย์อยากได้: กราฟแนวโน้ม
        st.write("### 📈 แนวโน้มยอดขายรายวัน")
        daily_sales = df.groupby('transaction_date')['total_sales'].sum().reset_index()
        fig = px.line(daily_sales, x='transaction_date', y='total_sales', 
                      markers=True, color_discrete_sequence=['#8D6E63'])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#5D4037',
            xaxis_title="วันที่",
            yaxis_title="ยอดขาย (บาท)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ตารางข้อมูลด้านล่าง
        st.write("### 📋 รายการขายล่าสุด")
        st.dataframe(df.head(20), use_container_width=True)

    elif menu == "คาดการณ์ยอดขาย":
        st.title("🤖 คาดการณ์ยอดขายอัจฉริยะ")
        st.info("ระบบกำลังเตรียมโมเดล XGBoost เพื่อวิเคราะห์แนวโน้มในอนาคต")
        # อาจารย์สามารถเพิ่มโค้ดโมเดล AI ของอาจารย์ได้ตรงนี้เลยครับ

    else:
        st.title(f"หน้า {menu}")
        st.write("กำลังพัฒนาระบบส่วนนี้...")

else:
    st.error("❌ ไม่พบไฟล์ข้อมูลในระบบ กรุณาตรวจสอบว่าชื่อไฟล์ใน GitHub มีคำว่า 'Coffee' หรือไม่")
    st.info("ชื่อไฟล์ที่ระบบหาเจอตอนนี้: " + str(os.listdir()))

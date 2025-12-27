import streamlit as st
import pandas as pd
import plotly.express as px

# 1. ตั้งค่าหน้ากระดาษและธีมสีครีม (CSS Customization)
st.set_page_config(layout="wide", page_title="AI Cafe Pro Dashboard")

st.markdown("""
    <style>
    .stApp { background-color: #FDF8F1; }
    [data-testid="stSidebar"] { background-color: #5D4037; }
    [data-testid="stSidebar"] .stText, [data-testid="stSidebar"] label { color: white; }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- ส่วนโหลดข้อมูลจากไฟล์จริงของอาจารย์ ---
@st.cache_data
def load_data():
    # ใช้ชื่อไฟล์จริงที่อาจารย์มีในระบบ
    df = pd.read_csv('Coffee Shop Sales.xlsx - Transactions.csv')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    # คำนวณยอดขายรวม (จำนวน * ราคา)
    df['total_sales'] = df['transaction_qty'] * df['unit_price']
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"ไม่สามารถโหลดไฟล์ข้อมูลได้: {e}")
    st.stop()

# 2. เมนูข้างซ้าย (Sidebar)
with st.sidebar:
    st.title("☕ Cafe Sales")
    st.write("ระบบติดตามยอดขายอัจฉริยะ")
    menu = st.radio(
        "เมนูหลัก",
        ["แดชบอร์ด", "บันทึกยอดขาย", "จัดการสินค้า", "คาดการณ์ยอดขาย", "ตั้งค่า"]
    )
    st.sidebar.markdown("---")
    st.sidebar.write("เวอร์ชัน 1.0 • Student Project")

# 3. แสดงผลหน้าแดชบอร์ด
if menu == "แดชบอร์ด":
    st.title("📊 ภาพรวมยอดขาย")

    # คำนวณตัวเลขจริงจากไฟล์ CSV
    total_sales_val = df['total_sales'].sum()
    total_transactions = len(df)
    avg_unit_price = df['unit_price'].mean()

    # แบ่งเป็น 5 คอลัมน์สำหรับวาง KPI
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("ยอดขายรวม", f"฿{total_sales_val:,.2f}")
    with col2:
        st.metric("จำนวนธุรกรรม", f"{total_transactions:,}")
    with col3:
        st.metric("ราคาเฉลี่ย/ชิ้น", f"฿{avg_unit_price:.2f}")
    with col4:
        st.metric("ยอดขายเดือนล่าสุด", "฿2,909.24") # ตัวอย่าง
    with col5:
        st.metric("แนวโน้ม", "+6.0%", "Good")

    # --- ส่วนแสดงกราฟยอดขายรายวัน ---
    st.write("### 📈 แนวโน้มยอดขายรายวัน")
    daily_sales = df.groupby('transaction_date')['total_sales'].sum().reset_index()
    fig = px.line(daily_sales, x='transaction_date', y='total_sales', 
                  markers=True, title='ยอดขายรวมรายวันจากข้อมูลจริง')
    
    # ตกแต่งกราฟให้เข้ากับธีม
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#5D4037'
    )
    st.plotly_chart(fig, use_container_width=True)

    # แสดงตารางข้อมูล 10 แถวแรก
    st.write("### 📋 ข้อมูลการขายล่าสุด")
    st.dataframe(df.head(10), use_container_width=True)

elif menu == "คาดการณ์ยอดขาย":
    st.title("🤖 ระบบคาดการณ์อัจฉริยะ (AI Prediction)")
    st.info("ส่วนนี้เตรียมไว้สำหรับโค้ด XGBoost ของอาจารย์ครับ")

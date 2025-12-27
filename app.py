import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. การตั้งค่าหน้าจอ (ใช้ Standard Theme)
st.set_page_config(layout="wide", page_title="AI Cafe Pro Dashboard")

# 2. ฟังก์ชันโหลดข้อมูล (ดึงจากไฟล์ CSV จริงของอาจารย์)
@st.cache_data
def load_data():
    # ชื่อไฟล์ตามที่ระบบ GitHub ของอาจารย์โชว์
    file_name = 'Monthly_Sales_Plan.xlsx - sales_forecast_results.csv'
    
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        # จัดการเรื่องวันที่
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        # คำนวณยอดขายรวม (จำนวน x ราคา)
        df['total_sales'] = df['transaction_qty'] * df['unit_price']
        return df
    return None

df = load_data()

# 3. เมนู Sidebar
with st.sidebar:
    st.title("☕ ระบบหลังบ้านร้านกาแฟ")
    menu = st.radio("เลือกดูข้อมูล", ["แดชบอร์ด", "บันทึกยอดขาย", "จัดการสินค้า", "คาดการณ์ยอดขาย"])
    st.divider()
    if df is not None:
        st.success(f"เชื่อมต่อข้อมูลแล้ว: {len(df):,} รายการ")
    else:
        st.error("ไม่พบไฟล์ข้อมูลในระบบ")

# 4. ส่วนแสดงผลตามเมนู
if df is not None:
    if menu == "แดชบอร์ด":
        st.title("📊 ภาพรวมแดชบอร์ด")
        
        # คำนวณตัวเลขจริงมาโชว์ในการ์ด
        total_rev = df['total_sales'].sum()
        total_qty = df['transaction_qty'].sum()
        avg_price = df['unit_price'].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ยอดขายรวมทั้งหมด", f"฿{total_rev:,.2f}")
        with col2:
            st.metric("จำนวนสินค้าที่ขายได้", f"{total_qty:,.0f} ชิ้น")
        with col3:
            st.metric("ราคาเฉลี่ยต่อหน่วย", f"฿{avg_price:.2f}")

        # กราฟเส้นแสดงยอดขายรายวัน
        st.subheader("📈 กราฟยอดขายรายวัน")
        daily_sales = df.groupby('transaction_date')['total_sales'].sum().reset_index()
        fig = px.line(daily_sales, x='transaction_date', y='total_sales', markers=True)
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "บันทึกยอดขาย":
        st.title("📝 รายการบันทึกการขายล่าสุด")
        # โชว์ตารางข้อมูลดิบ
        st.dataframe(df.sort_values(by='transaction_date', ascending=False).head(100), use_container_width=True)

    elif menu == "จัดการสินค้า":
        st.title("📦 วิเคราะห์แยกตามประเภทสินค้า")
        # กราฟแท่งแสดงหมวดหมู่ที่ขายดีที่สุด
        cat_sales = df.groupby('product_category')['total_sales'].sum().reset_index().sort_values(by='total_sales', ascending=False)
        fig_bar = px.bar(cat_sales, x='product_category', y='total_sales', color='product_category', title="ยอดขายแยกตามหมวดหมู่")
        st.plotly_chart(fig_bar, use_container_width=True)

    elif menu == "คาดการณ์ยอดขาย":
        st.title("🤖 ระบบคาดการณ์อัจฉริยะ (AI)")
        st.info("เตรียมข้อมูลสำหรับการทำโมเดล XGBoost")
        # กราฟย้อนหลัง 14 วัน
        recent_14 = df.groupby('transaction_date')['total_sales'].sum().tail(14)
        st.line_chart(recent_14)
        st.write("พร้อมนำข้อมูลนี้ไปรัน Model Prediction ต่อไป")

else:
    st.warning("⚠️ ไม่พบไฟล์ 'Coffee Shop Sales.xlsx - Transactions.csv' กรุณาตรวจสอบใน GitHub ของอาจารย์อีกครั้งครับ")

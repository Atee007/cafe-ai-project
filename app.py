import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. ตั้งค่าหน้าจอ (ใช้สีเดิมมาตรฐาน ไม่มีการตกแต่งสีครีม)
st.set_page_config(layout="wide", page_title="Coffee Shop Dashboard")

# 2. ฟังก์ชันดึงข้อมูลจากไฟล์ Excel (ชี้ไปที่ชื่อไฟล์ที่อาจารย์บอก)
@st.cache_data
def load_data():
    # ชื่อไฟล์ต้องตรงกับที่อยู่ใน GitHub (ห้ามมี /content/ นำหน้า)
    file_name = 'Coffee Shop Sales.xlsx' 
    
    if os.path.exists(file_name):
        # อ่านไฟล์ Excel
        df = pd.read_excel(file_name)
        
        # แปลงวันที่ให้โปรแกรมเข้าใจ
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # สร้างคอลัมน์ "ยอดขาย" โดยเอา (จำนวนขาย x ราคาต่อหน่วย)
        # ใช้ชื่อคอลัมน์จริงจากไฟล์ของอาจารย์
        df['total_sales'] = df['transaction_qty'] * df['unit_price']
        return df
    return None

df = load_data()

# 3. แถบเมนูข้างซ้าย (Sidebar)
with st.sidebar:
    st.title("☕ ระบบจัดการร้านกาแฟ")
    menu = st.radio("เลือกหน้าเมนู", ["แดชบอร์ด", "บันทึกยอดขาย", "จัดการสินค้า", "คาดการณ์ยอดขาย"])
    st.divider()
    if df is not None:
        st.success("เชื่อมต่อข้อมูลสำเร็จ")
    else:
        st.error("ไม่พบไฟล์ Coffee Shop Sales.xlsx")

# 4. การแสดงผล (Main Content)
if df is not None:
    if menu == "แดชบอร์ด":
        st.title("📊 สรุปยอดขาย")
        
        # คำนวณตัวเลขจริงมาโชว์บนการ์ด
        total_revenue = df['total_sales'].sum()
        total_items = df['transaction_qty'].sum()
        avg_ticket = df['total_sales'].mean()

        # สร้างการ์ดแสดงผล 3 ช่อง (แบบมาตรฐาน)
        col1, col2, col3 = st.columns(3)
        col1.metric("รายได้รวม", f"฿{total_revenue:,.2f}")
        col2.metric("จำนวนชิ้นที่ขายได้", f"{total_items:,.0f} ชิ้น")
        col3.metric("ยอดขายเฉลี่ยต่อบิล", f"฿{avg_ticket:.2f}")

        # กราฟเส้นแสดงยอดขายรายวัน
        st.subheader("📈 แนวโน้มการขายรายวัน")
        daily_sales = df.groupby('transaction_date')['total_sales'].sum().reset_index()
        fig = px.line(daily_sales, x='transaction_date', y='total_sales', markers=True)
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "บันทึกยอดขาย":
        st.title("📝 รายการธุรกรรม")
        st.dataframe(df, use_container_width=True)

    elif menu == "จัดการสินค้า":
        st.title("📦 วิเคราะห์แยกตามประเภท")
        cat_sales = df.groupby('product_category')['total_sales'].sum().reset_index()
        fig_bar = px.bar(cat_sales, x='product_category', y='total_sales', color='product_category')
        st.plotly_chart(fig_bar, use_container_width=True)

    elif menu == "คาดการณ์ยอดขาย":
        st.title("🤖 การคาดการณ์ (AI Forecasting)")
        st.info("ส่วนนี้พร้อมสำหรับใส่โมเดล XGBoost ของอาจารย์แล้ว")
        # โชว์กราฟย้อนหลัง 14 วันเพื่อเตรียมทำนาย
        recent_data = df.groupby('transaction_date')['total_sales'].sum().tail(14)
        st.line_chart(recent_data)

else:
    st.warning("⚠️ กรุณาอัปโหลดไฟล์ 'Coffee Shop Sales.xlsx' ขึ้นไปไว้ใน GitHub ที่เดียวกับไฟล์ app.py")

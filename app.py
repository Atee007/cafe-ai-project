import streamlit as st
import pandas as pd
import plotly.express as px
import os
from xgboost import XGBRegressor

# --- 1. ตั้งค่าพื้นฐาน (ตาม Non-Functional Requirements ข้อ 5.1 & 5.4) ---
st.set_page_config(layout="wide", page_title="ລະບົບ AI ຮ້ານກາເຟລາວ")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Noto Sans Lao', sans-serif; }
    .stApp { background-color: #F8F9FA; }
    .stMetric { background-color: white; border-radius: 10px; padding: 15px; border: 1px solid #D4AF37; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ฟังก์ชันโหลดข้อมูล (Data Preparation) ---
@st.cache_data
def load_data():
    file_path = 'Coffee Shop Sales.xlsx'
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        # ตรวจสอบและเปลี่ยนชื่อคอลัมน์ให้ตรงตามระบบ (ข้อ 4.2)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Mapping คอลัมน์ที่จำเป็น
        col_map = {
            'transaction_date': next((c for c in df.columns if 'date' in c), None),
            'transaction_qty': next((c for c in df.columns if 'qty' in c or 'quantity' in c), None),
            'unit_price': next((c for c in df.columns if 'price' in c), None),
            'product_category': next((c for c in df.columns if 'category' in c or 'product' in c), 'General')
        }
        
        if col_map['transaction_date'] and col_map['transaction_qty'] and col_map['unit_price']:
            df = df.rename(columns={
                col_map['transaction_date']: 'date',
                col_map['transaction_qty']: 'qty',
                col_map['unit_price']: 'price',
                col_map['product_category']: 'category'
            })
            df['date'] = pd.to_datetime(df['date'])
            df['total_sales'] = df['qty'] * df['price']
            return df
    return None

df = load_data()

# --- 3. ส่วนของเมนู (ตามข้อ 2 ผู้ใช้งานระบบ) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924514.png", width=80)
    st.title("☕ ລະບົບຄາເຟ່ AI")
    user_role = st.selectbox("ສິດການໃຊ້ງານ", ["Admin (ເຈົ້າຂອງຮ້ານ)", "Staff (ພະນັກງານ)"])
    st.divider()
    
    if user_role == "Admin (ເຈົ້າຂອງຮ้ານ)":
        menu = st.radio("ລາຍການລະບົບ", ["📊 ຕິດຕາມຍອດຂາຍ", "🤖 ຄາດການຍອດຂາຍ (AI)", "📦 ຈັດການສິນຄ້າ"])
    else:
        menu = st.radio("ລາຍการລະບົບ", ["📝 ບັນທຶກຍອດຂາຍรายວັນ", "📊 ຕິດຕາມຍອດຂาย"])

# --- 4. การแสดงผลตามฟังก์ชัน (Functional Requirements) ---
if df is not None:
    # 3.3 ระบบติดตามยอดขาย (Sales Monitoring)
    if menu == "📊 ຕິດຕามຍອດຂາຍ":
        st.header("📊 ລາຍງານການຕິດຕາມຍອດຂາຍ")
        
        c1, c2, c3 = st.columns(3)
        total_sales = df['total_sales'].sum()
        c1.metric("ຍອດຂາຍລວມທັງໝົດ", f"₭ {total_sales:,.0f}")
        c2.metric("ຈຳນວນລາຍການ", f"{len(df):,} ບິນ")
        c3.metric("ຍອດຂາຍສະເລ່ຍ/ວັນ", f"₭ {total_sales/df['date'].nunique():,.0f}")

        st.subheader("📈 ແນວໂນ້ມຍອດຂາຍ")
        daily_sales = df.groupby('date')['total_sales'].sum().reset_index()
        fig = px.line(daily_sales, x='date', y='total_sales', markers=True, 
                      title="ຍອດຂາຍລາຍວັນ (₭)", color_discrete_sequence=['#D4AF37'])
        st.plotly_chart(fig, use_container_width=True)

    # 3.5 ระบบคาดการณ์ยอดขาย (Sales Forecasting)
    elif menu == "🤖 ຄາດການຍອດຂາຍ (AI)":
        st.header("🤖 ການຄາດການຍອດຂາຍລ່ວງໜ້າ (AI Forecasting)")
        
        # เตรียมข้อมูล (Modeling Step)
        daily_df = df.groupby('date')['total_sales'].sum().reset_index()
        daily_df['dow'] = daily_df['date'].dt.dayofweek
        daily_df['month'] = daily_df['date'].dt.month
        
        model = XGBRegressor(n_estimators=100)
        model.fit(daily_df[['dow', 'month']], daily_df['total_sales'])
        
        # คาดการณ์ 7 วันข้างหน้า
        last_date = daily_df['date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
        future_X = pd.DataFrame({'dow': future_dates.dayofweek, 'month': future_dates.month})
        preds = model.predict(future_X)
        
        res_df = pd.DataFrame({'ວັນທີ': future_dates.strftime('%d/%m/%Y'), 'ຍອດຄາດການ (₭)': preds})
        
        st.subheader("🔮 ຜົນການຄາດການ 7 ວັນຂ້າງໜ້າ")
        st.table(res_df.style.format({'ຍອດຄາດການ (₭)': '{:,.0f}'}))
        
        # 3.4 ระบบ AI Automation (วิเคราะห์เบื้องต้น)
        st.info(f"💡 **AI Analysis:** ຍອດຂາຍສະເລ່ຍທີ່ຄາດການແມ່ນ ₭ {preds.mean():,.0f}. ກະລຸນາກຽມວັດຖຸດິບໃຫ້ພຽງພໍ.")

    # 3.2 ระบบบันทึกยอดขาย (Sales Recording)
    elif menu == "📝 ບັນທຶກຍອດຂາຍรายວັນ":
        st.header("📝 ບັນທຶກຂໍ້ມູນການຂາຍ (Staff Level)")
        with st.form("sales_form"):
            col1, col2 = st.columns(2)
            p_name = col1.selectbox("ຊື່ສິນຄ້າ", df['category'].unique())
            p_qty = col2.number_input("ຈຳນວນ", min_value=1)
            p_price = col1.number_input("ລາຄາຕໍ່หน่วย", min_value=0)
            submitted = st.form_submit_button("ບັນທຶກການຂາຍ")
            if submitted:
                st.success(f"ບັນທຶກ {p_name} ຈຳນວນ {p_qty} ລາຍການສຳເລັດ!")

    # 3.1 ระบบจัดการสินค้า (Product Management)
    elif menu == "📦 จัดการสินค้า":
        st.header("📦 ການຈັດການຂໍ້ມູນສິນຄ້າ")
        st.dataframe(df[['category', 'price']].drop_duplicates(), use_container_width=True)
        st.button("➕ ເພີ່ມສິນຄ້າໃໝ່")

else:
    st.error("❌ ບໍ່ພົບໄຟລ໌ 'Coffee Shop Sales.xlsx' ໃນລະບົບ GitHub ຂອງທ່ານ")

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from xgboost import XGBRegressor

# --- 1. ການຕັ້ງຄ່າໜ້າຈໍ (ຕາມຂໍ້ 5.1 & 5.4: UI/UX ແລະ ພາສາລາວ) ---
st.set_page_config(layout="wide", page_title="ລະບົບ AI ຮ້ານກາເຟລາວ")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Noto Sans Lao', sans-serif; }
    .stApp { background-color: #FDFBF7; }
    .stMetric { background-color: white; border: 1px solid #D4AF37; border-radius: 12px; padding: 15px; }
    [data-testid="stSidebar"] { background-color: #3D2B1F; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ຟັງຊັນຈັດການຂໍ້ມູນ (ຕາມຂໍ້ 4: ຂໍ້ມູນພາຍໃນລະບົບ) ---
@st.cache_data
def load_data():
    file_path = 'Coffee Shop Sales.xlsx'
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        # ປັບຊື່ຄໍລຳໃຫ້ເປັນມາດຕະຖານຕາມຂໍ້ 4.2
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in df.columns if 'date' in c), None)
        qty_col = next((c for c in df.columns if 'qty' in c or 'quantity' in c), None)
        price_col = next((c for c in df.columns if 'price' in c), None)
        cat_col = next((c for c in df.columns if 'category' in c or 'product' in c), 'Category')

        if date_col and qty_col and price_col:
            df = df.rename(columns={date_col: 'date', qty_col: 'qty', price_col: 'price', cat_col: 'category'})
            df['date'] = pd.to_datetime(df['date'])
            df['total_sales'] = df['qty'] * df['price']
            return df
    return None

df = load_data()

# --- 3. ລະບົບຄວາມປອດໄພ ແລະ ຜູ້ໃຊ້ (ຕາມຂໍ້ 2 & 5.3: Login & Roles) ---
with st.sidebar:
    st.title("☕ ຄາເຟ່ AI ໂປຣ")
    st.subheader("Login System")
    user_role = st.selectbox("ເລືອກສິດການເຂົ້າໃຊ້", ["Admin (ເຈົ້າຂອງຮ້ານ)", "Staff (ພະນັກງານ)"])
    st.divider()

# --- 4. ສ່ວນສະແດງຜົນຕາມ Functional Requirements (ຂໍ້ 3) ---
if df is not None:
    # --- ຂໍ້ 3.3 ລະບົບຕິດຕາມຍອດຂາຍ (Sales Monitoring) ---
    if user_role == "Admin (ເຈົ້າຂອງຮ້ານ)":
        menu = st.sidebar.radio("ເມນູຫຼັກ", ["📊 ຕິດຕາມຍອດຂາຍ", "🤖 ຄາດການຍອດຂາຍ (AI)", "📦 ຈັດການສິນຄ້າ"])
    else:
        menu = st.sidebar.radio("ເມນູຫຼັກ", ["📝 ບັນທຶກການຂາຍ", "📊 ຕິດຕາມຍອດຂາຍ"])

    if menu == "📊 ຕິດຕາມຍອດຂາຍ":
        st.header("📊 ລະບົບຕິດຕາມຍອດຂາຍ (Sales Monitoring)")
        
        # ຂໍ້ 3.4 ລະບົບ AI Automation (ສະຫຼຸບອັດຕະໂນມັດ)
        c1, c2, c3 = st.columns(3)
        total_revenue = df['total_sales'].sum()
        c1.metric("ຍອດຂາຍລວມ (₭)", f"{total_revenue:,.0f}")
        c2.metric("ຈຳນວນບິນທັງໝົດ", f"{len(df):,}")
        c3.metric("ຍອດຂາຍສະເລ່ຍ/ວັນ", f"{total_revenue/df['date'].nunique():,.0f}")

        st.subheader("📈 ກຣາຟສະແດງແນວໂນ້ມຍອດຂາຍ (Daily/Weekly)")
        daily_sales = df.groupby('date')['total_sales'].sum().reset_index()
        fig = px.area(daily_sales, x='date', y='total_sales', color_discrete_sequence=['#D4AF37'])
        st.plotly_chart(fig, use_container_width=True)

    # --- ຂໍ້ 3.5 ລະບົບຄາດການຍອດຂາຍ (Sales Forecasting) ---
    elif menu == "🤖 ຄາດການຍອດຂາຍ (AI)":
        st.header("🤖 ລະບົບ AI ພະຍາກອນຍອດຂາຍ (XGBoost)")
        
        # ການກຽມຂໍ້ມູນ ແລະ Modeling (ຂໍ້ 6 & 7)
        daily_df = df.groupby('date')['total_sales'].sum().reset_index()
        daily_df['dow'] = daily_df['date'].dt.dayofweek
        daily_df['month'] = daily_df['date'].dt.month
        
        model = XGBRegressor(n_estimators=100)
        model.fit(daily_df[['dow', 'month']], daily_df['total_sales'])
        
        # ພະຍາກອນລ່ວງໜ້າ 7 ວັນ
        last_date = daily_df['date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
        future_X = pd.DataFrame({'dow': future_dates.dayofweek, 'month': future_dates.month})
        preds = model.predict(future_X)
        
        res_df = pd.DataFrame({'ວັນທີ': future_dates.strftime('%d/%m/%Y'), 'ຍອດຄາດການ (₭)': preds})
        
        st.table(res_df.style.format({'ຍອດຄາດການ (₭)': '{:,.0f}'}))
        
        # ຂໍ້ 3.6 ລະບົບແຈ້ງເຕືອນ (Notification Alert)
        if preds.mean() < (total_revenue/df['date'].nunique()):
            st.warning("⚠️ ແຈ້ງເຕືອນ: ຍອດຂາຍຄາດການຕ່ຳກວ່າຄ່າສະເລ່ຍ! ກະລຸນາກຽມແຜນການຕະຫຼາດ.")
        else:
            st.success("✅ ແຈ້ງເຕືອນ: ຍອດຂາຍມີແນວໂນ້ມດີກວ່າຄ່າສະເລ່ຍ.")

    # --- ຂໍ້ 3.2 ລະບົບບັນທຶກຍອດຂາຍ (Sales Recording) ---
    elif menu == "📝 ບັນທຶກການຂາຍ":
        st.header("📝 ບັນທຶກຂໍ້ມູນການຂາຍລາຍວັນ (Staff)")
        with st.form("recording"):
            col1, col2 = st.columns(2)
            product = col1.selectbox("ເລືອກສິນຄ້າ", df['category'].unique())
            amount = col2.number_input("ຈຳນວນ", min_value=1)
            submit = st.form_submit_button("ບັນທຶກການຂາຍ")
            if submit:
                st.success(f"ບັນທຶກ {product} ຈຳນວນ {amount} ລາຍການສຳເລັດ!")

    # --- ຂໍ້ 3.1 ລະບົບຈັດການສິນຄ້າ (Product Management) ---
    elif menu == "📦 ຈັດການສິນຄ້າ":
        st.header("📦 ການຈັດການຂໍ້ມູນສິນຄ້າ (Admin Only)")
        st.write("ລາຍການສິນຄ້າພາຍໃນຮ້ານ")
        product_list = df[['category', 'price']].drop_duplicates()
        st.dataframe(product_list, use_container_width=True)
        if st.button("ເພີ່ມສິນຄ້າໃໝ່"):
            st.info("ຟັງຊັນນີ້ກຳລັງພັດທະນາ (Student Level Prototype)")

else:
    st.error("❌ ບໍ່ພົບໄຟລ໌ 'Coffee Shop Sales.xlsx' ກະລຸນາກວດສອບໃນ GitHub")

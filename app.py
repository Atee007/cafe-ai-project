import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from xgboost import XGBRegressor

# --- 1. UI Setup (ພາສາລາວ & Design) ---
st.set_page_config(layout="wide", page_title="ລະບົບ AI ຮ້ານກາເຟລາວ", page_icon="☕")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Noto Sans Lao', sans-serif; }
    .stApp { background-color: #FDFBF7; }
    [data-testid="stSidebar"] { background-color: #3D2B1F; }
    [data-testid="stSidebar"] * { color: #D4AF37 !important; }
    .stMetric { background-color: #FFFFFF; border: 1px solid #D4AF37; border-radius: 12px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Data Engine (ລະບົບໂຫລດຂໍ້ມູນອັດສະລິຍະ) ---
@st.cache_data
def load_and_clean():
    # ຄົ້ນຫາໄຟລ໌ .xlsx ໃນ GitHub
    files = [f for f in os.listdir() if f.endswith('.xlsx')]
    if not files:
        return None, None
    
    target_file = files[0]
    df = pd.read_excel(target_file)
    
    # Mapping ຫົວຕາຕະລາງອັດຕະໂນມັດ (ປ້ອງກັນ KeyError)
    date_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['date', 'ວັນທີ', 'วันที่'])), None)
    qty_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['qty', 'ຈຳນວນ', 'จำนวน'])), None)
    price_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['price', 'ລາຄາ', 'ราคา'])), None)
    time_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['time', 'ເວລາ', 'เวลา'])), None)

    if date_col and qty_col and price_col:
        df = df.rename(columns={date_col: 'transaction_date', qty_col: 'transaction_qty', price_col: 'unit_price'})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['total_sales'] = pd.to_numeric(df['transaction_qty'], errors='coerce') * pd.to_numeric(df['unit_price'], errors='coerce')
        
        if time_col:
            df['hour'] = pd.to_numeric(df[time_col].astype(str).str.split(':').str[0], errors='coerce').fillna(10)
        else:
            df['hour'] = 10
            
        return df.dropna(subset=['transaction_date']), target_file
    return None, target_file

# ເອີ້ນໃຊ້ຟັງຊັນ (ແກ້ໄຂ NameError ໂດຍການກຳນົດຕົວແປໃຫ້ຊັດເຈນ)
df, current_file = load_and_clean()

# --- 3. Sidebar Menu ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>ຄາເຟ່ AI ລະດັບໂປຣ</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/924/924514.png", width=100)
    st.divider()
    menu = st.radio("ເລືອກລາຍການ:", ["📊 ສະຫຼຸບຍອດຂາຍ", "🤖 AI ພະຍາກອນ", "📦 ສິນຄ້າຂາຍດີ"])
    st.divider()
    # ໃຊ້ຕົວແປ current_file ທີ່ປະກາດໄວ້ຂ້າງເທິງ
    st.info(f"📂 ໄຟລ໌: {current_file if current_file else 'ບໍ່ພົບຂໍ້ມູນ'}")

# --- 4. Main App Logic ---
if df is not None:
    if menu == "📊 ສະຫຼຸບຍອດຂາຍ":
        st.header("📊 ບົດສະຫຼຸບຍອດຂາຍລວມ (ສະກຸນເງິນກີບ)")
        
        c1, c2, c3 = st.columns(3)
        total_kip = df['total_sales'].sum()
        c1.metric("ຍອດຂາຍທັງໝົດ", f"₭ {total_kip:,.0f}")
        c2.metric("ຈຳນວນບິນ", f"{len(df):,} ລາຍການ")
        c3.metric("ສະເລ່ຍ/ບິນ", f"₭ {df['total_sales'].mean():,.0f}")

        st.subheader("📈 ກຣາຟຍອດຂາຍລາຍວັນ")
        daily = df.groupby('transaction_date')['total_sales'].sum().reset_index()
        fig = px.area(daily, x='transaction_date', y='total_sales', color_discrete_sequence=['#D4AF37'])
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "🤖 AI ພະຍາກອນ":
        st.header("🤖 AI ພະຍາກອນຍອດຂາຍ (XGBoost)")
        
        # ປັບແຕ່ງ Feature ສໍາລັບ AI
        daily_df = df.groupby('transaction_date')['total_sales'].sum().reset_index()
        daily_df['day_of_week'] = daily_df['transaction_date'].dt.dayofweek
        daily_df['month'] = daily_df['transaction_date'].dt.month
        
        X = daily_df[['day_of_week', 'month']]
        y = daily_df['total_sales']
        
        model = XGBRegressor(n_estimators=100).fit(X, y)
        
        # ພະຍາກອນ 7 ວັນ
        future_dates = pd.date_range(daily_df['transaction_date'].max() + pd.Timedelta(days=1), periods=7)
        future_X = pd.DataFrame({'day_of_week': future_dates.dayofweek, 'month': future_dates.month})
        preds = model.predict(future_X)
        
        res = pd.DataFrame({'ວັນທີ': future_dates.strftime('%d/%m/%Y'), 'ຍອດຄາດການ (₭)': preds})
        st.success("✅ AI ວິເຄາະຂໍ້ມູນສຳເລັດ!")
        st.table(res.style.format({'ຍອດຄາດການ (₭)': '{:,.0f}'}))
        
        st.warning("💡 **ຄຳແນະນຳ:** ອີງຕາມ AI, ຍອດຂາຍຂອງທ່ານຈະມີການປ່ຽນແປງຕາມວັນຢຸດພັກຜ່ອນ.")

    elif menu == "📦 ສິນຄ້າຂายດີ":
        st.header("📦 ວິເຄາະປະເພດສິນຄ້າ")
        if 'product_category' in df.columns:
            fig_pie = px.pie(df, values='total_sales', names='product_category', hole=0.5)
            st.plotly_chart(fig_pie)
        else:
            st.write("ບໍ່ພົບຂໍ້ມູນປະເພດສິນຄ້າ")

else:
    st.error("⚠️ ກະລຸນາກວດສອບໄຟລ໌ .xlsx ໃນ GitHub ຂອງທ່ານ")

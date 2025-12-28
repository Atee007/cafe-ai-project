import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from xgboost import XGBRegressor

# --- 1. UI Setup (ພາສາລາວ & Design ລະດັບ Premium) ---
st.set_page_config(layout="wide", page_title="ລະບົບ AI ຮ້ານກາເຟລາວ", page_icon="☕")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Noto Sans Lao', sans-serif; }
    .stApp { background-color: #FDFBF7; }
    [data-testid="stSidebar"] { background-color: #3D2B1F; }
    [data-testid="stSidebar"] * { color: #D4AF37 !important; }
    .stMetric { background-color: #FFFFFF; border-left: 5px solid #D4AF37; border-radius: 12px; padding: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ລະບົບຄົ້ນຫາ ແລະ ໂຫລດຂໍ້ມູນ (Smart Loader) ---
@st.cache_data
def load_and_clean_data():
    # ຄົ້ນຫາທຸກໄຟລ໌ .xlsx ທີ່ມີໃນ Folder
    all_files = [f for f in os.listdir() if f.endswith('.xlsx')]
    
    if not all_files:
        return None, None
    
    # ເລືອກໄຟລ໌ທຳອິດທີ່ເຈິ
    selected_file = all_files[0]
    try:
        df = pd.read_excel(selected_file)
        
        # Mapping ຫົວຕາຕະລາງແບບອັດສະລິຍະ (ຮອງຮັບທັງ ລາວ/ໄທ/ອັງກິດ)
        col_map = {
            'date': next((c for c in df.columns if any(k in str(c).lower() for k in ['date', 'ວັນທີ', 'วันที่'])), None),
            'qty': next((c for c in df.columns if any(k in str(c).lower() for k in ['qty', 'ຈຳນວນ', 'จำนวน'])), None),
            'price': next((c for c in df.columns if any(k in str(c).lower() for k in ['price', 'ລາຄາ', 'ราคา'])), None),
            'cat': next((c for c in df.columns if any(k in str(c).lower() for k in ['category', 'type', 'ປະເພດ', 'สินค้า'])), None)
        }

        if col_map['date'] and col_map['qty'] and col_map['price']:
            # ແປງຂໍ້ມູນໃຫ້ເປັນມາດຕະຖານ
            df = df.rename(columns={col_map['date']: 'date', col_map['qty']: 'qty', col_map['price']: 'price', col_map['cat']: 'category'})
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['total_sales'] = pd.to_numeric(df['qty'], errors='coerce') * pd.to_numeric(df['price'], errors='coerce')
            df = df.dropna(subset=['date', 'total_sales'])
            return df, selected_file
    except Exception as e:
        st.error(f"ເກີດຂໍ້ຜິດພາດໃນການອ່ານໄຟລ໌: {e}")
    
    return None, None

df, current_file = load_and_clean_data()

# --- 3. Sidebar Menu ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>☕ ຄາເຟ່ AI ໂປຣ</h2>", unsafe_allow_html=True)
    st.divider()
    menu = st.radio("ລາຍການລະບົບ:", ["📊 ສະຫຼຸບຍອດຂາຍ", "🤖 AI ພະຍາກອນ", "📦 ວິເຄາະສິນຄ້າ"])
    st.divider()
    if current_file:
        st.success(f"✅ ພົບໄຟລ໌: {current_file}")
    else:
        st.error("❌ ບໍ່ພົບໄຟລ໌ .xlsx")

# --- 4. Main Display ---
if df is not None:
    if menu == "📊 ສະຫຼຸບຍອດຂາຍ":
        st.header("📊 ບົດສະຫຼຸບຍອດຂາຍ (ສະກຸນເງິນກີບ)")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("ຍອດຂາຍລວມ", f"₭ {df['total_sales'].sum():,.0f}")
        c2.metric("ຈຳນວນລາຍການ", f"{len(df):,} ບິນ")
        c3.metric("ສະເລ່ຍຕໍ່ບິນ", f"₭ {df['total_sales'].mean():,.0f}")

        st.subheader("📈 ແນວໂນ້ມຍອດຂາຍລາຍວັນ")
        daily = df.groupby('date')['total_sales'].sum().reset_index()
        fig = px.line(daily, x='date', y='total_sales', markers=True, color_discrete_sequence=['#D4AF37'])
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "🤖 AI ພະຍາກອນ":
        st.header("🤖 AI ພະຍາກອນຍອດຂາຍ 7 ວັນ (XGBoost)")
        
        # ກຽມຂໍ້ມູນສຳລັບ AI (CRISP-DM Modeling Phase)
        daily_df = df.groupby('date')['total_sales'].sum().reset_index()
        daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
        daily_df['month'] = daily_df['date'].dt.month
        
        X = daily_df[['day_of_week', 'month']]
        y = daily_df['total_sales']
        
        # ສ້າງ Model
        model = XGBRegressor(n_estimators=100).fit(X, y)
        
        # ພະຍາກອນອະນາຄົດ
        future_dates = pd.date_range(daily_df['date'].max() + pd.Timedelta(days=1), periods=7)
        future_X = pd.DataFrame({'day_of_week': future_dates.dayofweek, 'month': future_dates.month})
        preds = model.predict(future_X)
        
        res = pd.DataFrame({'ວັນທີ': future_dates.strftime('%d/%m/%Y'), 'ຍອດພະຍາກອນ (₭)': preds})
        st.table(res.style.format({'ຍອດພະຍາກອນ (₭)': '{:,.0f}'}))
        st.info("💡 AI ແນະນຳ: ກຽມພ້ອມຮັບມືກັບຍອດຂາຍທີ່ຈະເພີ່ມຂຶ້ນໃນວັນຢຸດ!")

    elif menu == "📦 ວິເຄາະສິນຄ້າ":
        st.header("📦 ສັດສ່ວນຍອດຂາຍສິນຄ້າ")
        if 'category' in df.columns:
            fig_pie = px.pie(df, values='total_sales', names='category', hole=0.4)
            st.plotly_chart(fig_pie)

else:
    st.warning("⚠️ ກະລຸນາອັບໂຫລດໄຟລ໌ Excel (.xlsx) ເຂົ້າໄປໃນ GitHub ຂອງທ່ານກ່ອນ!")

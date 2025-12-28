import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from xgboost import XGBRegressor

# --- 1. UI Setup ---
st.set_page_config(layout="wide", page_title="ລະບົບ AI ຮ້ານກາເຟລາວ", page_icon="☕")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Noto Sans Lao', sans-serif; }
    .stApp { background-color: #FDFBF7; }
    [data-testid="stSidebar"] { background-color: #3D2B1F; }
    [data-testid="stSidebar"] * { color: #D4AF37 !important; }
    .stMetric { background-color: #FFFFFF; border-left: 5px solid #D4AF37; border-radius: 12px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Smart Data Loader (ແກ້ໄຂໃໝ່ໃຫ້ໂຫດກວ່າເກົ່າ) ---
@st.cache_data
def load_and_clean_data():
    all_files = [f for f in os.listdir() if f.endswith('.xlsx')]
    if not all_files:
        return None, None
    
    selected_file = all_files[0]
    try:
        df = pd.read_excel(selected_file)
        
        # --- ຂັ້ນຕອນຄົ້ນຫາຄໍລຳແບບບັງຄັບ (ບາດແຜສຸດທ້າຍ) ---
        # 1. ຫາຄໍລຳວັນທີ
        date_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['date', 'ວັນທີ', 'วันที่'])), df.columns[0])
        # 2. ຫາຄໍລຳຈຳນວນ
        qty_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['qty', 'ຈຳນວນ', 'จำนวน'])), df.columns[1] if len(df.columns) > 1 else df.columns[0])
        # 3. ຫາຄໍລຳລາຄາ
        price_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['price', 'ລາຄາ', 'ราคา'])), df.columns[2] if len(df.columns) > 2 else df.columns[0])
        
        # ແປງຊື່ໃຫ້ເປັນມາດຕະຖານ
        df = df.rename(columns={date_col: 'date', qty_col: 'qty', price_col: 'price'})
        
        # ເຮັດຄວາມສະອາດຂໍ້ມູນ
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(0)
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        df['total_sales'] = df['qty'] * df['price']
        
        # ລຶບແຖວທີ່ວັນທີຜິດພາດ
        df = df.dropna(subset=['date'])
        
        if len(df) > 0:
            return df, selected_file
        else:
            return None, selected_file
    except:
        return None, None

df, current_file = load_and_clean_data()

# --- 3. Sidebar Menu ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>☕ ຄາເຟ່ AI ໂປຣ</h2>", unsafe_allow_html=True)
    st.divider()
    menu = st.radio("ລາຍການລະບົບ:", ["📊 ສະຫຼຸບຍອດຂາຍ", "🤖 AI ພະຍາກອນ", "📦 ວິເຄາະສິນຄ້າ"])
    st.divider()
    if current_file:
        st.success(f"✅ ໄຟລ໌ທີ່ໃຊ້: {current_file}")
    else:
        st.error("❌ ບໍ່ພົບໄຟລ໌ .xlsx ໃນ GitHub")

# --- 4. Display Content ---
if df is not None and not df.empty:
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
        daily_df = df.groupby('date')['total_sales'].sum().reset_index()
        if len(daily_df) > 1:
            daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
            daily_df['month'] = daily_df['date'].dt.month
            model = XGBRegressor(n_estimators=100).fit(daily_df[['day_of_week', 'month']], daily_df['total_sales'])
            
            future_dates = pd.date_range(daily_df['date'].max() + pd.Timedelta(days=1), periods=7)
            future_X = pd.DataFrame({'day_of_week': future_dates.dayofweek, 'month': future_dates.month})
            preds = model.predict(future_X)
            
            res = pd.DataFrame({'ວັນທີ': future_dates.strftime('%d/%m/%Y'), 'ຍອດພະຍາກອນ (₭)': preds})
            st.table(res.style.format({'ຍອດພະຍາກອນ (₭)': '{:,.0f}'}))
        else:
            st.warning("ຂໍ້ມູນໜ້ອຍເກີນໄປສຳລັບການພະຍາກອນ (ຕ້ອງການຢ່າງໜ້ອຍ 2 ມື້)")

    elif menu == "📦 ວິເຄາະສິນຄ້າ":
        st.header("📦 ສັດສ່ວນຍອດຂາຍ")
        # ຖ້າບໍ່ມີຄໍລຳ Category ໃຫ້ໃຊ້ຄໍລຳອື່ນແທນ
        cat_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['cat', 'item', 'product', 'ລາຍການ'])), df.columns[0])
        fig_pie = px.pie(df, values='total_sales', names=cat_col, hole=0.4)
        st.plotly_chart(fig_pie)

else:
    st.error("⚠️ ລະບົບຫາຂໍ້ມູນໃນໄຟລ໌ບໍ່ເຈິ! ກະລຸນາກວດສອບວ່າໄຟລ໌ Excel ຂອງທ່ານມີຂໍ້ມູນວັນທີ, ຈຳນວນ ແລະ ລາຄາ ຫຼື ບໍ່?")

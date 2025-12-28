import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from xgboost import XGBRegressor

# --- 1. UI Setup (ລະດັບ Premium ພາສາລາວ) ---
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

# --- 2. Smart Loader (ປັບໃຫ້ເຂົ້າກັບ Coffee Shop Sales.xlsx) ---
@st.cache_data
def load_coffee_data():
    file_name = 'Coffee Shop Sales.xlsx'
    
    if not os.path.exists(file_name):
        return None
    
    try:
        df = pd.read_excel(file_name)
        
        # Mapping ຊື່ຄໍລຳໃຫ້ເປັນພາສາກາງ (Transaction Date, Qty, Unit Price)
        # ລະບົບຈະຄົ້ນຫາຊື່ທີ່ໃກ້ຄຽງທີ່ສຸດ
        cols = {
            'date': next((c for c in df.columns if 'date' in c.lower() or 'ວັນທີ' in c), df.columns[0]),
            'qty': next((c for c in df.columns if 'qty' in c.lower() or 'quantity' in c.lower() or 'ຈຳນວນ' in c), None),
            'price': next((c for c in df.columns if 'price' in c.lower() or 'ລາຄາ' in c), None),
            'cat': next((c for c in df.columns if 'category' in c.lower() or 'product' in c.lower() or 'ປະເພດ' in c), None)
        }

        if cols['date'] and cols['qty'] and cols['price']:
            df = df.rename(columns={cols['date']: 'date', cols['qty']: 'qty', cols['price']: 'price', cols['cat']: 'category'})
            
            # Cleaning Data
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(0)
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
            df['total_sales'] = df['qty'] * df['price']
            
            return df.dropna(subset=['date'])
    except Exception as e:
        st.error(f"ເກີດຂໍ້ຜິດພາດ: {e}")
    return None

df = load_coffee_data()

# --- 3. Sidebar Menu ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>☕ ຄາເຟ່ AI ໂປຣ</h2>", unsafe_allow_html=True)
    st.divider()
    menu = st.radio("ລາຍການລະບົບ:", ["📊 ສະຫຼຸບຍອດຂາຍ", "🤖 AI ພະຍາກອນ", "📦 ວິເຄາະສິນຄ້າ"])
    st.divider()
    if df is not None:
        st.success("✅ ໂຫລດໄຟລ໌ Coffee Shop Sales ສຳເລັດ!")
    else:
        st.error("❌ ບໍ່ພົບໄຟລ໌ Coffee Shop Sales.xlsx ໃນ GitHub")

# --- 4. Main Content ---
if df is not None:
    if menu == "📊 ສະຫຼຸບຍອດຂາຍ":
        st.header("📊 ບົດສະຫຼຸບຍອດຂາຍ (ສະກຸນເງິນກີບ)")
        
        c1, c2, c3 = st.columns(3)
        total_sales = df['total_sales'].sum()
        c1.metric("ຍອດຂາຍລວມ", f"₭ {total_sales:,.0f}")
        c2.metric("ຈຳນວນລາຍການ", f"{len(df):,} ລາຍການ")
        c3.metric("ສະເລ່ຍ/ບິນ", f"₭ {df['total_sales'].mean():,.0f}")

        st.subheader("📈 ແນວໂນ້ມຍອດຂາຍລາຍວັນ")
        daily = df.groupby('date')['total_sales'].sum().reset_index()
        fig = px.area(daily, x='date', y='total_sales', color_discrete_sequence=['#D4AF37'])
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "🤖 AI ພະຍາກອນ":
        st.header("🤖 AI ພະຍາກອນຍອດຂາຍ (XGBoost)")
        
        daily_df = df.groupby('date')['total_sales'].sum().reset_index()
        daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
        daily_df['month'] = daily_df['date'].dt.month
        
        # ສ້າງ Model (CRISP-DM Modeling Phase)
        model = XGBRegressor(n_estimators=100).fit(daily_df[['day_of_week', 'month']], daily_df['total_sales'])
        
        # Prediction
        last_date = daily_df['date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
        future_X = pd.DataFrame({'day_of_week': future_dates.dayofweek, 'month': future_dates.month})
        preds = model.predict(future_X)
        
        res = pd.DataFrame({'ວັນທີ': future_dates.strftime('%d/%m/%Y'), 'ຍອດພະຍາກອນ (₭)': preds})
        st.table(res.style.format({'ຍອດພະຍາກອນ (₭)': '{:,.0f}'}))
        st.info("💡 AI Insight: ຍອດຂາຍມີແນວໂນ້ມເພີ່ມຂຶ້ນຕາມຮອບວຽນຂອງອາທິດ.")

    elif menu == "📦 ວິເຄາະສິນຄ້າ":
        st.header("📦 ສັດສ່ວນຍອດຂາຍສິນຄ້າ")
        cat_col = 'category' if 'category' in df.columns else df.columns[0]
        fig_pie = px.pie(df, values='total_sales', names=cat_col, hole=0.4)
        st.plotly_chart(fig_pie)

else:
    st.warning("⚠️ ບໍ່ພົບໄຟລ໌ 'Coffee Shop Sales.xlsx'. ກະລຸນາກວດສອບວ່າຊື່ໄຟລ໌ໃນ GitHub ຂຽນຖືກຕ້ອງແລ້ວ.")

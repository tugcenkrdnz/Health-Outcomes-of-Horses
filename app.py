import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Paket Yükleme
@st.cache_resource
def load_all():
    package = joblib.load('horse_model_package.joblib')
    return package

pkg = load_all()
model, scaler, imputer, dummy_columns = pkg['model'], pkg['scaler'], pkg['imputer'], pkg['columns']
expected_num_cols = scaler.feature_names_in_

st.title("🐎 Gelişmiş At Sağlığı Tahmin Sistemi")
st.warning("Not: Yüksek Nabız ve PCV değerleri 'Died' tahminini güçlendirir.")

with st.form("detayli_form"):
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("Vital Bulgular")
        rectal_temp = st.number_input("Rektal Ateş (°C)", 30.0, 45.0, 38.0)
        pulse = st.number_input("Nabız (Vuruş/Dak)", 20, 250, 60)
        respiratory_rate = st.number_input("Solunum Hızı", 5, 120, 20)
        packed_cell_volume = st.number_input("PCV (Kan Yoğunluğu %)", 10.0, 80.0, 45.0)

    with c2:
        st.subheader("Klinik Durum")
        total_protein = st.number_input("Toplam Protein", 2.0, 100.0, 7.5)
        abdomo_protein = st.number_input("Abdomo Protein", 0.0, 20.0, 2.0)
        nasogastric_reflux_ph = st.number_input("Reflü pH", 0.0, 14.0, 5.0)
        pain = st.selectbox("Ağrı", ["alert", "depressed", "moderate", "severe_pain", "extreme_pain"])
        temp_of_extremities = st.selectbox("Ekstremite Isısı", ["normal", "warm", "cool", "cold"])

    with c3:
        st.subheader("Görsel Belirtiler")
        # ÖLÜMÜ BELİRLEYEN KRİTİK EKSİKLER:
        mucous_membrane = st.selectbox("Diş Eti Rengi", 
            ["normal_pink", "pale_pink", "pale_cyanotic", "bright_pink", "bright_red", "dark_cyanotic"])
        abdomo_appearance = st.selectbox("Karın Sıvısı", ["clear", "cloudy", "serosanguinous"])
        surgery = st.selectbox("Ameliyat?", ["yes", "no"])
        surgical_lesion = st.selectbox("Cerrahi Lezyon?", ["yes", "no"])
        age = st.selectbox("Yaş", ["adult", "young"])

    submit = st.form_submit_button("Analiz Et")

if submit:
    # 1. Veri Sözlüğü
    input_data = {
        'rectal_temp': rectal_temp, 'pulse': pulse, 'respiratory_rate': respiratory_rate,
        'packed_cell_volume': packed_cell_volume, 'total_protein': total_protein,
        'abdomo_protein': abdomo_protein, 'nasogastric_reflux_ph': nasogastric_reflux_ph,
        'surgery': surgery, 'age': age, 'pain': pain, 'surgical_lesion': surgical_lesion,
        'temp_of_extremities': temp_of_extremities, 'mucous_membrane': mucous_membrane,
        'abdomo_appearance': abdomo_appearance, 'lesion_1': 0, 'lesion_2': 0, 'lesion_3': 0
    }
    
    df = pd.DataFrame([input_data])

    # 2. Özellik Mühendisliği
    df['shock_index'] = df['pulse'] / df['rectal_temp']
    df['protein_gradient'] = df['total_protein'] - df['abdomo_protein']
    df['dehydration_index'] = df['packed_cell_volume'] / (df['total_protein'] + 1e-5)

    # 3. Scaling & Imputer
    for col in expected_num_cols:
        if col not in df.columns: df[col] = 0
    df[expected_num_cols] = scaler.transform(df[expected_num_cols])

    # 4. Dummies & Align
    df = pd.get_dummies(df)
    df = df.reindex(columns=dummy_columns, fill_value=0)

    # 5. Sonuç
    res = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    
    mapping = {0: 'ÖLDÜ (Died)', 1: 'ÖTENAZİ (Euthanized)', 2: 'YAŞADI (Lived)'}
    
    st.divider()
    color = "green" if res == 2 else "red"
    st.markdown(f"### Tahmin: :{color}[{mapping[res]}]")
    
    # Olasılıkları göster (Neden yaşadı dediğini anlarız)
    st.write("#### Olasılık Dağılımı:")
    st.progress(float(prob[2]), text=f"Yaşama Olasılığı: %{prob[2]*100:.1f}")
    st.progress(float(prob[0]), text=f"Ölüm Olasılığı: %{prob[0]*100:.1f}")
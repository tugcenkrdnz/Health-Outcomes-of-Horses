# Kaggle S3E22 - 🐎 At Sağlığı Tahmin Sistemi (Horse Health Prediction)

Bu proje, Kaggle "Health Outcomes of Horses" veri seti kullanılarak geliştirilmiş, atların klinik muayene bulgularına dayanarak sağlık durumlarını (Yaşadı, Öldü, Ötenazi) tahmin eden bir makine öğrenmesi uygulamasıdır.

## 📊 Başarı Oranı
* **Kaggle Public Score:** 0.80487
* **Model:** Voting Classifier (XGBoost + CatBoost)

## 🛠️ Özellikler
- **Akıllı Boşluk Doldurma:** Eksik veriler KNNImputer ile tamamlanmıştır.
- **Ölçeklendirme:** RobustScaler ile aykırı değerlere karşı dayanıklı hale getirilmiştir.
- **Özellik Mühendisliği:** Şok İndeksi, Protein Gradyanı gibi klinik parametreler türetilmiştir.

## 🚀 Kurulum ve Çalıştırma

1. Projeyi bilgisayarınıza indirin veya klonlayın.
2. Gerekli kütüphaneleri yükleyin:
  pip install -r requirements.txt
3. Uygulamayı başlatın:
   streamlit run app.py

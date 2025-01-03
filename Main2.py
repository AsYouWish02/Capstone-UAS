import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Fungsi utama
def main():
    st.title("Water Potability Prediction")
    st.write("Aplikasi menganalisis potabilitas air menggunakan algoritma klasifikasi.")

    # Sidebar
    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox("Pilih Halaman", ["Upload Dataset", "Visualisasi Data", "Analisis dan Evaluasi", "Prediksi Kelayakan Air"])

    # Halaman Upload Dataset
    if page == "Upload Dataset":
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader("Unggah file dataset (CSV)", type=["csv"])
        if uploaded_file:
            dataset = pd.read_csv(uploaded_file)

            # Tampilkan dataset
            st.write("### Dataset (5 Baris Pertama)")
            st.write(dataset.head())

            st.write("### Informasi Dataset")
            st.write("#### Nilai Unik")
            st.write(dataset.nunique())
            st.write("#### Jumlah Baris")
            st.write(f"Jumlah Baris: {dataset.shape[0]}")
            st.write("#### Tipe Data")
            st.write(dataset.dtypes)
            
    # Halaman Visualisasi Data
    elif page == "Visualisasi Data":
        st.header("Visualisasi Data")
        uploaded_file = st.sidebar.file_uploader("Unggah file dataset (CSV)", type=["csv"])
        if uploaded_file:
            dataset = pd.read_csv(uploaded_file)
            dataset = dataset.fillna(dataset.mean())

            # Outliers
            for col in dataset.columns:
                if dataset[col].dtype in ['int64', 'float64']:  # Hanya untuk kolom numerik
                    q1 = dataset[col].quantile(0.25)
                    q3 = dataset[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - (1.5 * iqr)
                    upper_bound = q3 + (1.5 * iqr)

                    # Ganti outliers dengan nilai rata-rata
                    mean_value = dataset[col].mean()
                    dataset[col] = dataset[col].apply(lambda x: mean_value if x < lower_bound or x > upper_bound else x)
    
            # Visualisasi Distribusi Potability Sebelum Resampling
            st.write("### Distribusi Potability Sebelum Resampling")
            plt.figure(figsize=(6, 4))
            sns.countplot(data=dataset, x='Potability')
            plt.title("Distribusi Data Kualitas Air Sebelum Resampling")
            plt.xlabel("Potability")
            plt.ylabel("Jumlah")
            st.pyplot(plt)
    
            # Resampling
            stratified_sample = dataset.groupby('Potability', group_keys=False).apply(lambda x: x.sample(2, random_state=1).reset_index(drop=True))
    
            # Visualisasi Distribusi Potability Setelah Resampling
            st.write("### Distribusi Potability Setelah Resampling")
            plt.figure(figsize=(6, 4))
            sns.countplot(data=stratified_sample, x='Potability')
            plt.title("Distribusi Data Kualitas Air Setelah Resampling")
            plt.xlabel("Potability")
            plt.ylabel("Jumlah")
            st.pyplot(plt)

            # Visualisasi korelasi
            st.subheader("Korelasi Heatmap")
            plt.figure(figsize=(10, 8))
            sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Korelasi Heatmap")
            st.pyplot(plt)
            
            #Visualisasi Histogram untuk setiap fitur
            st.subheader("Distribusi Histogram Plot")
            features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

            plt.figure(figsize=(15, 12))
            for i, feature in enumerate(features):
                plt.subplot(3, 3, i + 1)  # 3x3 grid
                sns.histplot(dataset[feature], kde=True, color='skyblue')
                plt.title(f"Distribusi {feature}")
                plt.xlabel(feature)
                plt.ylabel("Frekuensi")
                plt.tight_layout()  # Adjust layout to prevent overlap
            
            st.pyplot(plt)
    # Halaman Analisis dan Evaluasi
    elif page == "Analisis dan Evaluasi":
        st.header("Analisis dan Evaluasi")
        uploaded_file = st.sidebar.file_uploader("Unggah file dataset (CSV)", type=["csv"])
        if uploaded_file:
            dataset = pd.read_csv(uploaded_file)

            # Preprocessing
            st.subheader("Preprocessing Data")
            st.write("Mengganti nilai yang hilang dengan mean...")
            dataset = dataset.fillna(dataset.mean())

            # Fitur dan target
            fitur = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity','Organic_carbon', 'Trihalomethanes', 'Turbidity']
            target = 'Potability'

            X = dataset[fitur]
            y = dataset[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Normalisasi
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Pemilihan algoritma
            st.subheader("Pilih Algoritma")
            algorithms = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest': RandomForestClassifier(),
                'SVM': SVC(),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier()
            }
            selected_algo = st.selectbox("Pilih Algoritma", list(algorithms.keys()))
            model = algorithms[selected_algo]

            # Evaluasi sebelum normalisasi
            st.subheader("Evaluasi Sebelum Normalisasi")
            model.fit(X_train, y_train)
            y_pred_before = model.predict(X_test)
            accuracy_before = accuracy_score(y_test, y_pred_before)
            cm_before = confusion_matrix(y_test, y_pred_before)

            st.write(f"**Akurasi Sebelum Normalisasi: {accuracy_before:.4f}**")
            st.write("**Confusion Matrix Sebelum Normalisasi:**")
            st.write(cm_before)

            # Plot confusion matrix sebelum normalisasi
            fig, ax = plt.subplots()
            sns.heatmap(cm_before, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"Confusion Matrix - {selected_algo} (Sebelum Normalisasi)")
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig)

            # Evaluasi setelah normalisasi
            st.subheader("Evaluasi Setelah Normalisasi")
            model.fit(X_train_scaled, y_train)
            y_pred_after = model.predict(X_test_scaled)
            accuracy_after = accuracy_score(y_test, y_pred_after)
            cm_after = confusion_matrix(y_test, y_pred_after)

            st.write(f"**Akurasi Setelah Normalisasi: {accuracy_after:.4f}**")
            st.write("**Confusion Matrix Setelah Normalisasi:**")
            st.write(cm_after)

            # Plot confusion matrix setelah normalisasi
            fig, ax = plt.subplots()
            sns.heatmap(cm_after, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"Confusion Matrix - {selected_algo} (Setelah Normalisasi)")
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig)

    # Halaman Prediksi Kelayakan Air
    elif page == "Prediksi Kelayakan Air":
        st.header("Prediksi Kelayakan Air")
        st.write("Masukkan nilai dari 9 fitur untuk memprediksi apakah air layak diminum atau tidak.")
        
        # Upload dataset untuk pelatihan model
        uploaded_file = st.sidebar.file_uploader("Unggah file dataset (CSV)", type=["csv"])
        if uploaded_file:
            dataset = pd.read_csv(uploaded_file)
        
            # Menentukan fitur dan target
            features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity','Organic_carbon', 'Trihalomethanes', 'Turbidity']
            target = 'Potability'
        
            # Memisahkan fitur dan target
            X = dataset[features]
            y = dataset[target]
        
            # Mengatasi missing values dengan imputasi rata-rata
            X = X.fillna(X.mean())
        
            # Membagi data menjadi 80% pelatihan dan 20% pengujian
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
            # Normalisasi data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        
            # Pilih model dari sidebar
            model_choice = st.sidebar.selectbox(
                "Pilih Model:",
                ["Logistic Regression", "Random Forest", "SVM", "KNN", "Decision Tree"]
            )
        
            # Model yang dipilih
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier()
            elif model_choice == "SVM":
                model = SVC()
            elif model_choice == "KNN":
                model = KNeighborsClassifier()
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier()

            # Pelatihan model
            model.fit(X_train_scaled, y_train)
        
            # Evaluasi model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_percen = accuracy * 100
            st.write(f"Tingkat Akurasi Model: **{accuracy_percen:.2f}%**")
        
            # Form input untuk prediksi
            st.subheader("Form Input Prediksi")
            ph = st.number_input("pH (0-14)", min_value=0.0, max_value=14.0, step=0.01)
            hardness = st.number_input("Hardness", min_value=0.0, step=0.01)
            solids = st.number_input("Solids", min_value=0.0, step=0.01)
            chloramines = st.number_input("Chloramines", min_value=0.0, step=0.01)
            sulfate = st.number_input("Sulfate", min_value=0.0, step=0.01)
            conductivity = st.number_input("Conductivity", min_value=0.0, step=0.01)
            organic_carbon = st.number_input("Organic Carbon", min_value=0.0, step=0.01)
            trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, step=0.01)
            turbidity = st.number_input("Turbidity", min_value=0.0, step=0.01)
        
            # Button untuk prediksi
            if st.button("Prediksi"):
                # Data input untuk prediksi
                input_data = [[ph, hardness, solids, chloramines, sulfate, conductivity,organic_carbon, trihalomethanes, turbidity]]
                input_data_scaled = scaler.transform(input_data)
                prediction = model.predict(input_data_scaled)
                result = "Layak Minum" if prediction[0] == 1 else "Tidak Layak Minum"
                st.success(f"Hasil Prediksi: **{result}**")

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="Klasifikasi Kondisi Hewan",
    page_icon="🐾",
    layout="wide"
)

st.title("🐾 Sistem Klasifikasi Kondisi Hewan")
st.write(
    "Aplikasi klasifikasi kondisi hewan menggunakan "
    "KNN, Decision Tree, dan Naive Bayes."
)

@st.cache_data
def load_data():
    return pd.read_csv("test_mod.csv")

try:
    df = load_data()
except Exception as e:
    st.error("Dataset test_mod.csv tidak dapat dibaca.")
    st.code(str(e))
    st.stop()

required_columns = [
    "color_type", "length(m)", "height(cm)",
    "X1", "X2", "condition"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error(f"Kolom berikut tidak ditemukan: {missing}")
    st.write("Kolom yang tersedia:", df.columns.tolist())
    st.stop()

df = df[required_columns].copy()
df["condition"] = pd.to_numeric(df["condition"], errors="coerce")
df = df.dropna(subset=["condition"])
df["condition"] = df["condition"].astype(int)

X = df[["color_type", "length(m)", "height(cm)", "X1", "X2"]]
y = df["condition"]

if y.nunique() < 2 or y.value_counts().min() < 2:
    st.error(
        "Dataset harus memiliki minimal dua kelas kondisi, "
        "dengan setidaknya dua data pada setiap kelas."
    )
    st.stop()

categorical_features = ["color_type"]
numeric_features = ["length(m)", "height(cm)", "X1", "X2"]


def make_preprocessor(scale_numeric=False):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(numeric_steps)

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "KNN": Pipeline([
        ("prep", make_preprocessor(scale_numeric=True)),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Decision Tree": Pipeline([
        ("prep", make_preprocessor()),
        ("model", DecisionTreeClassifier(
            criterion="entropy",
            random_state=42
        ))
    ]),
    "Naive Bayes": Pipeline([
        ("prep", make_preprocessor()),
        ("model", GaussianNB())
    ])
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results[name] = {
        "model": model,
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True
        )
    }

st.subheader("1. Informasi Dataset")

col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Data", len(df))
col2.metric("Jumlah Fitur", len(X.columns))
col3.metric("Jumlah Kelas", y.nunique())

with st.expander("Lihat data dataset"):
    st.dataframe(df, use_container_width=True)

st.subheader("2. Evaluasi Model")

comparison = pd.DataFrame([
    {
        "Algoritma": name,
        "Akurasi (%)": round(result["accuracy"] * 100, 2)
    }
    for name, result in results.items()
]).sort_values("Akurasi (%)", ascending=False)

st.dataframe(comparison, use_container_width=True)

selected_model = st.selectbox(
    "Pilih algoritma untuk melihat hasil evaluasi",
    list(models.keys())
)

st.write(f"**Classification report: {selected_model}**")
report = results[selected_model]["report"]

report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.round(3), use_container_width=True)

st.subheader("3. Prediksi Kondisi Hewan")

with st.form("prediction_form"):
    color_options = sorted(
        X["color_type"].dropna().astype(str).unique().tolist()
    )

    if not color_options:
        color_options = ["Tidak diketahui"]

    color = st.selectbox("Warna hewan", color_options)

    col1, col2 = st.columns(2)

    with col1:
        length = st.number_input(
            "Panjang (m)",
            min_value=0.0,
            value=1.0,
            step=0.1
        )
        height = st.number_input(
            "Tinggi (cm)",
            min_value=0.0,
            value=10.0,
            step=0.5
        )

    with col2:
        x1 = st.number_input("Nilai X1", value=0.0)
        x2 = st.number_input("Nilai X2", value=0.0)

    algorithm = st.selectbox(
        "Algoritma prediksi",
        list(models.keys()),
        key="prediction_algorithm"
    )

    submitted = st.form_submit_button("Prediksi")

if submitted:
    new_data = pd.DataFrame([{
        "color_type": color,
        "length(m)": length,
        "height(cm)": height,
        "X1": x1,
        "X2": x2
    }])

    prediction = results[algorithm]["model"].predict(new_data)[0]

    st.success(f"Hasil prediksi kondisi hewan: {prediction}")

    st.caption(
        "Hasil berupa label kelas pada kolom condition "
        "di dataset, bukan diagnosis medis hewan."
    )

st.sidebar.header("Tentang Aplikasi")
st.sidebar.write("Metode klasifikasi:")
st.sidebar.write("- K-Nearest Neighbors (KNN)")
st.sidebar.write("- Decision Tree dengan entropy")
st.sidebar.write("- Gaussian Naive Bayes")
st.sidebar.write("Pembagian data latih dan uji: 80% : 20%.")

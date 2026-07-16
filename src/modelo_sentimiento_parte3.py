"""
================================================================================
TFM - Valoración financiera de Apple Inc. con IA
Parte 3. APRENDIZAJE AUTOMÁTICO
Clasificación supervisada del sentimiento (positivo / negativo / neutral)
--------------------------------------------------------------------------------
Script reproducible. Requisitos:
    pip install pandas numpy scikit-learn nltk openpyxl joblib
    (recursos NLTK: stopwords)
Entrada : resultado_actual.xlsx  (hoja 'Hoja1' con columnas Tweet, Fecha,
          Fuente, Sentimiento, *_Score) generado en la fase de recopilación.
Salidas : modelo_sentimiento.joblib  -> modelo entrenado (para MLOps / parte 5)
          predicciones_test.csv       -> test + predicción + probabilidades
================================================================================
"""
import re, warnings
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)

SEED = 42
CLASSES = ["negative", "neutral", "positive"]
ENTRADA = "resultado_actual.xlsx"

# ----------------------------------------------------------------------------
# 1. CARGA Y DEPURACIÓN DE REGISTROS
# ----------------------------------------------------------------------------
df = pd.read_excel(ENTRADA, sheet_name="Hoja1")
df.columns = [c.strip() for c in df.columns]
df["Tweet"] = df["Tweet"].astype(str).str.strip()

df = df[df["Sentimiento"].isin(CLASSES)]      # elimina etiquetas 'Error/overrate' y 'mixed'
df = df[df["Tweet"].str.len() > 15]           # elimina registros basura
df = df.drop_duplicates(subset="Tweet", keep="first")   # elimina duplicados exactos (evita data leakage)
print(f"Registros únicos tras depuración: {len(df)}")

# ----------------------------------------------------------------------------
# 2. PREPROCESAMIENTO DE TEXTO (NLP)
# ----------------------------------------------------------------------------
STOP_EN = set(stopwords.words("english"))
RUIDO = {"apple", "aapl", "iphone", "ipad", "ipod", "itunes", "ios", "mac",
         "macbook", "rt", "amp", "https", "http", "co", "new", "news",
         "says", "report"}
stemmer = SnowballStemmer("english")

def limpiar(texto: str) -> str:
    t = texto.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)   # URLs
    t = re.sub(r"@\w+", " ", t)                # menciones
    t = re.sub(r"#(\w+)", r"\1", t)            # quita '#' conservando la palabra
    t = re.sub(r"[^a-záéíóúñ\s]", " ", t)      # solo letras
    t = re.sub(r"\s+", " ", t).strip()
    toks = [w for w in t.split()
            if w not in STOP_EN and w not in RUIDO and len(w) > 2]
    return " ".join(stemmer.stem(w) for w in toks)

df["texto_limpio"] = df["Tweet"].apply(limpiar)
df = df[df["texto_limpio"].str.len() > 0]

# ----------------------------------------------------------------------------
# 3. PARTICIÓN ESTRATIFICADA 80/20
# ----------------------------------------------------------------------------
X, y = df["texto_limpio"].values, df["Sentimiento"].values
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, df.index.values, test_size=0.20, random_state=SEED, stratify=y)

# ----------------------------------------------------------------------------
# 4. COMPARACIÓN DE MODELOS (validación cruzada 5-fold, F1-macro)
# ----------------------------------------------------------------------------
vect_cmp = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
Xtr_vec = vect_cmp.fit_transform(X_tr)
modelos = {
    "Regresión logística": LogisticRegression(max_iter=2000, class_weight="balanced", C=3.0),
    "SVM lineal":          LinearSVC(class_weight="balanced", C=1.0),
    "Random Forest":       RandomForestClassifier(n_estimators=400,
                                                  class_weight="balanced_subsample", random_state=SEED),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=SEED),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print("\nComparación de modelos (F1-macro CV):")
for nombre, clf in modelos.items():
    sc = cross_val_score(clf, Xtr_vec, y_tr, cv=cv, scoring="f1_macro")
    print(f"  {nombre:22s} {sc.mean():.3f} ± {sc.std():.3f}")

# ----------------------------------------------------------------------------
# 5. MODELO FINAL: Regresión Logística
#    (empatada en CV con SVM; se elige por sus probabilidades nativas y su
#     interpretabilidad, necesarias para las partes 4, 6 y 7)
# ----------------------------------------------------------------------------
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
    ("clf",   LogisticRegression(max_iter=2000, class_weight="balanced", C=3.0)),
])
pipe.fit(X_tr, y_tr)
joblib.dump(pipe, "modelo_sentimiento.joblib")

# ----------------------------------------------------------------------------
# 6. EVALUACIÓN
# ----------------------------------------------------------------------------
for nombre, Xs, ys in [("ENTRENAMIENTO", X_tr, y_tr), ("PRUEBA", X_te, y_te)]:
    pred = pipe.predict(Xs)
    print(f"\n== {nombre} ==  F1-macro = {f1_score(ys, pred, average='macro'):.3f}")
    print(classification_report(ys, pred, labels=CLASSES, zero_division=0))
    print("Matriz de confusión:\n", confusion_matrix(ys, pred, labels=CLASSES))

# ----------------------------------------------------------------------------
# 7. FICHERO DE SALIDA: test + predicción + probabilidades por clase
# ----------------------------------------------------------------------------
proba = pipe.predict_proba(X_te)
clases = list(pipe.classes_)
out = df.loc[idx_te, ["Tweet", "Fecha", "Fuente", "Sentimiento"]].copy()
out.rename(columns={"Sentimiento": "Sentimiento_real"}, inplace=True)
out["Sentimiento_predicho"] = pipe.predict(X_te)
for c in CLASSES:
    out[f"prob_{c}"] = proba[:, clases.index(c)].round(4)
out["prob_predicha"] = proba.max(axis=1).round(4)
out.to_csv("predicciones_test.csv", index=False)
print("\nFichero 'predicciones_test.csv' generado:", out.shape)

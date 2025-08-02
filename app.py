# 🧠 لوحة ذكاء العملاء - Streamlit موحد

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import io

st.set_page_config(page_title="لوحة ذكاء العملاء", layout="wide")
st.title("📊 لوحة ذكاء العملاء - تنظيف، تحليل، تدريب وتنبؤ")

df = None
model = None

tab1, tab2, tab3, tab4 = st.tabs(["🧼 تنظيف البيانات", "📈 تحليل السمات", "🤖 تدريب النموذج", "🔍 تجربة عميل جديد"])

# -----------------------------
# 🧼 تنظيف البيانات
# -----------------------------
with tab1:
    st.subheader("🧼 تحميل وتنظيف ملف Excel")
    uploaded_file = st.file_uploader("📤 ارفع ملف Excel", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("✅ تم تحميل البيانات:")
        st.dataframe(df.head())

        df.columns = df.columns.str.strip()
        df = df.dropna()
        df = df.drop_duplicates()

        st.success("✅ تم تنظيف البيانات (إزالة القيم المفقودة والتكرار)")

        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        st.download_button("📥 تحميل البيانات النظيفة", data=buffer, file_name="cleaned_data.xlsx")

# -----------------------------
# 📈 تحليل السمات مقابل الهدف
# -----------------------------
with tab2:
    st.subheader("📊 تحليل العلاقة بين السمات والهدف")
    if df is not None:
        target_column = None
        for col in df.columns:
            if str(col).strip() == "سدد":
                target_column = col
                break

        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            selected_feature = st.selectbox("اختر سمة لعرض العلاقة", X.columns)

            boxplot_df = pd.DataFrame({"feature": X[selected_feature], "target": y}).dropna()

            st.subheader("📊 القيم الإحصائية")
            stats = boxplot_df.groupby("target")["feature"].describe()
            st.dataframe(stats)

            fig, ax = plt.subplots()
            sns.boxplot(x=boxplot_df["target"], y=boxplot_df["feature"], ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ لا يوجد عمود باسم 'سدد'")
    else:
        st.info("📂 يرجى رفع وتنظيف ملف البيانات أولاً")

# -----------------------------
# 🤖 تدريب نموذج الذكاء الاصطناعي
# -----------------------------
with tab3:
    st.subheader("🤖 تدريب نموذج سداد العملاء")
    if df is not None and target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        test_size = st.slider("🔀 نسبة بيانات الاختبار", 0.1, 0.5, 0.3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        train_data = pd.concat([X_train, y_train], axis=1).dropna()
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success(f"✅ دقة النموذج: {accuracy_score(y_test, y_pred):.2f}")
        st.text("📋 تقرير التصنيف:")
        st.text(classification_report(y_test, y_pred))

        joblib.dump(model, "model_sadad.pkl")
        st.download_button("💾 تحميل النموذج المدرب", data=open("model_sadad.pkl", "rb"), file_name="model_sadad.pkl")

    else:
        st.info("📂 يرجى تنظيف وتحليل البيانات أولاً")

# -----------------------------
# 🔍 تجربة تنبؤ عميل جديد
# -----------------------------
with tab4:
    st.subheader("🔍 تجربة نموذج التنبؤ على عميل جديد")
    try:
        model = joblib.load("model_sadad.pkl")
    except:
        st.warning("❌ لم يتم العثور على نموذج مدرب")
        st.stop()

    st.markdown("أدخل بيانات العميل الجديد:")
    c1, c2 = st.columns(2)
    with c1:
        r = st.number_input("الرصيد", min_value=0.0)
        age = st.number_input("عمر المديونية", min_value=0.0)
        sales = st.number_input("المبيعات الشهرية", min_value=0.0)
    with c2:
        avg = st.number_input("متوسط السداد التراكمي", min_value=0.0)
        sd = st.number_input("السداد الشهري للعميل", min_value=0.0)
        returns = st.number_input("نسبة المرتجع من المباع", min_value=0.0, format="%.4f")

    if st.button("🔍 تنبؤ الآن"):
        new_data = pd.DataFrame([[r, age, sales, avg, sd, returns]],
                                columns=["الرصيد", "عمر المديونية", "المبيعات الشهرية",
                                         "متوسط السداد التراكمي", "السداد الشهري للعميل",
                                         "نسبة المرتجع من المباع"])
        pred = model.predict(new_data)[0]
        if pred == 1:
            st.success("✅ العميل مرشح للسداد")
        else:
            st.error("❌ العميل معرض لعدم السداد")

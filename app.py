from io import BytesIO

import pandas as pd
import streamlit as st
from src.gigachat_client import ask_gigachat


st.set_page_config(
    page_title="LLM Data Analytics Agent",
    page_icon="📊",
    layout="wide",
)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(BytesIO(uploaded_file.read()))

    raise ValueError("Поддерживаются только CSV и Excel-файлы.")


def get_columns_info(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing": [int(df[column].isna().sum()) for column in df.columns],
            "missing_percent": [
                round(float(df[column].isna().mean() * 100), 2) for column in df.columns
            ],
            "unique": [int(df[column].nunique(dropna=True)) for column in df.columns],
        }
    )


def show_dataset_preview(df: pd.DataFrame) -> None:
    st.subheader("Предпросмотр датасета")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Строк", df.shape[0])
    col2.metric("Колонок", df.shape[1])
    col3.metric("Пропусков", int(df.isna().sum().sum()))
    col4.metric("Дублей", int(df.duplicated().sum()))

    st.write("Первые 20 строк:")
    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Структура колонок"):
        st.dataframe(get_columns_info(df), use_container_width=True)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if numeric_columns:
        with st.expander("Базовая статистика по числовым колонкам"):
            st.dataframe(df[numeric_columns].describe().T, use_container_width=True)


def main() -> None:
    st.title("LLM Data Analytics Agent")

    st.write(
        "Веб-интерфейс для анализа CSV/Excel-датасетов. "
        "На следующем этапе приложение будет использовать LLM через GigaChat API, "
        "генерировать Python-код анализа, выполнять его и собирать итоговый отчёт."
    )

    with st.sidebar:
        st.header("LLM API")

        if st.button("Проверить GigaChat"):
            try:
                with st.spinner("Проверяю подключение..."):
                    answer = ask_gigachat(
                        "Ответь одним коротким предложением: подключение к GigaChat работает."
                    )

                st.success("GigaChat подключён.")
                st.write(answer)
            except Exception as error:
                st.error(f"Ошибка подключения: {error}")

    uploaded_file = st.file_uploader(
        "Загрузите CSV или Excel-файл",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is None:
        st.info("Загрузите файл, чтобы начать работу.")
        return

    try:
        df = read_uploaded_file(uploaded_file)
    except Exception as error:
        st.error(f"Не удалось прочитать файл: {error}")
        return

    show_dataset_preview(df)

    st.subheader("Инструкция для будущего LLM-анализа")

    user_instruction = st.text_area(
        "Что нужно будет проанализировать?",
        value=(
            "Проанализируй датасет: найди ключевые метрики, связи между признаками, "
            "аномалии и построй несколько графиков."
        ),
        height=120,
    )

    st.info(
        "На этом этапе инструкция пока только отображается. "
        "Подключение LLM-агента добавим следующим коммитом."
    )

    with st.expander("Текущая инструкция"):
        st.write(user_instruction)


if __name__ == "__main__":
    main()
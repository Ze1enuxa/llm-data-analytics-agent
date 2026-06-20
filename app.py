from io import BytesIO
from pathlib import Path
import time
from urllib import response

import pandas as pd
import streamlit as st

from src.agent import AnalyticsAgent
from src.gigachat_client import ask_gigachat
from src.prompt_guard import check_user_instruction


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

def show_tool_tables(response: dict) -> None:
    result = response.get("result", {})
    tool_results = result.get("tool_results", [])

    if not tool_results:
        return

    st.subheader("Таблицы из результатов инструментов")

    for item in tool_results:
        tool = item.get("tool")
        status = item.get("status")

        if status != "ok":
            continue

        if tool == "correlation":
            table = item.get("table", [])

            if table:
                st.write("Корреляционная матрица")
                st.dataframe(pd.DataFrame(table), use_container_width=True)

        if tool == "top_values":
            table = item.get("result", [])
            sort_by = item.get("sort_by", "выбранному признаку")

            if table:
                st.write(f"Топ-{len(table)} строк по `{sort_by}`")
                st.dataframe(pd.DataFrame(table), use_container_width=True)

        if tool == "numeric_describe":
            table = item.get("result", [])

            if table:
                st.write("Описательная статистика")
                st.dataframe(pd.DataFrame(table), use_container_width=True)

        if tool == "groupby_mean":
            table = item.get("result", [])
            group_by = item.get("group_by")
            value = item.get("value")

            if table:
                st.write(f"Среднее `{value}` по `{group_by}`")
                st.dataframe(pd.DataFrame(table), use_container_width=True)

def show_analysis_result(response: dict) -> None:
    st.subheader("Итоговый отчёт")
    warning = response.get("warning")

    if warning:
        st.warning(warning)
    st.markdown(response["report"])

    charts = response.get("charts", [])
    if charts:
        st.subheader("Графики")

        for chart_path in charts:
            path = Path(chart_path)

            if path.exists():
                st.image(str(path))

    show_tool_tables(response)

    with st.expander("Результат выполнения кода"):
        st.json(response.get("result", {}))

    with st.expander("План анализа и результаты инструментов"):
        st.code(response.get("code", ""), language="json")

    stdout = response.get("stdout")

    if stdout:
        with st.expander("stdout"):
            st.text(stdout)


def show_sidebar() -> None:
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

        st.divider()

        st.caption(
            "Ключ GigaChat хранится в .env локально или в Streamlit Secrets при деплое."
        )


def main() -> None:
    st.title("LLM Data Analytics Agent")

    st.write(
        "Веб-интерфейс для агентного анализа CSV/Excel-датасетов. "
        "LLM генерирует Python-код анализа, приложение выполняет его в ограниченном окружении "
        "и формирует итоговый отчёт."
    )

    show_sidebar()

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

    st.subheader("Инструкция для LLM-анализа")

    user_instruction = st.text_area(
        "Что нужно проанализировать?",
        value=(
            "Проанализируй датасет: найди ключевые метрики, связи между признаками, "
            "аномалии и построй несколько графиков."
        ),
        height=120,
    )

    if st.button("Запустить LLM-анализ", type="primary"):
        is_safe, message = check_user_instruction(user_instruction)

        if not is_safe:
            st.error(message)
            return

        run_id = int(time.time())
        output_dir = Path("outputs") / f"run_{run_id}"

        try:
            with st.spinner("LLM создает план анализа, выполняет анализ и собирает отчёт..."):
                agent = AnalyticsAgent()
                response = agent.run(df, user_instruction, output_dir)

            st.success("Анализ завершён.")
            show_analysis_result(response)

        except Exception as error:
            st.error(f"Ошибка при выполнении анализа: {error}")
            st.warning(
                "Проверьте текст инструкции, структуру датасета и доступность GigaChat API."
            )


if __name__ == "__main__":
    main()
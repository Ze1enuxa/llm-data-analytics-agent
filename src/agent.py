import json
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from json_repair import repair_json

from src.gigachat_client import ask_gigachat


ALLOWED_TOOLS = {
    "dataset_profile",
    "missing_values",
    "numeric_describe",
    "correlation",
    "top_values",
    "histogram",
    "scatter",
    "groupby_mean",
}


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude="number").columns.tolist()


def get_identifier_columns(df: pd.DataFrame) -> list[str]:
    id_columns = []

    for column in df.columns:
        lowered = column.lower()

        if lowered == "id" or lowered.endswith("_id"):
            id_columns.append(column)

    return id_columns


def get_mentioned_columns(df: pd.DataFrame, user_instruction: str) -> list[str]:
    lowered = user_instruction.lower()
    mentioned = []

    for column in df.columns:
        if column.lower() in lowered:
            mentioned.append(column)

    return mentioned


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [make_jsonable(item) for item in value]

    if isinstance(value, tuple):
        return [make_jsonable(item) for item in value]

    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.DataFrame):
        return value.head(50).to_dict(orient="records")

    if isinstance(value, pd.Series):
        return value.head(50).to_dict()

    return value


def dataset_context(df: pd.DataFrame) -> str:
    columns_info = []

    for column in df.columns:
        columns_info.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing": int(df[column].isna().sum()),
                "unique": int(df[column].nunique(dropna=True)),
            }
        )

    context = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": get_numeric_columns(df),
        "categorical_columns": get_categorical_columns(df),
        "identifier_columns": get_identifier_columns(df),
        "columns_info": columns_info,
        "sample_rows": df.head(8).to_dict(orient="records"),
    }

    return json.dumps(context, ensure_ascii=False, indent=2, default=str)


def build_plan_prompt(df: pd.DataFrame, user_instruction: str) -> str:
    return f"""
Ты — LLM-аналитик данных. Твоя задача — составить план анализа загруженного табличного датасета.

Инструкция пользователя:
{user_instruction}

Контекст датасета:
{dataset_context(df)}

Ты НЕ пишешь Python-код. Ты выбираешь аналитические инструменты, которые должно выполнить приложение.

Доступные инструменты:
1. dataset_profile — структура датасета: строки, колонки, типы, пропуски, дубли.
2. missing_values — анализ пропусков.
3. numeric_describe — описательная статистика числовых колонок.
4. correlation — корреляция между выбранными числовыми колонками.
5. top_values — топ-N строк по выбранной колонке.
6. histogram — распределение числовой колонки.
7. scatter — график связи двух числовых колонок.
8. groupby_mean — среднее значение числовой колонки по категориям.

Верни только валидный JSON без markdown и без пояснений.

Структура ответа:
{{
  "analysis_goal": "краткая цель анализа",
  "analysis_steps": [
    {{
      "tool": "dataset_profile"
    }},
    {{
      "tool": "correlation",
      "columns": ["col1", "col2"]
    }},
    {{
      "tool": "histogram",
      "column": "col1"
    }},
    {{
      "tool": "scatter",
      "x": "col1",
      "y": "col2"
    }},
    {{
      "tool": "top_values",
      "sort_by": "col1",
      "n": 10
    }}
  ],
  "report_focus": [
    "на что обратить внимание в отчёте"
  ]
}}

Правила:
- Используй только реальные названия колонок из датасета.
- Для correlation, histogram и scatter выбирай только числовые колонки.
- Если пользователь явно указал названия колонок, обязательно включи их в план.
- Если пользователь просит распределение, добавь histogram.
- Если пользователь просит связь между признаками, добавь correlation и scatter.
- Если пользователь просит топ, добавь top_values.
- Не придумывай новых колонок.
- Не добавляй инструменты, которых нет в списке.
"""


def parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(repair_json(text))

    if not isinstance(data, dict):
        raise ValueError("LLM returned non-object JSON plan.")

    return data


def build_local_plan(df: pd.DataFrame, user_instruction: str) -> dict[str, Any]:
    numeric_columns = get_numeric_columns(df)
    mentioned_columns = get_mentioned_columns(df, user_instruction)
    mentioned_numeric = [column for column in mentioned_columns if column in numeric_columns]

    if not mentioned_numeric:
        mentioned_numeric = numeric_columns[:4]

    target_column = mentioned_numeric[0] if mentioned_numeric else None

    for preferred in [
        "target",
        "label",
        "score",
        "rating",
        "price",
        "sales",
        "revenue",
        "profit",
        "gpa",
        "grade",
        "exam_score",
        "final_score",
        "skill",
    ]:
        for column in mentioned_numeric:
            if column.lower() == preferred:
                target_column = column
                break

        if target_column == preferred:
            break

    steps = [{"tool": "dataset_profile"}]

    if len(mentioned_numeric) >= 2:
        steps.append({"tool": "correlation", "columns": mentioned_numeric})

    if target_column:
        steps.append({"tool": "histogram", "column": target_column})
        steps.append({"tool": "top_values", "sort_by": target_column, "n": 10})

    if target_column and len(mentioned_numeric) >= 2:
        for column in mentioned_numeric:
            if column != target_column:
                steps.append({"tool": "scatter", "x": target_column, "y": column})

    return {
        "analysis_goal": "Локальный план анализа по инструкции пользователя.",
        "analysis_steps": steps,
        "report_focus": [
            "Описать структуру датасета.",
            "Показать связи между выбранными числовыми признаками.",
            "Не делать причинно-следственных выводов по корреляции.",
        ],
    }


def normalize_plan(plan: dict[str, Any], df: pd.DataFrame, user_instruction: str) -> dict[str, Any]:
    numeric_columns = set(get_numeric_columns(df))
    all_columns = set(df.columns)

    steps = plan.get("analysis_steps")

    if not isinstance(steps, list):
        return build_local_plan(df, user_instruction)

    normalized_steps = []

    for step in steps:
        if not isinstance(step, dict):
            continue

        tool = step.get("tool")

        if tool not in ALLOWED_TOOLS:
            continue

        if tool in {"dataset_profile", "missing_values", "numeric_describe"}:
            normalized_steps.append({"tool": tool})
            continue

        if tool == "correlation":
            columns = step.get("columns", [])
            columns = [column for column in columns if column in numeric_columns]

            if len(columns) >= 2:
                normalized_steps.append({"tool": "correlation", "columns": columns})

            continue

        if tool == "histogram":
            column = step.get("column")

            if column in numeric_columns:
                normalized_steps.append({"tool": "histogram", "column": column})

            continue

        if tool == "scatter":
            x = step.get("x")
            y = step.get("y")

            if x in numeric_columns and y in numeric_columns and x != y:
                normalized_steps.append({"tool": "scatter", "x": x, "y": y})

            continue

        if tool == "top_values":
            sort_by = step.get("sort_by")
            n = step.get("n", 10)

            if sort_by in all_columns:
                try:
                    n = int(n)
                except (TypeError, ValueError):
                    n = 10

                normalized_steps.append(
                    {
                        "tool": "top_values",
                        "sort_by": sort_by,
                        "n": max(1, min(n, 30)),
                    }
                )

            continue

        if tool == "groupby_mean":
            group_by = step.get("group_by")
            value = step.get("value")

            if group_by in all_columns and value in numeric_columns:
                normalized_steps.append(
                    {
                        "tool": "groupby_mean",
                        "group_by": group_by,
                        "value": value,
                    }
                )

            continue

    if not normalized_steps:
        return build_local_plan(df, user_instruction)

    if not any(step["tool"] == "dataset_profile" for step in normalized_steps):
        normalized_steps.insert(0, {"tool": "dataset_profile"})

    return {
        "analysis_goal": str(plan.get("analysis_goal", "Анализ загруженного датасета.")),
        "analysis_steps": normalized_steps,
        "report_focus": plan.get("report_focus", []),
    }


def execute_dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    columns_info = []

    for column in df.columns:
        columns_info.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing": int(df[column].isna().sum()),
                "unique": int(df[column].nunique(dropna=True)),
            }
        )

    return {
        "tool": "dataset_profile",
        "status": "ok",
        "result": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_values": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "numeric_columns": get_numeric_columns(df),
            "categorical_columns": get_categorical_columns(df),
            "identifier_columns": get_identifier_columns(df),
            "columns_info": columns_info,
        },
    }


def execute_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    missing_table = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        if missing_count > 0:
            missing_table.append(
                {
                    "column": column,
                    "missing": missing_count,
                    "missing_percent": round(float(df[column].isna().mean() * 100), 2),
                }
            )

    return {
        "tool": "missing_values",
        "status": "ok",
        "result": {
            "total_missing": int(df.isna().sum().sum()),
            "columns_with_missing": missing_table,
        },
    }


def execute_numeric_describe(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = get_numeric_columns(df)

    if not numeric_columns:
        return {
            "tool": "numeric_describe",
            "status": "skipped",
            "reason": "No numeric columns found.",
        }

    describe_table = df[numeric_columns].describe().T.reset_index()
    describe_table = describe_table.rename(columns={"index": "column"})

    return {
        "tool": "numeric_describe",
        "status": "ok",
        "result": describe_table.round(3).to_dict(orient="records"),
    }


def execute_correlation(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    corr = df[columns].corr().round(3)

    return {
        "tool": "correlation",
        "status": "ok",
        "columns": columns,
        "result": corr.to_dict(),
        "table": corr.reset_index().to_dict(orient="records"),
    }


def execute_top_values(df: pd.DataFrame, sort_by: str, n: int) -> dict[str, Any]:
    ascending = False

    top_table = df.sort_values(by=sort_by, ascending=ascending).head(n)

    keep_columns = []

    identifier_columns = get_identifier_columns(df)

    if identifier_columns:
        keep_columns.append(identifier_columns[0])

    keep_columns.append(sort_by)

    keep_columns = list(dict.fromkeys(keep_columns))

    return {
        "tool": "top_values",
        "status": "ok",
        "sort_by": sort_by,
        "n": n,
        "result": top_table[keep_columns].to_dict(orient="records"),
    }


def execute_histogram(df: pd.DataFrame, column: str, output_dir: Path) -> dict[str, Any]:
    fig, ax = plt.subplots()
    df[column].dropna().hist(ax=ax, bins=25)

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")

    chart_path = output_dir / f"histogram_{column}.png"
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "tool": "histogram",
        "status": "ok",
        "column": column,
        "chart_path": str(chart_path),
    }


def execute_scatter(df: pd.DataFrame, x: str, y: str, output_dir: Path) -> dict[str, Any]:
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y], alpha=0.7)

    ax.set_title(f"{x} vs {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    chart_path = output_dir / f"scatter_{x}_vs_{y}.png"
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "tool": "scatter",
        "status": "ok",
        "x": x,
        "y": y,
        "chart_path": str(chart_path),
    }


def execute_groupby_mean(df: pd.DataFrame, group_by: str, value: str) -> dict[str, Any]:
    grouped = (
        df.groupby(group_by, dropna=False)[value]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
        .reset_index()
        .head(30)
    )

    return {
        "tool": "groupby_mean",
        "status": "ok",
        "group_by": group_by,
        "value": value,
        "result": grouped.round(3).to_dict(orient="records"),
    }


def execute_step(df: pd.DataFrame, step: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    tool = step.get("tool")

    if tool == "dataset_profile":
        return execute_dataset_profile(df)

    if tool == "missing_values":
        return execute_missing_values(df)

    if tool == "numeric_describe":
        return execute_numeric_describe(df)

    if tool == "correlation":
        return execute_correlation(df, step["columns"])

    if tool == "top_values":
        return execute_top_values(df, step["sort_by"], step["n"])

    if tool == "histogram":
        return execute_histogram(df, step["column"], output_dir)

    if tool == "scatter":
        return execute_scatter(df, step["x"], step["y"], output_dir)

    if tool == "groupby_mean":
        return execute_groupby_mean(df, step["group_by"], step["value"])

    return {
        "tool": str(tool),
        "status": "skipped",
        "reason": "Unsupported tool.",
    }


def execute_plan(df: pd.DataFrame, plan: dict[str, Any], output_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    tool_results = []
    charts = []

    for step in plan["analysis_steps"]:
        try:
            result = execute_step(df, step, output_dir)
        except Exception as error:
            result = {
                "tool": step.get("tool"),
                "status": "error",
                "reason": str(error),
            }

        tool_results.append(make_jsonable(result))

        chart_path = result.get("chart_path")

        if chart_path:
            charts.append(chart_path)

    return tool_results, charts


def build_report_prompt(
    user_instruction: str,
    plan: dict[str, Any],
    tool_results: list[dict[str, Any]],
) -> str:
    compact_payload = json.dumps(
        {
            "user_instruction": user_instruction,
            "analysis_plan": plan,
            "tool_results": tool_results,
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )[:16000]

    return f"""
Ты готовишь итоговый отчёт по анализу загруженного датасета.

Ниже находится JSON с инструкцией пользователя, планом анализа и реальными результатами выполненных инструментов:
{compact_payload}

Напиши отчёт на русском языке в Markdown.

Структура:
1. Краткое описание выполненного анализа
2. Основные метрики
3. Найденные закономерности
4. Аномалии или ограничения
5. Практический вывод

Жёсткие правила:
- Используй только числа из tool_results.
- Не придумывай числа, коэффициенты и названия предметной области.
- Называй объект нейтрально: "загруженный датасет", если предметная область не указана явно.
- Если в tool_results нет корреляции, не пиши про корреляцию.
- Если корреляция есть, используй точные значения из tool_results.
- Если abs(correlation) < 0.1, пиши: "связь практически отсутствует".
- Если 0.1 <= abs(correlation) < 0.3, пиши: "слабая связь".
- Если 0.3 <= abs(correlation) < 0.5, пиши: "умеренная связь".
- Если 0.5 <= abs(correlation) < 0.7, пиши: "заметная связь".
- Если abs(correlation) >= 0.7, пиши: "сильная связь".
- Корреляция не доказывает причину. Не используй слова "влияет", "приводит", "обусловлено", если причинность не доказана.
- Если в tool_results есть инструмент top_values со status = "ok", обязательно выведи его результат отдельной Markdown-таблицей.
- Если в tool_results есть инструмент correlation со status = "ok", обязательно выведи корреляции именно из result/table этого инструмента.
- Если в tool_results есть histogram или scatter, кратко перечисли построенные графики.
- Пиши конкретно и без воды.
"""


def build_local_report(
    user_instruction: str,
    plan: dict[str, Any],
    tool_results: list[dict[str, Any]],
) -> str:
    profile = next(
        (item for item in tool_results if item.get("tool") == "dataset_profile" and item.get("status") == "ok"),
        None,
    )

    rows = columns = missing = duplicates = "не указано"

    if profile:
        profile_result = profile.get("result", {})
        rows = profile_result.get("rows", rows)
        columns = profile_result.get("columns", columns)
        missing = profile_result.get("missing_values", missing)
        duplicates = profile_result.get("duplicates", duplicates)

    correlation_blocks = []

    for item in tool_results:
        if item.get("tool") != "correlation" or item.get("status") != "ok":
            continue

        corr_result = item.get("result", {})
        columns_list = item.get("columns", [])

        if not columns_list:
            continue

        target = columns_list[0]
        target_corr = corr_result.get(target, {})

        for column, value in target_corr.items():
            if column == target:
                continue

            try:
                corr_value = float(value)
            except (TypeError, ValueError):
                continue

            abs_value = abs(corr_value)

            if abs_value < 0.1:
                strength = "связь практически отсутствует"
            elif abs_value < 0.3:
                strength = "слабая связь"
            elif abs_value < 0.5:
                strength = "умеренная связь"
            elif abs_value < 0.7:
                strength = "заметная связь"
            else:
                strength = "сильная связь"

            direction = "положительная" if corr_value > 0 else "отрицательная"

            if abs_value < 0.1:
                correlation_blocks.append(
                    f"- `{target}` и `{column}`: корреляция `{corr_value:.3f}`, {strength}."
                )
            else:
                correlation_blocks.append(
                    f"- `{target}` и `{column}`: корреляция `{corr_value:.3f}`, {direction} {strength}."
                )

    correlation_text = "\n".join(correlation_blocks) if correlation_blocks else "Корреляционный анализ не выполнялся."

    return f"""
## 1. Краткое описание выполненного анализа

Выполнен анализ загруженного датасета по инструкции пользователя:

> {user_instruction}

LLM сформировала план анализа, приложение выполнило выбранные инструменты и получило фактические результаты.

## 2. Основные метрики

- Количество строк: `{rows}`
- Количество колонок: `{columns}`
- Количество пропусков: `{missing}`
- Количество дублей: `{duplicates}`

## 3. Найденные закономерности

{correlation_text}

## 4. Аномалии или ограничения

Корреляция показывает статистическую связь между признаками, но не доказывает причинно-следственную зависимость. Выводы относятся только к загруженному датасету.

## 5. Практический вывод

Приложение выполнило агентный анализ: LLM выбрала инструменты анализа, Python выполнил расчёты и построил графики, после чего был сформирован итоговый отчёт.
""".strip()


class AnalyticsAgent:
    def build_plan(self, df: pd.DataFrame, user_instruction: str) -> tuple[dict[str, Any], bool, str]:
        try:
            raw_response = ask_gigachat(build_plan_prompt(df, user_instruction))
            raw_plan = parse_json_object(raw_response)
            plan = normalize_plan(raw_plan, df, user_instruction)

            return plan, True, raw_response

        except Exception as error:
            plan = build_local_plan(df, user_instruction)

            return plan, False, str(error)

    def build_report(
        self,
        user_instruction: str,
        plan: dict[str, Any],
        tool_results: list[dict[str, Any]],
    ) -> str:
        try:
            return ask_gigachat(
                build_report_prompt(user_instruction, plan, tool_results)
            ).strip()

        except Exception:
            return build_local_report(user_instruction, plan, tool_results)

    def run(self, df: pd.DataFrame, user_instruction: str, output_dir: Path) -> dict[str, Any]:
        plan, used_llm_plan, raw_plan = self.build_plan(df, user_instruction)

        tool_results, charts = execute_plan(df, plan, output_dir)
        report = self.build_report(user_instruction, plan, tool_results)

        warning = ""

        if not used_llm_plan:
            warning = (
                "LLM не смогла сформировать план анализа, поэтому использован локальный план. "
                f"Причина: {raw_plan}"
            )

        result = {
            "analysis_plan": plan,
            "tool_results": tool_results,
        }

        return {
            "code": json.dumps(result, ensure_ascii=False, indent=2),
            "report": report,
            "result": result,
            "charts": charts,
            "stdout": "",
            "created_at": int(time.time()),
            "warning": warning,
        }
from __future__ import annotations

from typing import Any

from dataclasses import dataclass

from sqlalchemy import text

from app.database.connection import engine
from app.database.user_repository import get_last_7_days_kii_performance


ALLOWED_KII_IDS = {117, 118, 119, 120, 121}


def calculate_performance(
    user_id: int,
) -> dict[str, Any]:
    """
    Calculate the user's 7-day performance for KIIs 117-121
    and identify the lowest-performing KII.
    """

    rows = get_last_7_days_kii_performance(user_id)

    # Keep only the five KIIs defined by the Performance Plan.
    rows = [
        row
        for row in rows
        if int(row["kii_id"]) in ALLOWED_KII_IDS
    ]

    if not rows:
        raise ValueError(
            f"No performance data found for user: {user_id}"
        )

    performance_data: list[dict[str, Any]] = []

    for row in rows:
        kii_id = int(row["kii_id"])
        kii_name = str(row["kii_name"])

        daily_target = row.get("daily_target")
        seven_day_actual = row.get("seven_day_actual") or 0

        # Convert numeric database values safely.
        daily_target = (
            float(daily_target)
            if daily_target is not None
            else None
        )

        seven_day_actual = float(seven_day_actual)

        # 7-Day Target = Daily Target × 7
        if daily_target is not None:
            seven_day_target = daily_target * 7
        else:
            seven_day_target = None

        # Performance % = Actual / Target × 100
        if seven_day_target is None or seven_day_target <= 0:
            performance_percentage = 0.0
        else:
            performance_percentage = (
                seven_day_actual / seven_day_target
            ) * 100

        performance_data.append(
            {
                "kii_id": kii_id,
                "kii_name": kii_name,
                "monthly_target": row.get("monthly_target"),
                "daily_target": daily_target,
                "seven_day_target": seven_day_target,
                "seven_day_actual": seven_day_actual,
                "performance_percentage": performance_percentage,
            }
        )

    # Compare all five KIIs and select the lowest-performing KII.
    weakest_kii = min(
        performance_data,
        key=lambda item: item["performance_percentage"],
    )

    return {
        "user_id": user_id,
        "performance": performance_data,
        "improvement_area": {
            "kii_id": weakest_kii["kii_id"],
            "kii_name": weakest_kii["kii_name"],
            "performance_percentage": weakest_kii[
                "performance_percentage"
            ],
        },
    }


@dataclass(frozen=True)
class KiiPerformance:
    kii_id: int
    kii_name: str
    actual: float
    target: float
    percentage: float
    daily_target: float


def _get_performance_rows(user_id: int, db_engine=engine) -> list[dict[str, Any]]:
    query = text("""
        WITH latest AS (
            SELECT MAX(kii_date) AS anchor_date
            FROM public.key_input_indicators
            WHERE user_id = :user_id
              AND account_id = 14
              AND status = 1
              AND kii_id IN (117, 118, 119, 120, 121)
        ), days AS (
            SELECT DISTINCT kii_date
            FROM public.key_input_indicators, latest
            WHERE user_id = :user_id
              AND account_id = 14
              AND status = 1
              AND kii_id IN (117, 118, 119, 120, 121)
              AND kii_date <= latest.anchor_date
            ORDER BY kii_date DESC LIMIT 7
        ), actuals AS (
            SELECT kii.kii_id, COALESCE(SUM(kii.kill_value), 0) AS actual
            FROM public.key_input_indicators kii
            JOIN days ON days.kii_date = kii.kii_date
            WHERE kii.user_id = :user_id
              AND kii.account_id = 14
              AND kii.status = 1
              AND kii.kii_id IN (117, 118, 119, 120, 121)
            GROUP BY kii.kii_id
        ), user_targets AS (
            SELECT DISTINCT ON (kit.kii_id)
                   kit.kii_id, kit.kii_target
            FROM public.key_input_target kit
            WHERE kit.user_id = :user_id
              AND kit.account_id = 14
              AND kit.status = 1
              AND kit.kii_id IN (117, 118, 119, 120, 121)
            ORDER BY kit.kii_id, kit.target_date DESC, kit.created_at DESC
        )
        SELECT km.id AS kii_id,
               km.kii_name,
               COALESCE(ut.kii_target / 22.0, km.daily_target) AS daily_target,
               COALESCE(a.actual, 0) AS actual
        FROM public.kii_master km
        LEFT JOIN user_targets ut ON ut.kii_id = km.id
        LEFT JOIN actuals a ON a.kii_id = km.id
        WHERE km.id IN (117, 118, 119, 120, 121)
          AND km.account_id = 14
          AND km.status = 1
        ORDER BY km.id
    """)
    with db_engine.connect() as connection:
        return [dict(row) for row in connection.execute(query, {"user_id": user_id}).mappings()]


def get_performance_details(user_id: int, db_engine=engine) -> list[KiiPerformance]:
    performances = []
    for row in _get_performance_rows(user_id, db_engine):
        daily_target = float(row.get("daily_target", 0) or 0)
        target = daily_target * 7
        actual = float(row.get("actual", 0) or 0)
        percentage = actual / target * 100 if target > 0 else 0.0
        performances.append(
            KiiPerformance(
                kii_id=int(row["kii_id"]),
                kii_name=str(row["kii_name"]),
                actual=actual,
                target=target,
                percentage=percentage,
                daily_target=daily_target,
            )
        )
    return performances


def select_weakest_performance(performances: list[KiiPerformance]) -> KiiPerformance | None:
    return min(performances, key=lambda item: (item.percentage, item.kii_id)) if performances else None


def get_weakest_performance(user_id: int, db_engine=engine) -> KiiPerformance | None:
    return select_weakest_performance(get_performance_details(user_id, db_engine))
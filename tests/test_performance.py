from __future__ import annotations

from sqlalchemy import text

from app.database.connection import engine
from app.performance.performance import calculate_performance


ALLOWED_KII_IDS = {117, 118, 119, 120, 121}


def get_test_user_id() -> int:
    """
    Find a real user who has non-zero performance data
    using the latest available data date.
    """

    query = text(
        """
        WITH latest_date AS (
            SELECT MAX(kii_date) AS max_date
            FROM public.key_input_indicators
            WHERE account_id = 14
              AND status = 1
              AND kii_id IN (117, 118, 119, 120, 121)
        )

        SELECT ki.user_id
        FROM public.key_input_indicators ki

        CROSS JOIN latest_date ld

        WHERE ki.account_id = 14
          AND ki.status = 1
          AND ki.kii_id IN (117, 118, 119, 120, 121)

          AND ki.kii_date BETWEEN
                ld.max_date - INTERVAL '6 days'
                AND ld.max_date

        GROUP BY ki.user_id

        HAVING SUM(
            COALESCE(ki.kill_value, 0)
        ) > 0

        ORDER BY ki.user_id
        LIMIT 1
        """
    )

    with engine.connect() as connection:
        user_id = connection.execute(
            query
        ).scalar_one_or_none()

    if user_id is None:
        raise AssertionError(
            "No real user with non-zero kill_value "
            "was found for KIIs 117-121 in the "
            "latest available 7-day period."
        )

    return int(user_id)


def get_expected_targets(
    user_id: int,
) -> dict[int, dict]:
    """
    Get the expected daily targets directly from the database.

    This is used to independently verify performance.py.
    """

    query = text(
        """
        WITH user_targets AS (
            SELECT DISTINCT ON (kit.kii_id)
                kit.kii_id,
                kit.kii_target AS monthly_target
            FROM public.key_input_target kit
            WHERE kit.user_id = :user_id
              AND kit.account_id = 14
              AND kit.status = 1
            ORDER BY
                kit.kii_id,
                kit.target_date DESC,
                kit.created_at DESC
        )

        SELECT
            km.id AS kii_id,
            km.kii_name,
            ut.monthly_target,

            COALESCE(
                ut.monthly_target / 22.0,
                km.daily_target
            ) AS expected_daily_target

        FROM public.kii_master km

        LEFT JOIN user_targets ut
            ON ut.kii_id = km.id

        WHERE km.id IN (117, 118, 119, 120, 121)
          AND km.account_id = 14
          AND km.status = 1

        ORDER BY km.id
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(
            query,
            {"user_id": user_id},
        ).mappings().all()

    return {
        int(row["kii_id"]): dict(row)
        for row in rows
    }


def test_real_performance():
    """
    Full real-database integration test for performance.py.
    """

    # ---------------------------------------------------------
    # 1. Find a real user with actual performance data
    # ---------------------------------------------------------

    user_id = get_test_user_id()

    # ---------------------------------------------------------
    # 2. Run actual performance calculation
    # ---------------------------------------------------------

    result = calculate_performance(user_id)

    performance = result["performance"]
    improvement_area = result["improvement_area"]

    # ---------------------------------------------------------
    # 3. Verify all five KIIs are present
    # ---------------------------------------------------------

    actual_kii_ids = {
        int(row["kii_id"])
        for row in performance
    }

    assert actual_kii_ids == ALLOWED_KII_IDS, (
        f"Expected KIIs {ALLOWED_KII_IDS}, "
        f"got {actual_kii_ids}"
    )

    assert len(performance) == 5

    # ---------------------------------------------------------
    # 4. Get independent expected target values
    # ---------------------------------------------------------

    expected_targets = get_expected_targets(user_id)

    assert set(expected_targets.keys()) == ALLOWED_KII_IDS

    # ---------------------------------------------------------
    # 5. Verify every KII
    # ---------------------------------------------------------

    for row in performance:

        kii_id = int(row["kii_id"])

        expected = expected_targets[kii_id]

        monthly_target = expected["monthly_target"]

        expected_daily_target = float(
            expected["expected_daily_target"]
        )

        actual_daily_target = float(
            row["daily_target"]
        )

        # -----------------------------------------------------
        # Verify daily target
        # -----------------------------------------------------

        assert abs(
            actual_daily_target
            - expected_daily_target
        ) < 0.000001, (
            f"KII {kii_id}: "
            f"expected daily target "
            f"{expected_daily_target}, "
            f"got {actual_daily_target}"
        )

        # -----------------------------------------------------
        # Verify monthly target / 22
        # -----------------------------------------------------

        if monthly_target is not None:

            expected_from_monthly = (
                float(monthly_target) / 22
            )

            assert abs(
                actual_daily_target
                - expected_from_monthly
            ) < 0.000001, (
                f"KII {kii_id}: "
                f"monthly target / 22 is incorrect"
            )

        # -----------------------------------------------------
        # Verify 7-day target
        # -----------------------------------------------------

        expected_seven_day_target = (
            actual_daily_target * 7
        )

        actual_seven_day_target = float(
            row["seven_day_target"]
        )

        assert abs(
            actual_seven_day_target
            - expected_seven_day_target
        ) < 0.000001, (
            f"KII {kii_id}: "
            f"7-day target is incorrect"
        )

        # -----------------------------------------------------
        # Verify actual value
        # -----------------------------------------------------

        seven_day_actual = float(
            row["seven_day_actual"]
        )

        assert seven_day_actual >= 0, (
            f"KII {kii_id}: "
            f"7-day actual cannot be negative"
        )

        # -----------------------------------------------------
        # Verify performance percentage
        # -----------------------------------------------------

        if actual_seven_day_target > 0:

            expected_performance = (
                seven_day_actual
                / actual_seven_day_target
            ) * 100

        else:

            expected_performance = 0.0

        actual_performance = float(
            row["performance_percentage"]
        )

        assert abs(
            actual_performance
            - expected_performance
        ) < 0.000001, (
            f"KII {kii_id}: "
            f"performance percentage is incorrect. "
            f"Expected {expected_performance}, "
            f"got {actual_performance}"
        )

    # ---------------------------------------------------------
    # 6. Verify lowest-performing KII
    # ---------------------------------------------------------

    expected_weakest = min(
        performance,
        key=lambda row: row["performance_percentage"],
    )

    assert (
        improvement_area["kii_id"]
        == expected_weakest["kii_id"]
    )

    assert (
        improvement_area["kii_name"]
        == expected_weakest["kii_name"]
    )

    assert abs(
        float(
            improvement_area[
                "performance_percentage"
            ]
        )
        - float(
            expected_weakest[
                "performance_percentage"
            ]
        )
    ) < 0.000001

    # ---------------------------------------------------------
    # 7. Print readable real-data result
    # ---------------------------------------------------------

    print("\n")
    print("=" * 100)
    print(
        f"REAL PERFORMANCE TEST - USER {user_id}"
    )
    print("=" * 100)

    print(
        f"{'KII':<6}"
        f"{'Name':<42}"
        f"{'Daily':>10}"
        f"{'7-Day Target':>15}"
        f"{'Actual':>10}"
        f"{'Performance':>15}"
    )

    print("-" * 100)

    for row in performance:

        print(
            f"{row['kii_id']:<6}"
            f"{row['kii_name']:<42}"
            f"{float(row['daily_target']):>10.2f}"
            f"{float(row['seven_day_target']):>15.2f}"
            f"{float(row['seven_day_actual']):>10.2f}"
            f"{float(row['performance_percentage']):>14.2f}%"
        )

    print("-" * 100)

    print(
        f"Improvement Area : "
        f"{improvement_area['kii_id']} - "
        f"{improvement_area['kii_name']}"
    )

    print(
        f"Lowest Performance: "
        f"{float(improvement_area['performance_percentage']):.2f}%"
    )

    print("=" * 100)
    print("ALL PERFORMANCE PLAN CHECKS PASSED")
    print("=" * 100)
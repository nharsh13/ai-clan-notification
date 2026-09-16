from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.database.connection import engine
from app.main import app
from app.performance.performance import calculate_performance


# ============================================================
# CONFIGURATION
# ============================================================

ALLOWED_KII_IDS = {
    117,
    118,
    119,
    120,
    121,
}

ACCOUNT_ID = 14
DAYS = 7
MONTHLY_DIVISOR = 22

client = TestClient(app)


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_latest_kii_date() -> date:
    """
    Get the latest available KII date for the allowed KIIs.
    """

    query = text(
        """
        SELECT MAX(kii_date)::date
        FROM public.key_input_indicators
        WHERE account_id = :account_id
          AND status = 1
          AND kii_id IN (
              117,
              118,
              119,
              120,
              121
          )
        """
    )

    with engine.connect() as connection:
        latest_date = connection.execute(
            query,
            {
                "account_id": ACCOUNT_ID,
            },
        ).scalar_one_or_none()

    if latest_date is None:
        raise AssertionError(
            "No KII data found for account_id=14 "
            "and KIIs 117-121."
        )

    return latest_date


def get_test_user_id(
    latest_date: date,
) -> int:
    """
    Find a real user having non-zero performance
    during the latest available 7-day period.
    """

    start_date = (
        latest_date
        - timedelta(days=DAYS - 1)
    )

    query = text(
        """
        SELECT user_id

        FROM public.key_input_indicators

        WHERE account_id = :account_id
          AND status = 1

          AND kii_id IN (
              117,
              118,
              119,
              120,
              121
          )

          AND kii_date::date BETWEEN
              :start_date
              AND :end_date

        GROUP BY user_id

        HAVING SUM(
            COALESCE(kill_value, 0)
        ) > 0

        ORDER BY user_id

        LIMIT 1
        """
    )

    with engine.connect() as connection:
        user_id = connection.execute(
            query,
            {
                "account_id": ACCOUNT_ID,
                "start_date": start_date,
                "end_date": latest_date,
            },
        ).scalar_one_or_none()

    if user_id is None:
        raise AssertionError(
            "No real user with non-zero "
            "performance was found in the "
            "latest 7-day period."
        )

    return int(user_id)


# ============================================================
# EXPECTED TARGETS
# ============================================================

def get_expected_targets(
    user_id: int,
) -> dict[int, dict]:
    """
    Independently calculate the expected daily target.

    Priority:

        key_input_target.kii_target / 22
                    |
                    | unavailable
                    v
        kii_master.daily_target
    """

    query = text(
        """
        WITH latest_user_targets AS (

            SELECT DISTINCT ON (kit.kii_id)

                kit.kii_id,

                kit.kii_target
                    AS monthly_target

            FROM public.key_input_target kit

            WHERE kit.user_id = :user_id

              AND kit.account_id = :account_id

              AND kit.status = 1

              AND kit.kii_id IN (
                  117,
                  118,
                  119,
                  120,
                  121
              )

            ORDER BY
                kit.kii_id,
                kit.target_date DESC NULLS LAST,
                kit.created_at DESC NULLS LAST,
                kit.id DESC
        )

        SELECT

            km.id AS kii_id,

            km.kii_name,

            km.daily_target
                AS fallback_daily_target,

            lut.monthly_target,

            CASE

                WHEN lut.monthly_target IS NOT NULL
                THEN lut.monthly_target / 22.0

                ELSE km.daily_target

            END AS expected_daily_target

        FROM public.kii_master km

        LEFT JOIN latest_user_targets lut

            ON lut.kii_id = km.id

        WHERE km.id IN (
            117,
            118,
            119,
            120,
            121
        )

          AND km.account_id = :account_id

          AND km.status = 1

        ORDER BY km.id
        """
    )

    with engine.connect() as connection:

        rows = connection.execute(
            query,
            {
                "user_id": user_id,
                "account_id": ACCOUNT_ID,
            },
        ).mappings().all()

    result = {
        int(row["kii_id"]): dict(row)
        for row in rows
    }

    assert set(result.keys()) == ALLOWED_KII_IDS, (
        f"Expected target information for "
        f"{ALLOWED_KII_IDS}, "
        f"got {set(result.keys())}"
    )

    return result


# ============================================================
# EXPECTED ACTUAL PERFORMANCE
# ============================================================

def get_expected_actuals(
    user_id: int,
    latest_date: date,
) -> dict[int, float]:
    """
    Independently calculate 7-day actual performance.

    Actual =
        SUM(kill_value)

    NULL kill_value =
        0
    """

    start_date = (
        latest_date
        - timedelta(days=DAYS - 1)
    )

    query = text(
        """
        SELECT

            kii_id,

            COALESCE(
                SUM(
                    COALESCE(kill_value, 0)
                ),
                0
            ) AS seven_day_actual

        FROM public.key_input_indicators

        WHERE user_id = :user_id

          AND account_id = :account_id

          AND status = 1

          AND kii_id IN (
              117,
              118,
              119,
              120,
              121
          )

          AND kii_date::date BETWEEN
              :start_date
              AND :end_date

        GROUP BY kii_id

        ORDER BY kii_id
        """
    )

    with engine.connect() as connection:

        rows = connection.execute(
            query,
            {
                "user_id": user_id,
                "account_id": ACCOUNT_ID,
                "start_date": start_date,
                "end_date": latest_date,
            },
        ).mappings().all()

    actuals = {
        int(row["kii_id"]): float(
            row["seven_day_actual"] or 0
        )
        for row in rows
    }

    # If a KII has no record,
    # its actual performance is zero.
    for kii_id in ALLOWED_KII_IDS:
        actuals.setdefault(
            kii_id,
            0.0,
        )

    return actuals


# ============================================================
# HELPER
# ============================================================

def assert_performance_result(
    result: dict,
    user_id: int,
    expected_targets: dict[int, dict],
    expected_actuals: dict[int, float],
):
    """
    Validate the complete performance result.
    """

    assert result is not None, (
        "Performance result is None."
    )

    assert isinstance(result, dict), (
        "Performance result must be a dictionary."
    )

    assert "performance" in result, (
        "Missing 'performance' in result."
    )

    assert "improvement_area" in result, (
        "Missing 'improvement_area' in result."
    )

    performance = result["performance"]

    improvement_area = result[
        "improvement_area"
    ]

    # --------------------------------------------------------
    # Verify exactly five KIIs
    # --------------------------------------------------------

    actual_kii_ids = {
        int(row["kii_id"])
        for row in performance
    }

    assert actual_kii_ids == ALLOWED_KII_IDS, (
        f"Expected KIIs "
        f"{ALLOWED_KII_IDS}, "
        f"got {actual_kii_ids}"
    )

    assert len(performance) == 5, (
        f"Expected exactly 5 KIIs, "
        f"got {len(performance)}"
    )

    # --------------------------------------------------------
    # Verify each KII
    # --------------------------------------------------------

    for row in performance:

        kii_id = int(
            row["kii_id"]
        )

        assert kii_id in ALLOWED_KII_IDS

        expected = expected_targets[
            kii_id
        ]

        monthly_target = expected[
            "monthly_target"
        ]

        fallback_daily_target = expected[
            "fallback_daily_target"
        ]

        expected_daily_target = expected[
            "expected_daily_target"
        ]

        # ----------------------------------------------------
        # Daily target
        # ----------------------------------------------------

        actual_daily_target = float(
            row["daily_target"]
        )

        if monthly_target is not None:

            expected_from_monthly = (
                float(monthly_target)
                / MONTHLY_DIVISOR
            )

            assert abs(
                actual_daily_target
                - expected_from_monthly
            ) < 0.000001, (
                f"KII {kii_id}: "
                f"Expected monthly target / 22 = "
                f"{expected_from_monthly}, "
                f"got {actual_daily_target}"
            )

        else:

            assert abs(
                actual_daily_target
                - float(
                    fallback_daily_target
                )
            ) < 0.000001, (
                f"KII {kii_id}: "
                f"Expected fallback daily target "
                f"{fallback_daily_target}, "
                f"got {actual_daily_target}"
            )

        # ----------------------------------------------------
        # Independent target verification
        # ----------------------------------------------------

        assert abs(
            actual_daily_target
            - float(expected_daily_target)
        ) < 0.000001, (
            f"KII {kii_id}: "
            f"Daily target mismatch."
        )

        # ----------------------------------------------------
        # 7-day target
        # ----------------------------------------------------

        expected_7_day_target = (
            actual_daily_target * DAYS
        )

        actual_7_day_target = float(
            row["seven_day_target"]
        )

        assert abs(
            actual_7_day_target
            - expected_7_day_target
        ) < 0.000001, (
            f"KII {kii_id}: "
            f"Expected 7-day target "
            f"{expected_7_day_target}, "
            f"got {actual_7_day_target}"
        )

        # ----------------------------------------------------
        # 7-day actual
        # ----------------------------------------------------

        expected_actual = expected_actuals.get(
            kii_id,
            0.0,
        )

        actual_value = float(
            row["seven_day_actual"]
        )

        assert abs(
            actual_value
            - expected_actual
        ) < 0.000001, (
            f"KII {kii_id}: "
            f"Expected 7-day actual "
            f"{expected_actual}, "
            f"got {actual_value}"
        )

        assert actual_value >= 0, (
            f"KII {kii_id}: "
            f"7-day actual cannot be negative."
        )

        # ----------------------------------------------------
        # Performance percentage
        # ----------------------------------------------------

        if actual_7_day_target > 0:

            expected_percentage = (
                actual_value
                / actual_7_day_target
            ) * 100

        else:

            expected_percentage = 0.0

        actual_percentage = float(
            row["performance_percentage"]
        )

        assert abs(
            actual_percentage
            - expected_percentage
        ) < 0.000001, (
            f"KII {kii_id}: "
            f"Expected performance "
            f"{expected_percentage}, "
            f"got {actual_percentage}"
        )

        assert actual_percentage >= 0, (
            f"KII {kii_id}: "
            f"Performance percentage "
            f"cannot be negative."
        )

    # --------------------------------------------------------
    # Find weakest KII independently
    # --------------------------------------------------------

    expected_weakest = min(
        performance,
        key=lambda row: float(
            row["performance_percentage"]
        ),
    )

    # --------------------------------------------------------
    # Improvement area
    # --------------------------------------------------------

    assert improvement_area is not None, (
        "Improvement area must not be None."
    )

    assert int(
        improvement_area["kii_id"]
    ) == int(
        expected_weakest["kii_id"]
    ), (
        f"Expected weakest KII "
        f"{expected_weakest['kii_id']}, "
        f"got "
        f"{improvement_area['kii_id']}"
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
        -
        float(
            expected_weakest[
                "performance_percentage"
            ]
        )
    ) < 0.000001

    assert int(
        improvement_area["kii_id"]
    ) in ALLOWED_KII_IDS


# ============================================================
# TEST 1
# BUSINESS LOGIC
# ============================================================

def test_real_performance():
    """
    Test calculate_performance() directly.

    This validates the business logic.
    """

    # --------------------------------------------------------
    # Find latest date
    # --------------------------------------------------------

    latest_date = (
        get_latest_kii_date()
    )

    start_date = (
        latest_date
        - timedelta(days=DAYS - 1)
    )

    # --------------------------------------------------------
    # Find real user
    # --------------------------------------------------------

    user_id = get_test_user_id(
        latest_date
    )

    # --------------------------------------------------------
    # Get independent DB expectations
    # --------------------------------------------------------

    expected_targets = (
        get_expected_targets(
            user_id
        )
    )

    expected_actuals = (
        get_expected_actuals(
            user_id,
            latest_date,
        )
    )

    # --------------------------------------------------------
    # Run application calculation
    # --------------------------------------------------------

    result = calculate_performance(
        user_id
    )

    # --------------------------------------------------------
    # Validate complete result
    # --------------------------------------------------------

    assert_performance_result(
        result=result,
        user_id=user_id,
        expected_targets=expected_targets,
        expected_actuals=expected_actuals,
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    performance = result[
        "performance"
    ]

    improvement_area = result[
        "improvement_area"
    ]

    print()
    print("=" * 110)
    print(
        "REAL PERFORMANCE BUSINESS LOGIC TEST"
    )
    print("=" * 110)

    print(
        f"User ID      : {user_id}"
    )

    print(
        f"Account ID   : {ACCOUNT_ID}"
    )

    print(
        f"Period Start : {start_date}"
    )

    print(
        f"Period End   : {latest_date}"
    )

    print(
        f"Period       : {DAYS} days"
    )

    print("-" * 110)

    print(
        f"{'KII':<6}"
        f"{'Name':<42}"
        f"{'Monthly':>12}"
        f"{'Daily':>12}"
        f"{'7D Target':>14}"
        f"{'7D Actual':>14}"
        f"{'Performance':>15}"
    )

    print("-" * 110)

    for row in performance:

        kii_id = int(
            row["kii_id"]
        )

        monthly_target = (
            expected_targets[
                kii_id
            ]["monthly_target"]
        )

        monthly_display = (
            f"{float(monthly_target):.2f}"
            if monthly_target is not None
            else "FALLBACK"
        )

        print(
            f"{kii_id:<6}"
            f"{str(row['kii_name'])[:40]:<42}"
            f"{monthly_display:>12}"
            f"{float(row['daily_target']):>12.2f}"
            f"{float(row['seven_day_target']):>14.2f}"
            f"{float(row['seven_day_actual']):>14.2f}"
            f"{float(row['performance_percentage']):>14.2f}%"
        )

    print("-" * 110)

    print(
        f"Improvement Area : "
        f"{improvement_area['kii_id']} - "
        f"{improvement_area['kii_name']}"
    )

    print(
        f"Lowest Performance: "
        f"{float(improvement_area['performance_percentage']):.2f}%"
    )

    print("=" * 110)
    print(
        "BUSINESS LOGIC TEST PASSED"
    )
    print("=" * 110)


# ============================================================
# TEST 2
# GET /notification/performance API
# ============================================================

def test_performance_api():
    """
    Test the actual:

        GET /notification/performance

    Validates the real API response against:
    1. Independent database calculations
    2. calculate_performance() service result
    """

    # --------------------------------------------------------
    # 1. Find real test user
    # --------------------------------------------------------

    latest_date = get_latest_kii_date()

    user_id = get_test_user_id(
        latest_date
    )

    # --------------------------------------------------------
    # 2. Get independent DB expectations
    # --------------------------------------------------------

    expected_targets = get_expected_targets(
        user_id
    )

    expected_actuals = get_expected_actuals(
        user_id,
        latest_date,
    )

    # --------------------------------------------------------
    # 3. Call actual API
    # --------------------------------------------------------

    response = client.get(
        "/notification/performance",
        params={
            "user_id": user_id,
        },
    )

    # --------------------------------------------------------
    # 4. Verify HTTP status
    # --------------------------------------------------------

    assert response.status_code == 200, (
        "\n"
        "GET /notification/performance failed.\n"
        f"Status Code: {response.status_code}\n"
        f"Response: {response.text}"
    )

    # --------------------------------------------------------
    # 5. Verify JSON
    # --------------------------------------------------------

    try:
        data = response.json()

    except ValueError as exc:

        raise AssertionError(
            "Performance API did not return valid JSON.\n"
            f"Response: {response.text}"
        ) from exc

    assert isinstance(data, dict), (
        f"Expected JSON object, got {type(data)}"
    )

    # --------------------------------------------------------
    # 6. Verify actual API response structure
    # --------------------------------------------------------

    assert "user_id" in data, (
        "API response is missing 'user_id'."
    )

    assert "seven_day_performance" in data, (
        "API response is missing "
        "'seven_day_performance'."
    )

    assert "weakest_kii" in data, (
        "API response is missing 'weakest_kii'."
    )

    # --------------------------------------------------------
    # 7. Verify user_id
    # --------------------------------------------------------

    assert int(data["user_id"]) == user_id, (
        f"Expected user_id {user_id}, "
        f"got {data['user_id']}"
    )

    # --------------------------------------------------------
    # 8. Get API performance data
    # --------------------------------------------------------

    api_performance = data[
        "seven_day_performance"
    ]

    assert isinstance(
        api_performance,
        list,
    ), (
        "'seven_day_performance' "
        "must be a list."
    )

    # --------------------------------------------------------
    # 9. Verify exactly five KIIs
    # --------------------------------------------------------

    api_kii_ids = {
        int(row["kii_id"])
        for row in api_performance
    }

    assert api_kii_ids == ALLOWED_KII_IDS, (
        f"Expected KIIs "
        f"{ALLOWED_KII_IDS}, "
        f"got {api_kii_ids}"
    )

    assert len(api_performance) == 5, (
        f"Expected exactly 5 KIIs, "
        f"got {len(api_performance)}"
    )

    # --------------------------------------------------------
    # 10. Get service result
    # --------------------------------------------------------

    service_result = calculate_performance(
        user_id
    )

    service_performance = (
        service_result["performance"]
    )

    # --------------------------------------------------------
    # 11. Convert API and service results
    #     into KII dictionaries
    # --------------------------------------------------------

    api_by_kii = {
        int(row["kii_id"]): row
        for row in api_performance
    }

    service_by_kii = {
        int(row["kii_id"]): row
        for row in service_performance
    }

    # --------------------------------------------------------
    # 12. Compare every KII
    # --------------------------------------------------------

    for kii_id in ALLOWED_KII_IDS:

        assert kii_id in api_by_kii
        assert kii_id in service_by_kii

        api_row = api_by_kii[
            kii_id
        ]

        service_row = service_by_kii[
            kii_id
        ]

        # ----------------------------------------------------
        # KII name
        # ----------------------------------------------------

        assert (
            api_row["kii_name"]
            == service_row["kii_name"]
        ), (
            f"KII {kii_id}: "
            "API KII name does not "
            "match service."
        )

        # ----------------------------------------------------
        # Monthly target
        # ----------------------------------------------------

        if (
            "monthly_target" in api_row
            and "monthly_target" in service_row
        ):

            api_monthly = (
                api_row["monthly_target"]
            )

            service_monthly = (
                service_row["monthly_target"]
            )

            if (
                api_monthly is not None
                and service_monthly is not None
            ):

                assert abs(
                    float(api_monthly)
                    - float(service_monthly)
                ) < 0.000001, (
                    f"KII {kii_id}: "
                    "API monthly target does "
                    "not match service."
                )

        # ----------------------------------------------------
        # Daily target
        # ----------------------------------------------------

        assert abs(
            float(
                api_row["daily_target"]
            )
            -
            float(
                service_row["daily_target"]
            )
        ) < 0.000001, (
            f"KII {kii_id}: "
            "API daily target does "
            "not match service."
        )

        # ----------------------------------------------------
        # 7-day target
        # ----------------------------------------------------

        assert abs(
            float(
                api_row["seven_day_target"]
            )
            -
            float(
                service_row["seven_day_target"]
            )
        ) < 0.000001, (
            f"KII {kii_id}: "
            "API 7-day target does "
            "not match service."
        )

        # ----------------------------------------------------
        # 7-day actual
        # ----------------------------------------------------

        assert abs(
            float(
                api_row["seven_day_actual"]
            )
            -
            float(
                service_row["seven_day_actual"]
            )
        ) < 0.000001, (
            f"KII {kii_id}: "
            "API 7-day actual does "
            "not match service."
        )

        # ----------------------------------------------------
        # Performance percentage
        # ----------------------------------------------------

        assert abs(
            float(
                api_row[
                    "performance_percentage"
                ]
            )
            -
            float(
                service_row[
                    "performance_percentage"
                ]
            )
        ) < 0.000001, (
            f"KII {kii_id}: "
            "API performance percentage "
            "does not match service."
        )

    # --------------------------------------------------------
    # 13. Independently verify API actual values
    # --------------------------------------------------------

    for kii_id in ALLOWED_KII_IDS:

        api_row = api_by_kii[
            kii_id
        ]

        expected_actual = (
            expected_actuals.get(
                kii_id,
                0.0,
            )
        )

        assert abs(
            float(
                api_row["seven_day_actual"]
            )
            -
            expected_actual
        ) < 0.000001, (
            f"KII {kii_id}: "
            f"Expected actual "
            f"{expected_actual}, "
            f"got "
            f"{api_row['seven_day_actual']}"
        )

    # --------------------------------------------------------
    # 14. Independently verify targets
    # --------------------------------------------------------

    for kii_id in ALLOWED_KII_IDS:

        api_row = api_by_kii[
            kii_id
        ]

        expected_target = (
            expected_targets[
                kii_id
            ]
        )

        expected_daily = float(
            expected_target[
                "expected_daily_target"
            ]
        )

        actual_daily = float(
            api_row["daily_target"]
        )

        assert abs(
            actual_daily
            - expected_daily
        ) < 0.000001, (
            f"KII {kii_id}: "
            "API daily target does "
            "not match database."
        )

    # --------------------------------------------------------
    # 15. Verify weakest KII
    # --------------------------------------------------------

    weakest_kii = data[
        "weakest_kii"
    ]

    assert weakest_kii is not None, (
        "API weakest_kii must not be None."
    )

    expected_weakest = min(
        api_performance,
        key=lambda row: float(
            row["performance_percentage"]
        ),
    )

    assert int(
        weakest_kii["kii_id"]
    ) == int(
        expected_weakest["kii_id"]
    ), (
        "API weakest KII is incorrect."
    )

    assert (
        weakest_kii["kii_name"]
        == expected_weakest["kii_name"]
    ), (
        "API weakest KII name is incorrect."
    )

    assert abs(
        float(
            weakest_kii[
                "performance_percentage"
            ]
        )
        -
        float(
            expected_weakest[
                "performance_percentage"
            ]
        )
    ) < 0.000001, (
        "API weakest KII performance "
        "percentage is incorrect."
    )

    # --------------------------------------------------------
    # 16. Print API result
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "GET /notification/performance API TEST"
    )
    print("=" * 100)

    print(
        f"User ID       : {user_id}"
    )

    print(
        f"Status Code   : "
        f"{response.status_code}"
    )

    print(
        f"KIIs Returned : "
        f"{sorted(api_kii_ids)}"
    )

    print(
        f"Weakest KII   : "
        f"{weakest_kii['kii_id']} - "
        f"{weakest_kii['kii_name']}"
    )

    print(
        f"Performance   : "
        f"{float(weakest_kii['performance_percentage']):.2f}%"
    )

    print("=" * 100)
    print(
        "PERFORMANCE API TEST PASSED"
    )
    print("=" * 100)

def test_performance_function_matches_api():
    """
    Verify that calculate_performance() and
    GET /notification/performance return the same
    performance result for the same user.
    """

    latest_date = get_latest_kii_date()
    user_id = get_test_user_id(latest_date)

    # --------------------------------------------------------
    # 1. Direct function result
    # --------------------------------------------------------

    function_result = calculate_performance(
        user_id
    )

    # --------------------------------------------------------
    # 2. API result
    # --------------------------------------------------------

    response = client.get(
        "/notification/performance",
        params={
            "user_id": user_id,
        },
    )

    assert response.status_code == 200, (
        f"API failed: {response.text}"
    )

    api_result = response.json()

    # --------------------------------------------------------
    # 3. Compare KII IDs
    # --------------------------------------------------------

    function_performance = {
        int(row["kii_id"]): row
        for row in function_result[
            "performance"
        ]
    }

    api_performance = {
        int(row["kii_id"]): row
        for row in api_result[
            "seven_day_performance"
        ]
    }

    assert set(function_performance.keys()) == (
        set(api_performance.keys())
    )

    # --------------------------------------------------------
    # 4. Compare every KII
    # --------------------------------------------------------

    for kii_id in function_performance:

        function_row = function_performance[
            kii_id
        ]

        api_row = api_performance[
            kii_id
        ]

        assert function_row["kii_name"] == (
            api_row["kii_name"]
        )

        assert abs(
            float(function_row["daily_target"])
            - float(api_row["daily_target"])
        ) < 0.000001

        assert abs(
            float(function_row["seven_day_target"])
            - float(api_row["seven_day_target"])
        ) < 0.000001

        assert abs(
            float(function_row["seven_day_actual"])
            - float(api_row["seven_day_actual"])
        ) < 0.000001

        assert abs(
            float(
                function_row[
                    "performance_percentage"
                ]
            )
            -
            float(
                api_row[
                    "performance_percentage"
                ]
            )
        ) < 0.000001

    # --------------------------------------------------------
    # 5. Compare weakest KII
    # --------------------------------------------------------

    function_weakest = (
        function_result["improvement_area"]
    )

    api_weakest = (
        api_result["weakest_kii"]
    )

    assert int(
        function_weakest["kii_id"]
    ) == int(
        api_weakest["kii_id"]
    )

    assert (
        function_weakest["kii_name"]
        == api_weakest["kii_name"]
    )

    assert abs(
        float(
            function_weakest[
                "performance_percentage"
            ]
        )
        -
        float(
            api_weakest[
                "performance_percentage"
            ]
        )
    ) < 0.000001

    print()
    print("=" * 80)
    print(
        "FUNCTION vs API COMPARISON"
    )
    print("=" * 80)
    print(f"User ID: {user_id}")
    print("calculate_performance()  ==  API")
    print("All 5 KII results match")
    print("Weakest KII matches")
    print("=" * 80)
    print(
        "FUNCTION AND API MATCH PASSED"
    )
    print("=" * 80)
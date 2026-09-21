"""Context-independent exact arithmetic for persisted decimal money values."""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable

MAX_DECIMAL_PLACES = 30
MAX_INTEGER_DIGITS = 30


def validate_exact_decimal(value: Decimal, *, field: str) -> Decimal:
    exponent = int(value.as_tuple().exponent)
    if exponent < -MAX_DECIMAL_PLACES:
        raise ValueError(
            f"{field} supports at most {MAX_DECIMAL_PLACES} decimal places"
        )
    if value != 0 and value.adjusted() + 1 > MAX_INTEGER_DIGITS:
        raise ValueError(
            f"{field} supports at most {MAX_INTEGER_DIGITS} integer digits"
        )
    return value


def _parts(value: Decimal) -> tuple[int, int]:
    sign, digits, exponent = value.as_tuple()
    coefficient = int("".join(str(digit) for digit in digits) or "0")
    return (-coefficient if sign else coefficient), int(exponent)


def _from_parts(coefficient: int, exponent: int) -> Decimal:
    sign = 1 if coefficient < 0 else 0
    digits = tuple(int(character) for character in str(abs(coefficient)))
    return Decimal((sign, digits, exponent))


def exact_sum(values: Iterable[Decimal]) -> Decimal:
    materialized = tuple(values)
    if not materialized:
        return Decimal("0")
    parts = tuple(_parts(value) for value in materialized)
    common_exponent = min(exponent for _, exponent in parts)
    coefficient = sum(
        value * (10 ** (exponent - common_exponent))
        for value, exponent in parts
    )
    return _from_parts(coefficient, common_exponent)


def exact_add(*values: Decimal) -> Decimal:
    return exact_sum(values)


def exact_subtract(left: Decimal, right: Decimal) -> Decimal:
    coefficient, exponent = _parts(right)
    return exact_sum((left, _from_parts(-coefficient, exponent)))


def exact_multiply(left: Decimal, right: Decimal) -> Decimal:
    left_coefficient, left_exponent = _parts(left)
    right_coefficient, right_exponent = _parts(right)
    return _from_parts(
        left_coefficient * right_coefficient,
        left_exponent + right_exponent,
    )


__all__ = [
    "MAX_DECIMAL_PLACES",
    "MAX_INTEGER_DIGITS",
    "exact_add",
    "exact_multiply",
    "exact_subtract",
    "exact_sum",
    "validate_exact_decimal",
]

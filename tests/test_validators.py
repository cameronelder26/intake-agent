from datetime import date

import pytest

from intake_agent.confidence import VALIDATOR_FAIL_CAP, score_field
from intake_agent.schema import RawField
from intake_agent.validators import (
    normalize_name,
    normalize_value,
    npi_check_digit,
    npi_is_valid,
    validate,
)

TODAY = date(2026, 11, 10)


def test_npi_checksum_cms_example():
    # 1234567893 is the worked example in the CMS NPI check-digit document.
    assert npi_is_valid("1234567893")
    assert not npi_is_valid("1234567890")
    assert npi_check_digit("123456789") == "3"


@pytest.mark.parametrize("npi", ["123456789", "12345678931", "3234567893", "12345678a3"])
def test_npi_rejects_bad_shape(npi):
    assert not npi_is_valid(npi)


@pytest.mark.parametrize("code", ["M17.11", "Z12.11", "S83.241A", "I10", "G47.33"])
def test_icd10_ok(code):
    assert validate("icd10_codes", [code], TODAY) == []


@pytest.mark.parametrize("code", ["17.11", "U07.1", "M1", "m17.11x!"])
def test_icd10_bad(code):
    assert validate("icd10_codes", [code], TODAY)


@pytest.mark.parametrize("code", ["73721", "0001F", "0042T"])
def test_cpt_ok(code):
    assert validate("cpt_codes", [code], TODAY) == []


@pytest.mark.parametrize("code", ["7372", "737211", "ABCDE"])
def test_cpt_bad(code):
    assert validate("cpt_codes", [code], TODAY)


def test_dates():
    assert validate("patient_dob", "1961-03-04", TODAY) == []
    assert validate("patient_dob", "2027-01-01", TODAY) == ["date of birth not in the past"]
    assert validate("patient_dob", "not a date", TODAY)
    assert validate("request_date", "2026-11-11", TODAY)  # future request date
    assert normalize_value("patient_dob", "March 4, 1961") == "1961-03-04"
    assert normalize_value("patient_dob", "03/04/1961") == "1961-03-04"


def test_missing_and_enums():
    assert validate("member_id", None, TODAY) == ["missing"]
    assert validate("icd10_codes", [], TODAY) == ["missing"]
    assert validate("urgency", "stat", TODAY) == []
    assert validate("urgency", "asap", TODAY)


def test_normalize_name():
    assert normalize_name("DOE, JANE") == "Jane Doe"
    assert normalize_name("Dr. Jane Doe, MD") == "Jane Doe"
    assert normalize_name("Dr Doe, Jane") == "Jane Doe"


def test_validator_failure_caps_confidence():
    scored = score_field(
        "referring_npi", RawField(value="1234567890", model_confidence=0.99), TODAY
    )
    assert scored.confidence == VALIDATOR_FAIL_CAP
    assert scored.issues == ["NPI fails checksum"]


def test_missing_field_scores_zero():
    scored = score_field("member_id", RawField(value=None, model_confidence=1.0), TODAY)
    assert scored.confidence == 0.0


def test_valid_field_keeps_model_confidence():
    scored = score_field(
        "referring_npi", RawField(value="1234567893", model_confidence=0.91), TODAY
    )
    assert scored.confidence == 0.91
    assert scored.issues == []

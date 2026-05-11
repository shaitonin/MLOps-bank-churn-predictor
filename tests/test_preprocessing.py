"""Tests for src/preprocessing.py — sklearn-compatible transformers."""
import numpy as np
import pandas as pd
import pytest
from src.preprocessing import (
    ColumnDropper,
    BinaryFlagTransformer,
    CategoricalEncoder,
    AgeBinTransformer,
)


# ── ColumnDropper ─────────────────────────────────────────────────────────────

def test_column_dropper_removes_specified_columns():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    result = ColumnDropper(columns=['A', 'C']).fit_transform(df)
    assert list(result.columns) == ['B']


def test_column_dropper_ignores_missing_columns():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    # 'Z' doesn't exist — should not raise
    result = ColumnDropper(columns=['A', 'Z']).fit_transform(df)
    assert list(result.columns) == ['B']


def test_column_dropper_no_columns_specified():
    df = pd.DataFrame({'A': [1], 'B': [2]})
    result = ColumnDropper(columns=[]).fit_transform(df)
    assert list(result.columns) == ['A', 'B']


# ── BinaryFlagTransformer ────────────────────────────────────────────────────

def test_binary_flag_zero_balance():
    df = pd.DataFrame({'Balance': [0.0, 100.0, 0.0, 50000.0]})
    result = BinaryFlagTransformer().fit_transform(df)
    assert 'HasZeroBalance' in result.columns
    assert list(result['HasZeroBalance']) == [1, 0, 1, 0]


def test_binary_flag_high_risk_products():
    df = pd.DataFrame({'NumOfProducts': [1, 2, 3, 4]})
    result = BinaryFlagTransformer().fit_transform(df)
    assert 'HighRiskProducts' in result.columns
    # products 3 and 4 are high risk
    assert list(result['HighRiskProducts']) == [0, 0, 1, 1]


# ── CategoricalEncoder ────────────────────────────────────────────────────────

def test_categorical_encoder_ohe_geography():
    df = pd.DataFrame({'Geography': ['France', 'Germany', 'Spain', 'France']})
    enc = CategoricalEncoder()
    result = enc.fit_transform(df)
    assert 'Geography_France' in result.columns
    assert 'Geography_Germany' in result.columns
    assert 'Geography_Spain' in result.columns
    assert 'Geography' not in result.columns


def test_categorical_encoder_binary_gender():
    df = pd.DataFrame({'Gender': ['Male', 'Female', 'Male']})
    enc = CategoricalEncoder()
    result = enc.fit_transform(df)
    assert 'Gender_Male' in result.columns
    assert 'Gender_Female' in result.columns
    assert 'Gender' not in result.columns


def test_categorical_encoder_gender_values():
    df = pd.DataFrame({'Gender': ['Male', 'Female', 'Male']})
    result = CategoricalEncoder().fit_transform(df)
    assert list(result['Gender_Male']) == [1, 0, 1]
    assert list(result['Gender_Female']) == [0, 1, 0]


# ── AgeBinTransformer ────────────────────────────────────────────────────────

def test_age_bin_transformer_creates_column():
    df = pd.DataFrame({'Age': [25, 35, 50, 65]})
    result = AgeBinTransformer().fit_transform(df)
    assert 'AgeGroup' in result.columns


def test_age_bin_transformer_values():
    df = pd.DataFrame({'Age': [25, 35, 50, 65]})
    result = AgeBinTransformer().fit_transform(df)
    groups = list(result['AgeGroup'])
    # young < 30, middle 30-55, senior >= 55
    assert groups[0] < groups[1]   # young < middle (ordinal)
    assert groups[2] < groups[3]   # middle < senior (ordinal)


def test_age_bin_transformer_preserves_age_column():
    df = pd.DataFrame({'Age': [30, 45]})
    result = AgeBinTransformer().fit_transform(df)
    assert 'Age' in result.columns

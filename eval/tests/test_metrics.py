"""Known-answer tests for the metric functions (TDD core).

Every case has a hand-computed expected value so a regression in the math is
caught immediately.
"""
from __future__ import annotations

import math

import pytest

from eval import metrics


# --- retrieval: recall@k ----------------------------------------------------

def test_recall_at_k_partial():
    assert metrics.recall_at_k({"a", "b"}, ["a", "c", "b", "d"], k=2) == pytest.approx(0.5)


def test_recall_at_k_full():
    assert metrics.recall_at_k({"a", "b"}, ["a", "c", "b", "d"], k=4) == pytest.approx(1.0)


def test_recall_at_k_none_when_no_gold():
    assert metrics.recall_at_k(set(), ["a", "b"], k=2) is None


def test_recall_at_k_zero():
    assert metrics.recall_at_k({"a"}, ["b", "c"], k=2) == pytest.approx(0.0)


# --- retrieval: MRR ---------------------------------------------------------

def test_mrr_second_position():
    assert metrics.mrr({"b"}, ["a", "b", "c"]) == pytest.approx(0.5)


def test_mrr_first_position():
    assert metrics.mrr({"a"}, ["a", "b"]) == pytest.approx(1.0)


def test_mrr_no_hit():
    assert metrics.mrr({"z"}, ["a", "b"]) == pytest.approx(0.0)


# --- retrieval: nDCG@k ------------------------------------------------------

def test_ndcg_perfect():
    assert metrics.ndcg_at_k({"a"}, ["a", "b", "c"], k=3) == pytest.approx(1.0)


def test_ndcg_rank2_single_gold():
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1.0
    assert metrics.ndcg_at_k({"a"}, ["b", "a", "c"], k=3) == pytest.approx(1 / math.log2(3))


def test_ndcg_two_gold_one_missing():
    # DCG = 1/log2(2) = 1.0; IDCG = 1/log2(2) + 1/log2(3)
    idcg = 1.0 + 1 / math.log2(3)
    assert metrics.ndcg_at_k({"a", "b"}, ["a", "x"], k=2) == pytest.approx(1.0 / idcg)


def test_ndcg_no_gold():
    assert metrics.ndcg_at_k(set(), ["a"], k=3) == pytest.approx(0.0)


# --- answer: token-F1 -------------------------------------------------------

def test_token_f1_exact_after_normalization():
    assert metrics.token_f1("The cat sat.", "the cat sat") == pytest.approx(1.0)


def test_token_f1_partial():
    # pred={cat}, gold={cat,sat}: P=1, R=0.5, F1 = 2/3
    assert metrics.token_f1("cat", "the cat sat") == pytest.approx(2 / 3)


def test_token_f1_both_empty():
    assert metrics.token_f1("", "") == pytest.approx(1.0)


def test_token_f1_one_empty():
    assert metrics.token_f1("", "something") == pytest.approx(0.0)


def test_token_f1_disjoint():
    assert metrics.token_f1("dog", "cat") == pytest.approx(0.0)


# --- answer: exact-match ----------------------------------------------------

def test_exact_match_normalized_equal():
    assert metrics.exact_match("The Cat.", "the cat") == pytest.approx(1.0)


def test_exact_match_unequal():
    assert metrics.exact_match("dog", "cat") == pytest.approx(0.0)


def test_best_over_golds():
    assert metrics.best_over_golds(metrics.token_f1, "cat", ["dog", "cat"]) == pytest.approx(1.0)


# --- variance aggregation ---------------------------------------------------

def test_aggregate_runs_multi():
    stats = metrics.aggregate_runs([0.5, 0.7, 0.6])
    assert stats.mean == pytest.approx(0.6)
    assert stats.stddev == pytest.approx(math.sqrt(0.02 / 3))
    assert stats.n_runs == 3


def test_aggregate_runs_single():
    stats = metrics.aggregate_runs([0.5])
    assert (stats.mean, stats.stddev, stats.n_runs) == (pytest.approx(0.5), 0.0, 1)


def test_aggregate_runs_empty():
    stats = metrics.aggregate_runs([])
    assert (stats.mean, stats.stddev, stats.n_runs) == (0.0, 0.0, 0)


# --- systems ----------------------------------------------------------------

def test_token_cost_usd():
    assert metrics.token_cost_usd(1_000_000, 0) == pytest.approx(1.0)
    assert metrics.token_cost_usd(0, 1_000_000) == pytest.approx(5.0)


def test_storage_footprint_bytes():
    assert metrics.storage_footprint_bytes(["ab", "c"]) == 3


def test_measure_latency_positive():
    with metrics.measure_latency() as t:
        sum(range(1000))
    assert t[0] >= 0.0

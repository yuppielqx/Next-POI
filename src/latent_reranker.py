"""Latent-intent reranker trained on prior-bank retrieval."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    EMBEDDING_DIM,
    PRIOR_BANK_TOP_K,
    RERANKER_BATCH_SIZE,
    RERANKER_HIDDEN_DIM,
    RERANKER_LR,
    RERANKER_MODEL_PATH,
    RERANKER_NEGATIVES,
)
from src.embedding_utils import embed_texts
from src.prior_bank import build_or_load_poi_embeddings, build_prefix_text, build_prior_bank, load_prior_bank_index
from src.utils import haversine, logger


FEATURE_DIM = 11
MAX_PREFIX_EXAMPLES_PER_TRAJ = 6
MIN_PREFIX_LEN = 2


def _poi_loc_to_index(poi_meta: dict) -> dict[int, int]:
    return poi_meta["loc_id_to_index"]


def _poi_vectors(poi_meta: dict) -> np.ndarray:
    return poi_meta["vectors"]


def _poi_categories(poi_meta: dict) -> list[str]:
    return poi_meta["categories"]


def _poi_popularity(poi_meta: dict) -> np.ndarray:
    return poi_meta["popularity"]


def _poi_temporal_stats(poi_meta: dict) -> dict[int, dict]:
    return poi_meta.get("temporal_stats", {})


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        return vec / max(np.linalg.norm(vec), 1e-12)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vec / norms


class LatentIntentReranker(nn.Module):
    def __init__(self, embed_dim: int = EMBEDDING_DIM, hidden_dim: int = RERANKER_HIDDEN_DIM, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.prefix_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        fused_dim = hidden_dim * 4 + hidden_dim
        self.scorer = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, prefix_emb: torch.Tensor, candidate_emb: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        p = self.prefix_proj(prefix_emb)
        c = self.candidate_proj(candidate_emb)
        f = self.feature_proj(features)
        x = torch.cat([p, c, p * c, torch.abs(p - c), f], dim=-1)
        return self.scorer(x).squeeze(-1)


@dataclass
class RerankerArtifacts:
    model: LatentIntentReranker
    poi_vectors: np.ndarray
    poi_loc_ids: list[int]
    poi_loc_to_index: dict[int, int]
    poi_popularity: np.ndarray
    poi_categories: list[str]
    poi_temporal_stats: dict[int, dict]
    prior_index: object
    device: torch.device


def build_prefix_embedding(context: list[dict], data_loader) -> np.ndarray:
    return embed_texts([build_prefix_text(context, data_loader)])[0]


def _load_model(payload: dict) -> LatentIntentReranker:
    model = LatentIntentReranker(
        embed_dim=payload["embed_dim"],
        hidden_dim=payload["hidden_dim"],
        feature_dim=payload["feature_dim"],
    )
    model.load_state_dict(payload["state_dict"])
    return model


def load_or_build_artifacts(data_loader, force: bool = False) -> RerankerArtifacts | None:
    if not force and RERANKER_MODEL_PATH.exists():
        try:
            payload = torch.load(RERANKER_MODEL_PATH, map_location="cpu")
            model = _load_model(payload)
            poi_artifacts = build_or_load_poi_embeddings(data_loader, force=False)
            prior_index = load_prior_bank_index()
            if prior_index is None:
                return None
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            return RerankerArtifacts(
                model=model,
                poi_vectors=poi_artifacts["vectors"],
                poi_loc_ids=poi_artifacts["loc_ids"],
                poi_loc_to_index=poi_artifacts["loc_id_to_index"],
                poi_popularity=poi_artifacts["popularity"],
                poi_categories=poi_artifacts["categories"],
                poi_temporal_stats=poi_artifacts.get("temporal_stats", {}),
                prior_index=prior_index,
                device=device,
            )
        except Exception as exc:
            logger.warning(f"Failed to load reranker artifacts: {exc}")
            return None
    return None


def save_artifacts(model: LatentIntentReranker) -> None:
    RERANKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embed_dim": EMBEDDING_DIM,
        "hidden_dim": RERANKER_HIDDEN_DIM,
        "feature_dim": FEATURE_DIM,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, RERANKER_MODEL_PATH)
    logger.info(f"Saved latent reranker to {RERANKER_MODEL_PATH}")


def _candidate_category_rate(candidate_category: str, recent_categories: list[str]) -> float:
    if not recent_categories:
        return 0.0
    return sum(1 for cat in recent_categories if cat.lower() == candidate_category.lower()) / float(len(recent_categories))


def build_feature_vector(
    candidate: dict,
    context: list[dict],
    retrieval_support: dict[int, dict[str, float]],
    poi_meta: dict,
    target_checkin: dict | None,
    data_loader,
) -> np.ndarray:
    last = context[-1]
    candidate_loc = int(candidate["loc_id"])
    idx = _poi_loc_to_index(poi_meta)[candidate_loc]
    candidate_category = _poi_categories(poi_meta)[idx]
    last_category = data_loader.get_poi_category(last["loc_id"])
    recent_categories = [data_loader.get_poi_category(c["loc_id"]) for c in context[-3:]]
    support = retrieval_support.get(candidate_loc, {"count": 0.0, "max_sim": -1e9, "mean_sim": -1e9})
    revisit_count = float(sum(1 for c in context if c["loc_id"] == candidate_loc))
    cand_lon, cand_lat = data_loader.id2loc[candidate_loc]
    dist_km = haversine(last["lon"], last["lat"], cand_lon, cand_lat)

    temporal = _poi_temporal_stats(poi_meta).get(candidate_loc, {})
    total = float(temporal.get("total", 0.0))
    dow_counts = temporal.get("by_dow", {})
    hb_counts = temporal.get("by_hb", {})
    temporal_ref = target_checkin or last
    dow_key = str(temporal_ref.get("weekday", 0))
    hb_key = str(temporal_ref.get("hour", 0) // 3)
    dow_prob = float(dow_counts.get(dow_key, 0.0)) / total if total > 0 else 0.0
    hb_prob = float(hb_counts.get(hb_key, 0.0)) / total if total > 0 else 0.0

    return np.array([
        dist_km,
        math.log1p(float(_poi_popularity(poi_meta)[idx])),
        1.0 if candidate_category and last_category and candidate_category.lower() == last_category.lower() else 0.0,
        _candidate_category_rate(candidate_category, recent_categories),
        dow_prob,
        hb_prob,
        float(support.get("count", 0.0)),
        float(support.get("max_sim", -1e9)),
        float(support.get("mean_sim", -1e9)),
        revisit_count,
        min(float(len(context)), 10.0) / 10.0,
    ], dtype=np.float32)


def score_candidates_with_model(
    model: LatentIntentReranker,
    prefix_emb: np.ndarray,
    candidates: list[dict],
    context: list[dict],
    data_loader,
    poi_artifacts: dict,
    prior_index,
    target_checkin: dict | None = None,
    top_k_prior: int = PRIOR_BANK_TOP_K,
    exclude_traj_id: str | None = None,
    device: torch.device | None = None,
) -> list[dict]:
    if device is None:
        device = next(model.parameters()).device

    prefix_emb = _normalize(prefix_emb)
    retrieval = prior_index.retrieve(prefix_emb, top_k=top_k_prior, exclude_traj_id=exclude_traj_id)
    scored = []
    for cand in candidates:
        loc_id = int(cand["loc_id"])
        idx = _poi_loc_to_index(poi_artifacts)[loc_id]
        cand_emb = _poi_vectors(poi_artifacts)[idx]
        feats = build_feature_vector(
            cand,
            context,
            retrieval["support_by_loc"],
            poi_artifacts,
            target_checkin,
            data_loader,
        )
        with torch.no_grad():
            score = model(
                torch.tensor(prefix_emb[None, :], dtype=torch.float32, device=device),
                torch.tensor(cand_emb[None, :], dtype=torch.float32, device=device),
                torch.tensor(feats[None, :], dtype=torch.float32, device=device),
            ).item()
        cand = dict(cand)
        support = retrieval["support_by_loc"].get(loc_id, {})
        cand["reranker_score"] = float(score)
        cand["retrieval_support_count"] = float(support.get("count", 0.0))
        cand["retrieval_support_max_sim"] = float(support.get("max_sim", -1e9))
        cand["retrieval_support_mean_sim"] = float(support.get("mean_sim", -1e9))
        scored.append(cand)

    scored.sort(key=lambda x: x["reranker_score"], reverse=True)
    return scored


def _evenly_spaced_prefixes(traj: list[dict], max_examples: int = MAX_PREFIX_EXAMPLES_PER_TRAJ) -> list[tuple[list[dict], dict]]:
    if len(traj) <= MIN_PREFIX_LEN:
        return []
    end_positions = list(range(MIN_PREFIX_LEN, len(traj)))
    if len(end_positions) <= max_examples:
        chosen = end_positions
    else:
        idxs = np.linspace(0, len(end_positions) - 1, num=max_examples, dtype=int)
        chosen = [end_positions[i] for i in idxs]
    examples = []
    for end in chosen:
        examples.append((traj[:end], traj[end]))
    return examples


def _candidate_pool_by_category(poi_artifacts: dict) -> dict[str, list[int]]:
    by_cat: dict[str, list[int]] = {}
    for lid, cat in zip(poi_artifacts["loc_ids"], poi_artifacts["categories"]):
        by_cat.setdefault(cat.lower(), []).append(int(lid))
    return by_cat


def _sample_negatives(
    all_loc_ids: list[int],
    target_loc_id: int,
    target_category: str,
    prefix: list[dict],
    data_loader,
    n: int,
    prior_index,
    prefix_vec: np.ndarray,
    poi_artifacts: dict,
    by_category: dict[str, list[int]],
) -> list[int]:
    import random

    rng = random.Random(42)
    seen = {target_loc_id}
    negatives: list[int] = []

    same_cat = list(by_category.get(target_category.lower(), []))
    rng.shuffle(same_cat)
    for lid in same_cat:
        if lid not in seen:
            negatives.append(lid)
            seen.add(lid)
        if len(negatives) >= max(1, n // 3):
            break

    if prefix:
        last = prefix[-1]
        nearby = data_loader.get_nearby_pois(last["lon"], last["lat"], top_n=min(40, len(all_loc_ids)))
        for lid, _ in nearby:
            if lid not in seen:
                negatives.append(lid)
                seen.add(lid)
            if len(negatives) >= max(1, (2 * n) // 3):
                break

    if prior_index is not None:
        retrieved = prior_index.retrieve(prefix_vec, top_k=min(PRIOR_BANK_TOP_K, len(all_loc_ids)))
        hard = list(retrieved["support_by_loc"].keys())
        rng.shuffle(hard)
        for lid in hard:
            if lid not in seen:
                negatives.append(lid)
                seen.add(lid)
            if len(negatives) >= n:
                break

    while len(negatives) < n:
        lid = rng.choice(all_loc_ids)
        if lid not in seen:
            negatives.append(lid)
            seen.add(lid)

    return negatives[:n]


def train_reranker(
    data_loader,
    epochs: int = 4,
    batch_size: int = RERANKER_BATCH_SIZE,
    negatives: int = RERANKER_NEGATIVES,
    force_rebuild: bool = False,
) -> dict:
    """Train the learned reranker on multi-prefix supervision."""
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poi_artifacts = build_or_load_poi_embeddings(data_loader, force=force_rebuild)
    prior_payload = build_prior_bank(data_loader, force=force_rebuild)
    prior_index = load_prior_bank_index(prior_payload)
    by_category = _candidate_pool_by_category(poi_artifacts)

    examples = []
    prefix_texts = []
    for tid in data_loader.train_traj_ids:
        traj = data_loader.trips[tid]
        for prefix, target in _evenly_spaced_prefixes(traj):
            examples.append({
                "traj_id": tid,
                "prefix": prefix,
                "target_loc_id": target["loc_id"],
                "target_category": data_loader.get_poi_category(target["loc_id"]),
                "target_checkin": target,
            })
            prefix_texts.append(build_prefix_text(prefix, data_loader))

    if not examples:
        raise RuntimeError("No training examples available for reranker.")

    prefix_embs = embed_texts(prefix_texts)
    all_loc_ids = list(poi_artifacts["loc_ids"])
    poi_vectors = poi_artifacts["vectors"]
    loc_to_idx = poi_artifacts["loc_id_to_index"]

    model = LatentIntentReranker(embed_dim=prefix_embs.shape[1], hidden_dim=RERANKER_HIDDEN_DIM, feature_dim=FEATURE_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=RERANKER_LR)
    rng = random.Random(42)

    for epoch in range(1, epochs + 1):
        order = list(range(len(examples)))
        rng.shuffle(order)
        model.train()
        total_loss = 0.0
        total = 0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            batch_prefix = prefix_embs[batch_idx]
            batch_examples = [examples[i] for i in batch_idx]

            cand_emb_batches = []
            feat_batches = []
            labels = []
            for ex, prefix_vec in zip(batch_examples, batch_prefix):
                prefix = ex["prefix"]
                target = ex["target_loc_id"]
                target_category = ex["target_category"]
                cand_ids = [target] + _sample_negatives(
                    all_loc_ids,
                    target,
                    target_category,
                    prefix,
                    data_loader,
                    negatives,
                    prior_index,
                    prefix_vec,
                    poi_artifacts,
                    by_category,
                )
                retrieval = prior_index.retrieve(prefix_vec, top_k=PRIOR_BANK_TOP_K, exclude_traj_id=ex["traj_id"])
                cand_emb_rows = []
                feat_rows = []
                for lid in cand_ids:
                    idx = loc_to_idx[lid]
                    cand_emb_rows.append(poi_vectors[idx])
                    cand = {"loc_id": lid}
                    feat_rows.append(
                        build_feature_vector(
                            cand,
                            prefix,
                            retrieval["support_by_loc"],
                            poi_artifacts,
                            ex.get("target_checkin"),
                            data_loader,
                        )
                    )
                cand_emb_batches.append(cand_emb_rows)
                feat_batches.append(feat_rows)
                labels.append(0)

            prefix_t = torch.tensor(batch_prefix, dtype=torch.float32, device=device)
            cand_t = torch.tensor(np.asarray(cand_emb_batches), dtype=torch.float32, device=device)
            feat_t = torch.tensor(np.asarray(feat_batches), dtype=torch.float32, device=device)
            label_t = torch.tensor(labels, dtype=torch.long, device=device)

            bsz, num_cands, _ = cand_t.shape
            scores = model(
                prefix_t[:, None, :].expand(-1, num_cands, -1).reshape(-1, prefix_t.shape[-1]),
                cand_t.reshape(-1, cand_t.shape[-1]),
                feat_t.reshape(-1, feat_t.shape[-1]),
            ).reshape(bsz, num_cands)
            loss = F.cross_entropy(scores, label_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * bsz
            total += bsz

        logger.info(f"Epoch {epoch}/{epochs} loss={total_loss / max(total, 1):.4f}")

    save_artifacts(model.cpu())
    return {"train_examples": len(examples), "epochs": epochs}

"""Entraînement d'un run : une perte, des réglages, une graine.

Le réseau entier est entraîné, BatchNorm gelés (protocole, sections 4 et 7).
Une époque = ``num_batches`` lots de 64 personnes × 4 images. À la fin de chaque
époque : MAP@R sur la validation, meilleure époque gardée, point de reprise
écrit. Un run coupé reprend à l'époque suivante, avec les mêmes lots et les
mêmes tirages d'augmentation que s'il n'avait pas été coupé.

Fichiers d'un run (``run_dir``) :

- ``params.json``  : perte, réglages, graine, nombre d'époques prévu ;
- ``last.pt``      : point de reprise (réseau, poids de la perte, optimiseur…),
  supprimé à la fin du run ;
- ``best.pt``      : réseau à la meilleure époque (gardé si ``keep_weights``) ;
- ``history.csv``  : une ligne par époque ;
- ``result.json``  : résumé final ; sa présence marque un run terminé.
"""

from __future__ import annotations

import math
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from facegeom.data.augment import GpuAugment
from facegeom.data.batches import CachedBatches, augment_generator, batch_loader, epoch_batches
from facegeom.eval.retrieval import retrieval_per_query, summarize
from facegeom.losses import WEIGHTED, compute_loss, make_loss, margin_at, set_arcface_margin
from facegeom.models.network import EmbeddingNet, autocast_dtype, embed_images, load_backbone
from facegeom.utils import read_json, resolve_workers, save_torch, set_seed, write_json


class ConfigMismatchError(RuntimeError):
    """Un run existant a été lancé avec d'autres réglages."""


@dataclass
class TrainData:
    """Données d'un entraînement.

    ``train_labels`` : indices de classes 0..C−1, dans l'ordre du cache.
    ``val_images`` : cache de validation (memmap) ; ``val_labels`` : identités.
    """

    train_cache: str
    train_labels: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray

    @property
    def n_classes(self) -> int:
        return int(self.train_labels.max()) + 1


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Facteur du taux d'apprentissage au pas ``step`` (0, 1, …).

    Montée linéaire pendant ``warmup_steps`` pas, puis décroissance en cosinus
    jusqu'à 0 à ``total_steps`` (les deux s'appliquent en même temps : on prend
    le plus petit des deux facteurs, comme dans l'essai du 17 novembre).
    """
    warm = (step + 1) / warmup_steps if warmup_steps > 0 else 1.0
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(step / total_steps, 1.0)))
    return min(warm, cosine)


def build_optimizer(model: EmbeddingNet, loss_fn, loss_name: str, params: dict,
                    tcfg: dict) -> torch.optim.Optimizer:
    """SGD avec trois groupes : réseau (taux de la tête × rapport), tête, poids de la perte."""
    ocfg = tcfg["optimizer"]
    lr = float(params["lr"])
    groups = [
        {"name": "backbone", "lr": lr * float(ocfg["backbone_lr_ratio"]),
         "params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"name": "head", "lr": lr, "params": list(model.head.parameters())},
    ]
    if loss_name in WEIGHTED:
        groups.append({"name": "loss_weights", "lr": float(params["lr_weights"]),
                       "params": list(loss_fn.parameters())})
    return torch.optim.SGD(groups, lr=lr, momentum=float(ocfg["momentum"]),
                           nesterov=bool(ocfg["nesterov"]),
                           weight_decay=float(ocfg["weight_decay"]))


RECIPE_KEYS = ("backbone", "head", "batch", "optimizer", "schedule",
               "arcface_margin_warmup_epochs", "augment", "early_stopping", "grad_clip_norm")


def run_identity(tcfg: dict, dcfg: dict, loss_name: str, params: dict, seed: int, epochs: int,
                 schedule_epochs: int, max_steps_per_epoch: int | None) -> dict:
    """Ce qui définit un run : perte, réglages, graine, durées et recette d'entraînement.

    Écrit dans ``params.json`` ; un run existant n'est repris (ou réutilisé) que
    si tout est identique. La recette comprend les sections de ``train.yaml``
    qui changent l'entraînement, et la préparation des images.
    """
    identity = {
        "loss": loss_name, "params": {k: float(v) for k, v in params.items()},
        "seed": int(seed), "epochs": int(epochs), "schedule_epochs": int(schedule_epochs),
        "recipe": {**{k: tcfg.get(k) for k in RECIPE_KEYS},  # clé absente = None (anciens fichiers)
                   "input_size": dcfg["input_size"], "normalize": dcfg["normalize"]},
    }
    if max_steps_per_epoch:
        identity["max_steps_per_epoch"] = int(max_steps_per_epoch)  # essai rapide
    return identity


def _same_identity(stored: dict, expected: dict) -> bool:
    """Égalité des identités, à l'arrondi près pour les réglages tirés au hasard."""
    if set(stored) != set(expected) or stored["params"].keys() != expected["params"].keys():
        return False
    close = all(abs(stored["params"][k] - v) <= 1e-12 * max(1.0, abs(v))
                for k, v in expected["params"].items())
    rest = {k: v for k, v in stored.items() if k != "params"}
    return close and rest == {k: v for k, v in expected.items() if k != "params"}


class RunJournal:
    """Messages d'un run : affichés et ajoutés à ``journal.txt`` dans son dossier."""

    def __init__(self, run_dir: Path, log: Callable[[str], None]):
        self.path = run_dir / "journal.txt"
        self.log = log

    def __call__(self, msg: str) -> None:
        self.log(msg)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def train_run(
    run_dir: str | Path,
    tcfg: dict,
    dcfg: dict,
    loss_name: str,
    params: dict,
    seed: int,
    epochs: int,
    data: TrainData,
    device: str,
    pretrained: bool = True,
    keep_weights: bool = True,
    log: Callable[[str], None] = print,
    heartbeat: Callable[[], None] | None = None,
    max_steps_per_epoch: int | None = None,
    schedule_epochs: int | None = None,
) -> dict:
    """Entraîne (ou reprend, ou saute s'il est fini) un run ; renvoie son résumé.

    ``tcfg`` : ``configs/train.yaml`` ; ``dcfg`` : ``configs/data.yaml``.
    ``schedule_epochs`` : durée sur laquelle le taux décroît en cosinus (défaut :
    ``epochs``) ; le pilote fait 1 époque d'un calendrier de réglage complet.
    ``max_steps_per_epoch`` : époques raccourcies, pour un essai rapide seulement
    (noté dans ``params.json`` ; jamais pour un résultat du protocole).
    Une erreur pendant le run est écrite dans ``journal.txt`` avant d'être relancée.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    schedule_epochs = int(schedule_epochs or epochs)
    expected = run_identity(tcfg, dcfg, loss_name, params, seed, epochs, schedule_epochs,
                            max_steps_per_epoch)
    if (run_dir / "params.json").exists():
        if not _same_identity(read_json(run_dir / "params.json"), expected):
            raise ConfigMismatchError(
                f"{run_dir} existe avec d'autres réglages ou une autre recette "
                "(voir params.json). Utiliser un nouveau dossier (--new)."
            )
    else:
        write_json(run_dir / "params.json", expected)
    if (run_dir / "result.json").exists():
        result = read_json(run_dir / "result.json")
        log(f"{run_dir.name} : déjà terminé (MAP@R {result.get('best_map_at_r')})")
        return result

    journal = RunJournal(run_dir, log)
    try:
        return _train(run_dir, tcfg, dcfg, loss_name, params, seed, epochs, schedule_epochs, data,
                      device, pretrained, keep_weights, journal, heartbeat, max_steps_per_epoch,
                      expected)
    except Exception:
        journal(f"ERREUR, run interrompu :\n{traceback.format_exc()}")
        raise


def _train(run_dir, tcfg, dcfg, loss_name, params, seed, epochs, schedule_epochs, data, device,
           pretrained, keep_weights, log, heartbeat, max_steps_per_epoch, expected) -> dict:
    """Corps de ``train_run`` (mêmes paramètres)."""
    dev = torch.device(device)
    amp_dtype = autocast_dtype(dev, tcfg.get("amp", "auto"))
    channels_last = bool(tcfg.get("channels_last", False)) and dev.type == "cuda"
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    bcfg, hcfg, acfg = tcfg["batch"], tcfg["head"], tcfg["augment"]
    batch_size = int(bcfg["classes_per_batch"]) * int(bcfg["images_per_class"])
    steps_per_epoch = len(data.train_labels) // batch_size
    if max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, int(max_steps_per_epoch))
    total_steps = steps_per_epoch * schedule_epochs
    warmup_steps = round(float(tcfg["schedule"]["warmup_epochs"]) * steps_per_epoch)
    margin_warmup = float(tcfg["arcface_margin_warmup_epochs"])
    checkpoint_every = int(tcfg.get("checkpoint_every", 1))
    clip_norm = float(tcfg["grad_clip_norm"]) if tcfg.get("grad_clip_norm") else None

    set_seed(seed)
    model = EmbeddingNet(load_backbone(tcfg["backbone"], pretrained), hcfg["in_dim"],
                         hcfg["out_dim"]).to(dev)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    loss_fn, miner = make_loss(loss_name, params, data.n_classes, hcfg["out_dim"])
    loss_fn = loss_fn.to(dev)
    optimizer = build_optimizer(model, loss_fn, loss_name, params, tcfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda t: lr_factor(t, warmup_steps, total_steps)
    )
    scaler = torch.amp.GradScaler("mps" if dev.type == "mps" else "cuda",
                                  enabled=amp_dtype == torch.float16)
    augment = GpuAugment(dcfg["input_size"], dcfg["normalize"]["mean"], dcfg["normalize"]["std"],
                         acfg["scale"], acfg["ratio"], acfg["flip_p"], dev)
    precision = str(amp_dtype).replace("torch.", "") if amp_dtype else "float32"

    history: list[dict] = []
    best = {"map_at_r": -1.0, "epoch": 0}
    bad_epochs, start_epoch, elapsed = 0, 1, 0.0
    if (run_dir / "last.pt").exists():
        ckpt = torch.load(run_dir / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        loss_fn.load_state_dict(ckpt["loss"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler.is_enabled() and ckpt["scaler"]:  # précision changée de session en session
            scaler.load_state_dict(ckpt["scaler"])
        history, best, bad_epochs = ckpt["history"], ckpt["best"], ckpt["bad_epochs"]
        start_epoch, elapsed = ckpt["epoch"] + 1, ckpt["elapsed"]
        log(f"{run_dir.name} : reprise à l'époque {start_epoch} ({precision})")
    else:
        log(f"{run_dir.name} : début, {steps_per_epoch} pas par époque, {precision}, {device}")

    patience = int(tcfg["early_stopping"]["patience"])
    status = "done"
    num_workers = resolve_workers(tcfg.get("num_workers", 0))
    for epoch in range(start_epoch, int(epochs) + 1):
        if bad_epochs >= patience:
            break
        t0 = time.perf_counter()
        model.train()
        batches = epoch_batches(data.train_labels, int(bcfg["images_per_class"]), batch_size,
                                steps_per_epoch, seed, epoch)
        loader = batch_loader(CachedBatches(data.train_cache, batches, data.train_labels),
                              num_workers, pin_memory=dev.type == "cuda")
        gen = augment_generator(seed, epoch)
        loss_sum = torch.zeros((), device=dev)
        for i, (images, labels) in enumerate(loader):
            step = (epoch - 1) * steps_per_epoch + i
            if loss_name == "arcface":
                set_arcface_margin(loss_fn, margin_at(step, steps_per_epoch, margin_warmup,
                                                      params["m"]))
            x = augment.train(images, gen)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            labels = labels.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                emb = model(x)
            loss = compute_loss(loss_fn, miner, emb.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [q for g in optimizer.param_groups for q in g["params"]], clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.detach()
            if (i + 1) % 50 == 0 or i + 1 == steps_per_epoch:
                if not torch.isfinite(loss_sum):
                    status = "diverged"
                    break
                if heartbeat:
                    heartbeat()
            if (i + 1) % 200 == 0:
                rate = (i + 1) * batch_size / (time.perf_counter() - t0)
                log(f"   époque {epoch} pas {i + 1}/{steps_per_epoch} : perte "
                    f"{loss_sum.item() / (i + 1):.4f}, {rate:.0f} images/s")
        if status == "diverged":
            log(f"{run_dir.name} époque {epoch} : perte non finie, arrêt du run (divergé)")
            break
        t_train = time.perf_counter() - t0

        try:
            val = embed_images(model, data.val_images, augment, int(tcfg["eval_batch_size"]),
                               amp_dtype, channels_last)
        except FloatingPointError as err:  # réseau détruit : run consigné comme divergé
            status = "diverged"
            log(f"{run_dir.name} époque {epoch} : {err}, arrêt du run (divergé)")
            break
        metrics = summarize(retrieval_per_query(val, data.val_labels))
        t_eval = time.perf_counter() - t0 - t_train
        elapsed += t_train + t_eval
        row = {"epoch": epoch, "train_loss": loss_sum.item() / steps_per_epoch,
               "lr_head": optimizer.param_groups[1]["lr"], **metrics,
               "seconds_train": round(t_train, 1), "seconds_eval": round(t_eval, 1),
               "precision": precision}
        if loss_name == "arcface":
            row["margin"] = float(loss_fn.margin)
        history.append(row)
        log(f"{run_dir.name} époque {epoch:2d} : perte {row['train_loss']:.4f}  MAP@R val "
            f"{metrics['map_at_r']:.2f}  P@1 {metrics['p_at_1']:.2f}  "
            f"({t_train:.0f} s + {t_eval:.0f} s)")

        if metrics["map_at_r"] > best["map_at_r"]:
            best = {"map_at_r": metrics["map_at_r"], "epoch": epoch, **metrics}
            bad_epochs = 0
            if keep_weights:
                save_torch({k: v.detach().cpu() for k, v in model.state_dict().items()},
                           run_dir / "best.pt")
        else:
            bad_epochs += 1
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if epoch % checkpoint_every == 0 and epoch < int(epochs) and bad_epochs < patience:
            save_torch({"epoch": epoch, "model": model.state_dict(), "loss": loss_fn.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "history": history, "best": best,
                        "bad_epochs": bad_epochs, "elapsed": elapsed}, run_dir / "last.pt")
        if heartbeat:
            heartbeat()
        if bad_epochs >= patience:
            log(f"{run_dir.name} : arrêt, {patience} époques sans progrès")

    n_done = len(history)
    result = {
        **{k: v for k, v in expected.items() if k != "recipe"},
        "status": status,
        "precision": precision,
        "best_epoch": best["epoch"],
        "best_map_at_r": best["map_at_r"] if best["epoch"] else None,
        "best_p_at_1": best.get("p_at_1"),
        "best_r_precision": best.get("r_precision"),
        "steps_per_epoch": steps_per_epoch,
        "epochs_run": n_done,
        "stopped_early": n_done < int(epochs) and status == "done",
        "seconds_total": round(elapsed),
        "seconds_per_epoch_train": round(float(np.mean([h["seconds_train"] for h in history])), 1)
        if history else None,
        "seconds_per_epoch_eval": round(float(np.mean([h["seconds_eval"] for h in history])), 1)
        if history else None,
    }
    write_json(run_dir / "result.json", result)
    log(f"{run_dir.name} : terminé ({status}), meilleur MAP@R {result['best_map_at_r']} "
        f"à l'époque {result['best_epoch']}")
    (run_dir / "last.pt").unlink(missing_ok=True)
    return result


def load_trained(run_dir: str | Path, tcfg: dict, device: str = "cpu") -> EmbeddingNet:
    """Réseau d'un run terminé, à sa meilleure époque, en mode évaluation."""
    run_dir = Path(run_dir)
    hcfg = tcfg["head"]
    model = EmbeddingNet(load_backbone(tcfg["backbone"], pretrained=False), hcfg["in_dim"],
                         hcfg["out_dim"])
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval()

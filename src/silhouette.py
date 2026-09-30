# # -*- coding: utf-8 -*-
#
# """
# Created on 25. 08. 2026 at 20:59:30
#
# Author: Richard Redina
# Email: 195715@vut.cz
# Affiliation:
#          International Clinical Research Center, Brno
#          Brno University of Technology, Brno
# GitHub: RicRedi
#
# (._.)
#  <|>
# _/|_
#
# Description:
#     Silhouetová analýza kvality shlukování.
#
#     Měří, jak dobře každý bod zapadá do svého shluku v porovnání s nejbližším
#     sousedním shlukem. Pracuje s tvrdými popisky — kompatibilní s k-means i FCM.
# """
#
# from __future__ import annotations
#
# from typing import TYPE_CHECKING
#
# import numpy as np
#
# if TYPE_CHECKING:
#     from src.distance import Distance
#
#
# def silhouette_samples(
#     x: np.ndarray,
#     labels: np.ndarray,
#     distance: "Distance",
# ) -> np.ndarray:
#     """Vypočítá silhouetovou hodnotu pro každý bod datasetu.
#
#     Úkol:
#         Implementujte výpočet silhouetové hodnoty ``s(i)`` pro každý bod ``i``:
#
#         .. math::
#             s(i) = \\frac{b(i) - a(i)}{\\max(a(i),\\, b(i))}
#
#         kde:
#
#         - ``a(i)`` = průměrná vzdálenost bodu ``i`` od všech ostatních bodů
#           ve **stejném** shluku (míra soudržnosti),
#         - ``b(i)`` = průměrná vzdálenost bodu ``i`` od všech bodů v **nejbližším
#           jiném** shluku (míra oddělenosti).
#
#         Postup:
#         1. Pro každý bod ``i`` identifikujte jeho shluk ``c = labels[i]``.
#         2. Spočítejte ``a(i)`` jako průměr vzdáleností ke všem ostatním bodům
#            se stejným popiskem (vylučte bod ``i`` samotný).
#         3. Pro každý jiný shluk ``c'`` spočítejte průměrnou vzdálenost bodu
#            ``i`` od všech bodů shluku ``c'``. Hodnota ``b(i)`` je minimum
#            těchto průměrů přes všechny ``c' ≠ c``.
#         4. Vraťte ``(b(i) - a(i)) / max(a(i), b(i))``.
#
#         Speciální případ: pokud shluk obsahuje jen jeden bod, nastavte ``s(i) = 0``.
#
#         Pro výpočet vzdáleností volejte ``distance.calculate(X[i], X[j])`` —
#         zajišťuje konzistenci s metrikou použitou při shlukování.
#
#     Parameters
#     ----------
#     x:
#         Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
#     labels:
#         Tvrdé popisky shluků, tvar ``(n_bodů,)``, hodnoty 0 … k-1.
#     distance:
#         Instance metriky vzdálenosti — stejná jako při shlukování.
#
#     Returns
#     -------
#     np.ndarray
#         Silhouetové hodnoty, tvar ``(n_bodů,)``, hodnoty v [-1, 1].
#         Vyšší hodnota znamená lepší zařazení bodu do shluku.
#     """
#     # assert: Ověřte, že x je 2D matice, labels je 1D pole stejné délky
#     # a obsahuje alespoň 2 různé shluky
#     raise NotImplementedError(
#         "Úkol: implementujte silhouette_samples() — vypočítejte silhouetovou "
#         "hodnotu s(i) = (b(i) - a(i)) / max(a(i), b(i)) pro každý bod."
#     )
#
#
# def silhouette_score(
#     x: np.ndarray,
#     labels: np.ndarray,
#     distance: "Distance",
# ) -> float:
#     """Vypočítá průměrné silhouetové skóre přes všechny body.
#
#     Úkol:
#         Zavolejte ``silhouette_samples`` a vraťte průměr výsledného pole.
#         Průměrné skóre slouží jako jednočíselná míra kvality shlukování —
#         používá se pro výběr optimálního ``k``.
#
#     Parameters
#     ----------
#     x:
#         Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
#     labels:
#         Tvrdé popisky shluků, tvar ``(n_bodů,)``.
#     distance:
#         Instance metriky vzdálenosti.
#
#     Returns
#     -------
#     float
#         Průměrné silhouetové skóre v rozsahu [-1, 1].
#         Blíže k 1 → kvalitnější shlukování.
#     """
#     raise NotImplementedError(
#         "Úkol: implementujte silhouette_score() — vraťte průměr "
#         "výstupu silhouette_samples(X, labels, distance)."
#     )
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.distance import Distance


def silhouette_samples(
    x: np.ndarray,
    labels: np.ndarray,
    distance: "Distance",
) -> np.ndarray:

    if x.ndim != 2:
        raise ValueError(
            "x musí být 2D matice."
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels musí být 1D pole."
        )

    if len(labels) != len(x):
        raise ValueError(
            "Počet labels musí odpovídat počtu bodů."
        )

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError(
            "Musí existovat alespoň dva shluky."
        )

    n = x.shape[0]

    result = np.zeros(
        n,
        dtype=float,
    )

    for i in range(n):

        current_cluster = labels[i]

        # Body stejného shluku
        same_cluster = np.where(
            labels == current_cluster
        )[0]

        # Shluk obsahuje pouze tento bod
        if len(same_cluster) == 1:
            result[i] = 0.0
            continue

        # --------------------------------------------------
        # a(i) - průměrná vzdálenost uvnitř vlastního shluku
        # --------------------------------------------------

        own_distances = []

        for j in same_cluster:

            if j == i:
                continue

            d = distance.calculate(
                x[i],
                x[j],
            )

            own_distances.append(d)

        a = np.mean(own_distances)

        # --------------------------------------------------
        # b(i) - nejbližší jiný shluk
        # --------------------------------------------------

        other_cluster_distances = []

        for other_cluster in unique_labels:

            if other_cluster == current_cluster:
                continue

            other_indices = np.where(
                labels == other_cluster
            )[0]

            distances_to_cluster = []

            for j in other_indices:

                d = distance.calculate(
                    x[i],
                    x[j],
                )

                distances_to_cluster.append(d)

            mean_distance = np.mean(
                distances_to_cluster
            )

            other_cluster_distances.append(
                mean_distance
            )

        b = min(other_cluster_distances)

        # --------------------------------------------------
        # silhouetová hodnota
        # --------------------------------------------------

        denominator = max(a, b)

        if denominator == 0:
            result[i] = 0.0

        else:
            result[i] = (
                (b - a)
                / denominator
            )

    return result


def silhouette_score(
    x: np.ndarray,
    labels: np.ndarray,
    distance: "Distance",
) -> float:

    values = silhouette_samples(
        x,
        labels,
        distance,
    )

    return float(np.mean(values))

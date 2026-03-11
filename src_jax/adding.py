"""Adding algorithm: combine two layers into a composite layer."""

import jax.numpy as jnp


def _add_sources_general(top, bot, T_ba_D1, T_bc_D2):
    """Combine thermal and solar sources for the general adding case."""
    results = {}

    for suffix in ("", "_solar"):
        s1_up = top[f"s_up{suffix}"]
        s1_down = top[f"s_down{suffix}"]
        s2_up = bot[f"s_up{suffix}"]
        s2_down = bot[f"s_down{suffix}"]

        Rbc_s1m = bot["R_ab"] @ s1_down
        ans_up = s1_up + T_ba_D1 @ (s2_up + Rbc_s1m)

        Rba_s2p = top["R_ba"] @ s2_up
        ans_down = s2_down + T_bc_D2 @ (s1_down + Rba_s2p)

        results[f"s_up{suffix}"] = ans_up
        results[f"s_down{suffix}"] = ans_down

    return results


def _add_layers_general(top, bot):
    """General adding: both layers scattering."""
    N = top["R_ab"].shape[0]
    I_mat = jnp.eye(N)

    A1 = I_mat - bot["R_ab"] @ top["R_ba"]
    A2 = I_mat - top["R_ba"] @ bot["R_ab"]

    # T_ba_D1 = top.T_ba @ A1^{-1}  -> solve A1^T X^T = T_ba^T
    T_ba_D1 = jnp.linalg.solve(A1.T, top["T_ba"].T).T
    T_bc_D2 = jnp.linalg.solve(A2.T, bot["T_ab"].T).T

    R_ab = top["R_ab"] + T_ba_D1 @ bot["R_ab"] @ top["T_ab"]
    R_ba = bot["R_ba"] + T_bc_D2 @ top["R_ba"] @ bot["T_ba"]
    T_ab = T_bc_D2 @ top["T_ab"]
    T_ba = T_ba_D1 @ bot["T_ba"]

    sources = _add_sources_general(top, bot, T_ba_D1, T_bc_D2)

    return {
        "R_ab": R_ab, "R_ba": R_ba,
        "T_ab": T_ab, "T_ba": T_ba,
        "is_scattering": True,
        **sources,
    }


def _add_layers_nonscat_top(top, bot):
    """Optimized adding: non-scattering top layer."""
    N = top["R_ab"].shape[0]
    t = jnp.diag(top["T_ab"])  # diagonal transmission

    R_ab = t[:, None] * bot["R_ab"] * t[None, :]
    R_ba = bot["R_ba"]
    T_ab = bot["T_ab"] * t[None, :]
    T_ba = t[:, None] * bot["T_ba"]

    results = {
        "R_ab": R_ab, "R_ba": R_ba,
        "T_ab": T_ab, "T_ba": T_ba,
        "is_scattering": bot["is_scattering"],
    }

    for suffix in ("", "_solar"):
        s1_up = top[f"s_up{suffix}"]
        s1_down = top[f"s_down{suffix}"]
        s2_up = bot[f"s_up{suffix}"]
        s2_down = bot[f"s_down{suffix}"]

        Rbc_s1m = bot["R_ab"] @ s1_down
        results[f"s_up{suffix}"] = s1_up + t * (s2_up + Rbc_s1m)

        Tbc_s1m = bot["T_ab"] @ s1_down
        results[f"s_down{suffix}"] = s2_down + Tbc_s1m

    return results


def _add_layers_nonscat_bot(top, bot):
    """Optimized adding: non-scattering bottom layer."""
    N = top["R_ab"].shape[0]
    t = jnp.diag(bot["T_ab"])  # diagonal transmission

    R_ab = top["R_ab"]
    R_ba = t[:, None] * top["R_ba"] * t[None, :]
    T_ab = t[:, None] * top["T_ab"]
    T_ba = top["T_ba"] * t[None, :]

    results = {
        "R_ab": R_ab, "R_ba": R_ba,
        "T_ab": T_ab, "T_ba": T_ba,
        "is_scattering": top["is_scattering"],
    }

    for suffix in ("", "_solar"):
        s1_up = top[f"s_up{suffix}"]
        s1_down = top[f"s_down{suffix}"]
        s2_up = bot[f"s_up{suffix}"]
        s2_down = bot[f"s_down{suffix}"]

        Tba_s2p = top["T_ba"] @ s2_up
        results[f"s_up{suffix}"] = s1_up + Tba_s2p

        Rba_s2p = top["R_ba"] @ s2_up
        results[f"s_down{suffix}"] = s2_down + t * (s1_down + Rba_s2p)

    return results


def add_layers(top, bot):
    """Combine two layers, dispatching to the optimal variant.

    Args:
        top: Layer dict with R_ab, R_ba, T_ab, T_ba, s_up, s_down,
             s_up_solar, s_down_solar, is_scattering.
        bot: Same structure.

    Returns:
        Combined layer dict.
    """
    if not top["is_scattering"]:
        return _add_layers_nonscat_top(top, bot)
    if not bot["is_scattering"]:
        return _add_layers_nonscat_bot(top, bot)
    return _add_layers_general(top, bot)

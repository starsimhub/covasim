"""
Connectors for Covasim on the Starsim base.

``cv.CrossImmunity(ss.Connector)`` applies Covasim's **static, NAb-free** cross-immunity
(M3) to the single ``cv.COVID`` module. It runs at the connector loop slot -- after every
disease ``step_state`` and before transmission -- exactly where v3 ran ``check_immunity``.
Each step it reads every **ever-recovered** agent's ``recovered_variant`` and the asymmetric
``nv x nv`` cross-immunity matrix and writes the module's 2D ``sus_imm``/``symp_imm``/``sev_imm``
protection arrays, which ``cv.COVID.infect``/``set_prognoses`` then consume.

This is the M3 *static* form: the matrix is applied directly (no per-agent NAb level, no
``calc_VE`` logistic). NAb time-kinetics / waning layer on top of this in M4 (the connector is
the natural home for that engine -- "fold into cv.CrossImmunity"). Attaching the connector also
flips the module's ``cross_immunity_active`` flag on, enabling reinfection (recovery restores
susceptibility) without touching the public ``use_waning`` flag (reserved for M4).

**Documented divergence (M3):** because protection is the raw matrix value, a same-variant
challenge sees ``sus_imm = matrix[v, v] = 1.0`` => same-variant reinfection is exactly 0. v3's
literal NAb numerics (``calc_VE(nab x 1.0) < 1``) permit a tiny same-variant reinfection; this is
intended and recorded in the M3 baseline note.
"""
import numpy as np
import starsim as ss

from . import immunity as cvimm

__all__ = ['CrossImmunity']


class CrossImmunity(ss.Connector):
    """Static, NAb-free cross-immunity over the single cv.COVID module's variant axis.

    Args:
        immunity (array): optional explicit ``nv x nv`` cross-immunity matrix; default builds it
            from ``get_cross_immunity()`` via ``build_immunity_matrix`` (asymmetric, diagonal 1.0).
        disease (str): the disease module key to operate on (default ``'covid'``).
    """

    def __init__(self, pars=None, immunity=None, disease='covid', **kwargs):
        super().__init__()
        self.define_pars(
            immunity = immunity,  # optional override matrix; None => default cross-immunity
            disease  = disease,   # the cv.COVID module key in sim.diseases
        )
        self.update_pars(pars, **kwargs)
        self.matrix = None  # nv x nv, built in init_post
        return

    def _covid(self):
        """Resolve the (post-deepcopy) COVID module this connector operates on."""
        return self.sim.diseases[self.pars.disease]

    def init_post(self):
        """Build the cross-immunity matrix and switch the module into the reinfection regime."""
        super().init_post()
        covid = self._covid()
        self.matrix = cvimm.build_immunity_matrix(covid.variant_map, override=self.pars.immunity)
        covid.immunity_matrix = self.matrix     # expose on the module for backwards-compat reads
        covid.cross_immunity_active = True       # enable reinfection (Open Q B; use_waning reserved for M4)
        return

    def step(self):
        """Write per-variant protection for every ever-recovered agent from the matrix.

        Keyed off the **ever-recovered** mask (finite ``ti_recovered`` whose recovery date has
        passed -- v3's ``was_inf = t >= date_recovered``), NOT the transient ``recovered`` BoolState,
        so protection persists correctly across the reinfection window and (in M4) once
        time-since-recovery kinetics arrive. NAb-free: ``imm = matrix[target_variant, source_variant]``.
        """
        covid = self._covid()
        ti = covid.ti
        rec = (covid.ti_recovered <= ti).uids   # finite ti_recovered that has passed (NaN <= ti is False)
        if not len(rec):
            return
        ru = np.asarray(rec)
        src_v = np.asarray(covid.recovered_variant[rec]).astype(int)  # the variant each recovered from
        for v in range(covid.nv):                # target variant v; matrix[v, source] is the protection
            imm = self.matrix[v, src_v]
            covid.sus_imm[v, ru]  = imm
            covid.symp_imm[v, ru] = imm
            covid.sev_imm[v, ru]  = imm
        return

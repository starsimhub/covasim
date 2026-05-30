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

    @staticmethod
    def _advance_nab(covid, ti):
        """Step every agent's NAb level forward along the precomputed kinetic kernel (v3 update_nab).

        ``nab += nab_kin[ti − t_nab_event] × peak_nab``, clamped to ``[0, peak_nab]``. The kernel index
        is clamped to the kernel length so long horizons (or post-peak agents) stay in range.
        """
        nab_uids = (covid.t_nab_event <= ti).uids  # agents with a past NAb event (NaN <= ti is False)
        if not len(nab_uids):
            return
        t_since = (ti - np.asarray(covid.t_nab_event[nab_uids])).astype(int)
        t_since = np.clip(t_since, 0, len(covid.nab_kin) - 1)
        peak = np.asarray(covid.peak_nab[nab_uids])
        new = np.asarray(covid.nab[nab_uids]) + covid.nab_kin[t_since] * peak
        covid.nab[nab_uids] = np.clip(new, 0.0, peak)
        return

    def step(self):
        """Write per-variant protection (the v3 ``check_immunity`` slot). Two regimes:

          - ``use_waning=False`` (M2/M3): static, NAb-free -- ``imm = matrix[target, source]`` for every
            ever-recovered agent (finite ``ti_recovered ≤ ti`` -- v3's ``was_inf = t >= date_recovered``).
          - ``use_waning=True`` (M4/M6): advance NAb kinetics, then for each agent take
            ``imm = max(natural_cross_immunity, vaccine_efficacy)`` per variant and set
            ``sus_imm/symp_imm/sev_imm = calc_VE(nab × imm, axis)`` -- so natural + vaccine immunity share
            one NAb-weighted efficacy curve. With no vaccine registered the vaccine term is 0, so this is
            byte-identical to M4 (only ever-recovered agents get nonzero protection).
        """
        covid = self._covid()
        ti = covid.ti
        if not bool(covid.pars.use_waning):
            # M3 static path: matrix protection for ever-recovered agents only.
            rec = (covid.ti_recovered <= ti).uids
            if not len(rec):
                return
            src = np.asarray(covid.recovered_variant[rec])
            finite = np.isfinite(src)
            if not finite.any():
                return
            ru = np.asarray(rec[finite])
            src_v = src[finite].astype(int)
            for v in range(covid.nv):
                imm = self.matrix[v, src_v]
                covid.sus_imm[v, ru]  = imm
                covid.symp_imm[v, ru] = imm
                covid.sev_imm[v, ru]  = imm
            return

        # M4/M6 NAb-weighted path. Advance NAb kinetics, then compute protection over ALL active agents
        # (v3 check_immunity): natural cross-immunity OR vaccine efficacy, whichever is larger, x current NAb.
        self._advance_nab(covid, ti)
        nab_eff = covid.pars.nab_eff
        auids = covid.sim.people.auids
        ru = np.asarray(auids)
        nab_vals = np.asarray(covid.nab[auids])
        # Natural source variant per active agent (was_inf = finite ti_recovered<=ti with finite variant).
        tirec = np.asarray(covid.ti_recovered[auids])
        rvar  = np.asarray(covid.recovered_variant[auids])
        was_inf = np.isfinite(tirec) & (tirec <= ti) & np.isfinite(rvar)
        # Vaccine source per active agent (only if vaccines are registered).
        has_vacc = bool(covid.vaccine_pars)
        if has_vacc:
            is_vacc = np.asarray(covid.vaccinated[auids])
            vsrc = np.asarray(covid.vaccine_source[auids])
        for v in range(covid.nv):
            natural = np.zeros(len(ru))
            if was_inf.any():
                natural[was_inf] = self.matrix[v, rvar[was_inf].astype(int)]
            imm = natural
            if has_vacc and is_vacc.any():
                var_label = covid.variant_map[v]
                eff_by_vacc = np.array([covid.vaccine_pars[covid.vaccine_map[i]].get(var_label, 1.0)
                                        for i in sorted(covid.vaccine_map)])
                vaccine = np.zeros(len(ru))
                vaccine[is_vacc] = eff_by_vacc[vsrc[is_vacc].astype(int)]
                imm = np.maximum(natural, vaccine)
            eff = nab_vals * imm
            covid.sus_imm[v, ru]  = cvimm.calc_VE(eff, 'sus',  nab_eff)
            covid.symp_imm[v, ru] = cvimm.calc_VE(eff, 'symp', nab_eff)
            covid.sev_imm[v, ru]  = cvimm.calc_VE(eff, 'sev',  nab_eff)
        return

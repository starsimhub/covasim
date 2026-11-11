"""
Connectors for Covasim on the Starsim base.

``cv.CrossImmunity(ss.Connector)`` applies Covasim's cross-immunity and waning immunity
to the ``cv.COVID`` module. It runs after the disease states are updated and before the
interventions and transmission, where v3 ran ``check_immunity()``. Each step it reads every
ever-recovered agent's ``recovered_variant`` and the asymmetric ``nv x nv`` cross-immunity
matrix and writes the module's 2D ``sus_imm``/``symp_imm``/``sev_imm`` protection arrays,
which ``cv.COVID.infect()`` and ``set_prognoses()`` then use.

With ``use_waning=True`` (the default), protection is computed from the neutralizing antibody
(NAb) levels (which ``cv.COVID`` updates) via ``calc_VE()``. With ``use_waning=False``, the matrix
is applied directly (no per-agent NAb level). Adding the connector also turns on the module's
``cross_immunity_active`` flag, enabling reinfection (recovery restores susceptibility).

``cv.Sim`` adds this connector by default only when ``use_waning=True``; as in v3, there is then no
reinfection with ``use_waning=False``. To use static cross-immunity between variants without waning,
supply it explicitly, e.g. ``cv.Sim(use_waning=False, connectors=cv.CrossImmunity(immunity=...))``.
"""
import numpy as np
import starsim as ss

from . import immunity as cvimm

__all__ = ['CrossImmunity']


class CrossImmunity(ss.Connector):
    """
    Cross-immunity between variants, and waning immunity, for cv.COVID.

    With ``use_waning=True``, each agent's protection against each variant is computed from their current
    NAb level, weighted by the cross-immunity from the variant they last recovered from (or the
    efficacy of their vaccine, whichever is larger). With ``use_waning=False``, agents who have recovered
    are protected by the cross-immunity matrix value directly, with no waning.

    Args:
        immunity (array): optional ``nv x nv`` cross-immunity matrix, indexed by ``[challenge variant, variant recovered from]``
            (default: the values from ``cv.get_cross_immunity()``, and 1.0 for the same variant)
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

    def init_post(self):
        """Build the cross-immunity matrix and switch the module into the reinfection regime."""
        super().init_post()
        covid = self.sim.diseases[self.pars.disease]
        self.matrix = cvimm.build_immunity_matrix(covid.variant_map, override=self.pars.immunity)
        covid.cross_immunity_active = True       # enable reinfection
        return

    def step(self):
        """Write per-variant protection (the v3 ``check_immunity()``). Two regimes:

          - ``use_waning=False``: static, NAb-free -- ``imm = matrix[target, source]`` for every
            ever-recovered agent (finite ``ti_recovered ≤ ti`` -- v3's ``was_inf = t >= date_recovered``).
          - ``use_waning=True``: for each agent take
            ``imm = max(natural_cross_immunity, vaccine_efficacy)`` per variant and set
            ``sus_imm/symp_imm/sev_imm = calc_VE(nab × imm, axis)`` -- so natural + vaccine immunity share
            one NAb-weighted efficacy curve. With no vaccine registered the vaccine term is 0 (only
            ever-recovered agents get nonzero protection).
        """
        covid = self.sim.diseases[self.pars.disease]
        ti = covid.ti
        if not bool(covid.pars.use_waning):
            # Static path: matrix protection for ever-recovered agents only
            rec = (covid.ti_recovered <= ti).uids
            if not len(rec):
                return
            src = covid.recovered_variant[rec]
            finite = np.isfinite(src)
            if not finite.any():
                return
            ru = rec[finite]
            src_v = src[finite].astype(int)
            for v in range(covid.nv):
                imm = self.matrix[v, src_v]
                covid.sus_imm[ru, v]  = imm
                covid.symp_imm[ru, v] = imm
                covid.sev_imm[ru, v]  = imm
            return

        # NAb-weighted path: compute protection over all active agents (as in v3's check_immunity()):
        # natural cross-immunity or vaccine efficacy, whichever is larger, x current NAb level
        nab_eff = covid.pars.nab_eff
        auids = covid.sim.people.auids
        nab_vals = covid.nab[auids]
        # Natural source variant per active agent (was_inf = finite ti_recovered<=ti with finite variant).
        tirec = covid.ti_recovered[auids]
        rvar  = covid.recovered_variant[auids]
        was_inf = np.isfinite(tirec) & (tirec <= ti) & np.isfinite(rvar)
        # Vaccine efficacy against each variant, by vaccine index (vaccine_map) and variant index
        vsrc = covid.vaccine_source[auids]
        is_vacc = np.isfinite(vsrc) # Agents who received a NAb-based vaccine (cv.simple_vaccine doesn't set this)
        vsrc = vsrc[is_vacc].astype(int)
        vacc_eff = np.zeros((len(covid.vaccine_map), covid.nv))
        for i, vacc_label in covid.vaccine_map.items():
            for v, var_label in covid.variant_map.items():
                vacc_eff[i, v] = covid.vaccine_pars[vacc_label].get(var_label, 1.0)
        for v in range(covid.nv):
            natural_imm = np.zeros(len(auids))
            natural_imm[was_inf] = self.matrix[v, rvar[was_inf].astype(int)]
            vaccine_imm = np.zeros(len(auids))
            vaccine_imm[is_vacc] = vacc_eff[vsrc, v]
            imm = np.maximum(natural_imm, vaccine_imm)
            eff = nab_vals * imm
            covid.sus_imm[auids, v]  = cvimm.calc_VE(eff, 'sus',  nab_eff)
            covid.symp_imm[auids, v] = cvimm.calc_VE(eff, 'symp', nab_eff)
            covid.sev_imm[auids, v]  = cvimm.calc_VE(eff, 'sev',  nab_eff)
        return

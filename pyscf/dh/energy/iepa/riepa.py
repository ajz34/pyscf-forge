r""" Restricted IEPA-like methods. """

from pyscf.dh.energy import EngPostSCFBase
from pyscf.dh import util
from pyscf import lib, ao2mo, __config__, df
import numpy as np
from scipy.special import erfc
import warnings

CONFIG_tol_eng_pair_iepa = getattr(__config__, "tol_eng_pair_iepa", 1e-10)
CONFIG_max_cycle_pair_iepa = getattr(__config__, "max_cycle_pair_iepa", 64)
CONFIG_iepa_schemes = getattr(__config__, "iepa_schemes", ["MP2", "IEPA", "sIEPA", "MP2cr"])


def kernel_energy_riepa(
    mo_energy, gen_g_IJab, mo_occ, iepa_schemes,
    screen_func=erfc,
    tol=1e-10, max_cycle=64,
    tensors=None,
    verbose=lib.logger.NOTE):
    """ Kernel of restricted IEPA-like methods.

    Parameters
    ----------
    mo_energy : np.ndarray
        Molecular orbital energy levels.
    gen_g_IJab : callable
        Generate ERI block :math:`(ij|ab)` where :math:`i, j` is specified.
        Function signature should be ``gen_g_IJab(i: int, j: int) -> np.ndarray``
        with shape of returned array (a, b).
    mo_occ : mp.ndarray
        Molecular orbitals occupation numbers.
    iepa_schemes : list[str] or str
        IEPA schemes. Currently MP2, IEPA, SIEPA, MP2cr, MP2cr2 accepted.

    screen_func : callable
        Function used in screened IEPA. Default is erfc, as applied in functional ZRPS.
    tol : float
        Threshold of pair energy convergence for IEPA or sIEPA methods.
    max_cycle : int
        Maximum iteration number of energy convergence for IEPA or sIEPA methods.
    tensors : HybridDict
        Storage space for intermediate and output pair-energy. Values will be changed in-place.
    verbose : int
        Verbose level for PySCF.

    Notes
    -----
    This kernel generates several intermediate tensor entries:

    - ``pair_METHOD_aa`` and ``pair_METHOD_ab``: pair energy of specified IEPA ``METHOD``.
    - ``n2_pair_aa`` and ``n2_pair_ab``: sum of squares of tensor :math:`n_{ij} = \\sum_{ab} (t_{ij}^{ab})^2`.
    - ``norm_METHOD``: normalization factors of ``MP2CR`` or ``MP2CR2``.
    """
    log = lib.logger.new_logger(verbose=verbose)
    log.info("[INFO] Start restricted IEPA")

    mask_occ = mo_occ != 0
    mask_vir = mo_occ == 0
    eo = mo_energy[mask_occ]
    ev = mo_energy[mask_vir]
    nocc = mask_occ.sum()
    tensors = tensors if tensors is not None else dict()

    # parse IEPA schemes
    # `iepa_schemes` option is either str or list[str]; change to list
    if not isinstance(iepa_schemes, str):
        iepa_schemes = [i.upper() for i in iepa_schemes]
    else:
        iepa_schemes = [iepa_schemes.upper()]

    # check IEPA scheme sanity
    check_iepa_scheme = set(iepa_schemes).difference(["MP2", "MP2CR", "MP2CR2", "DCPT2", "IEPA", "SIEPA"])
    if len(check_iepa_scheme) != 0:
        raise ValueError(f"Several schemes are not recognized IEPA schemes: {check_iepa_scheme}")
    log.info("[INFO] Recognized IEPA schemes: " + ", ".join(iepa_schemes))

    # allocate pair energies
    for scheme in iepa_schemes:
        tensors[f"pair_{scheme}_aa"] = np.zeros(shape=(nocc, nocc))
        tensors[f"pair_{scheme}_ab"] = np.zeros(shape=(nocc, nocc))
        if scheme in ["MP2CR", "MP2CR2"]:
            if "pair_MP2_aa" not in tensors:
                tensors["pair_MP2_aa"] = np.zeros(shape=(nocc, nocc))
                tensors["pair_MP2_ab"] = np.zeros(shape=(nocc, nocc))
            if "n2_pair_aa" not in tensors:
                tensors["n2_pair_aa"] = np.zeros(shape=(nocc, nocc))
                tensors["n2_pair_ab"] = np.zeros(shape=(nocc, nocc))

    # In evaluation of MP2/cr or MP2/cr2, MP2 pair energy is evaluated first.
    schemes_for_pair = set(iepa_schemes)
    if "MP2CR" in schemes_for_pair or "MP2CR2" in schemes_for_pair:
        schemes_for_pair.difference_update(["MP2CR", "MP2CR2"])
        schemes_for_pair.add("MP2")
    # scratch tensor of - e_a - e_b
    D_ab = - ev[:, None] - ev[None, :]
    # main driver
    for I in range(nocc):
        for J in range(I + 1):
            log.debug(f"In IEPA kernel, pair {I, J}")
            D_IJab = eo[I] + eo[J] + D_ab
            g_IJab = gen_g_IJab(I, J)  # Y_OV[:, I].T @ Y_OV[:, J]
            g_IJab_asym = g_IJab - g_IJab.T
            # evaluate pair energy for different schemes
            for scheme in schemes_for_pair:
                pair_aa = tensors[f"pair_{scheme}_aa"]
                pair_ab = tensors[f"pair_{scheme}_ab"]
                if scheme == "MP2":
                    e_pair_os = get_pair_mp2(g_IJab, D_IJab, 1)
                    e_pair_ss = get_pair_mp2(g_IJab_asym, D_IJab, 0.5)
                elif scheme == "DCPT2":
                    e_pair_os = get_pair_dcpt2(g_IJab, D_IJab, 1)
                    e_pair_ss = get_pair_dcpt2(g_IJab_asym, D_IJab, 0.5)
                elif scheme == "IEPA":
                    e_pair_os = get_pair_iepa(g_IJab, D_IJab, 1, tol=tol, max_cycle=max_cycle)
                    e_pair_ss = get_pair_iepa(g_IJab_asym, D_IJab, 0.5, tol=tol, max_cycle=max_cycle)
                elif scheme == "SIEPA":
                    e_pair_os = get_pair_siepa(g_IJab, D_IJab, 1,
                                               screen_func=screen_func, tol=tol, max_cycle=max_cycle)
                    e_pair_ss = get_pair_siepa(g_IJab_asym, D_IJab, 0.5,
                                               screen_func=screen_func, tol=tol, max_cycle=max_cycle)
                else:
                    assert False
                pair_aa[I, J] = pair_aa[J, I] = e_pair_ss
                pair_ab[I, J] = pair_ab[J, I] = e_pair_os
            # MP2/cr methods require norm
            if "MP2CR" in iepa_schemes or "MP2CR2" in iepa_schemes:
                n2_aa = tensors["n2_pair_aa"]
                n2_ab = tensors["n2_pair_ab"]
                n2_aa[I, J] = n2_aa[J, I] = ((g_IJab_asym / D_IJab) ** 2).sum()
                n2_ab[I, J] = n2_ab[J, I] = ((g_IJab / D_IJab) ** 2).sum()

    # process MP2/cr afterwards
    # MP2/cr I
    if "MP2CR" in iepa_schemes:
        n2_aa = tensors["n2_pair_aa"]
        n2_ab = tensors["n2_pair_ab"]
        norm = get_rmp2cr_norm(n2_aa, n2_ab)
        tensors["norm_MP2CR"] = norm
        tensors["pair_MP2CR_aa"] = tensors["pair_MP2_aa"] / norm
        tensors["pair_MP2CR_ab"] = tensors["pair_MP2_ab"] / norm
    # MP2/cr II
    if "MP2CR2" in iepa_schemes:
        n2_aa = tensors["n2_pair_aa"]
        n2_ab = tensors["n2_pair_ab"]
        norm = get_rmp2cr2_norm(n2_aa, n2_ab)
        tensors["norm_MP2CR2"] = norm
        tensors["pair_MP2CR2_aa"] = tensors["pair_MP2_aa"] * norm
        tensors["pair_MP2CR2_ab"] = tensors["pair_MP2_ab"] * norm

    # Finalize energy evaluation
    results = dict()
    for scheme in iepa_schemes:
        eng_aa = 0.5 * tensors[f"pair_{scheme}_aa"].sum()
        eng_ab = tensors[f"pair_{scheme}_ab"].sum()
        eng_os = eng_ab
        eng_ss = 2 * eng_aa
        eng_tot = eng_os + eng_ss
        results[f"eng_corr_{scheme}_aa"] = eng_aa
        results[f"eng_corr_{scheme}_ab"] = eng_ab
        results[f"eng_corr_{scheme}_OS"] = eng_os
        results[f"eng_corr_{scheme}_SS"] = eng_ss
        results[f"eng_corr_{scheme}"] = eng_tot
        log.info(f"[RESULT] Energy corr {scheme}_OS: {eng_os :18.10f}")
        log.info(f"[RESULT] Energy corr {scheme}_SS: {eng_ss :18.10f}")
        log.info(f"[RESULT] Energy corr {scheme}   : {eng_tot:18.10f}")
    return results


def get_pair_mp2(g_ab, D_ab, scale_e):
    """ Pair energy evaluation for MP2.

    .. math::
        e_{ij} = s (\\tilde g_{ij}^{ab})^2 / D_{ij}^{ab}

    In this function, :math:`i, j` are defined.

    Parameters
    ----------
    g_ab : np.ndarray
        :math:`\\tilde g_{ij}^{ab}` refers to :math:`\\langle ij || ab \\rangle`.

        For oppo-spin, :math:`\\tilde g_{ij}^{ab} = (ia|jb)`;
        for same-spin, :math:`\\tilde g_{ij}^{ab} = (ia|jb) - (ib|ja)`.

        Should be matrix of indices (a, b).
    D_ab : np.ndarray
        :math:`D_{ij}^{ab}` refers to :math:`\\varepsilon_i + \\varepsilon_j - \\varepsilon_a - \\varepsilon_b`.

        Should be matrix of indices (a, b).
    scale_e : float
        :math:`s` is scale of MP2.

        Generally, :math:`s = 1` for oppo-spin, :math:`s = 0.5` for same-spin.

    Returns
    -------
    float
        Pair energy :math:`e_{ij}`.
    """
    return scale_e * (g_ab * g_ab / D_ab).sum()


def get_pair_dcpt2(g_ab, D_ab, scale_e):
    """ Pair energy evaluation for DCPT2.

    .. math::
        e_{ij} = s \\frac{1}{2} (D_{ij}^{ab} - \\sqrt{(D_{ij}^{ab})^2 + 4 (\\tilde g_{ij}^{ab})^2})

    See Also
    --------
    get_pair_mp2
    """
    return 0.5 * scale_e * (- D_ab - np.sqrt(D_ab ** 2 + 4 * g_ab ** 2)).sum()


def get_pair_siepa(g_ab, D_ab, scale_e, screen_func, tol=1e-10, max_cycle=64):
    """ Pair energy evaluation for screened IEPA.

    .. math::
        e_{ij} = s \\frac{(\\tilde g_{ij}^{ab})^2}{D_{ij}^{ab} + s \\times \\mathrm{screen}(- D_{ij}^{ab})} e_{ij}

    Parameters
    ----------
    g_ab : np.ndarray
        :math:`\\tilde g_{ij}^{ab}` refers to :math:`\\langle ij || ab \\rangle`.
    D_ab : np.ndarray
        :math:`D_{ij}^{ab}` refers to :math:`\\varepsilon_i + \\varepsilon_j - \\varepsilon_a - \\varepsilon_b`.
    scale_e : float
        :math:`s` is scale of MP2.
    screen_func : callable
        Function used in screened IEPA. For example erfc, which is applied in functional ZRPS.
    tol : float
        Threshold of pair energy convergence for IEPA or sIEPA methods.
    max_cycle : int
        Maximum iteration number of energy convergence for IEPA or sIEPA methods.

    Returns
    -------
    float
        Pair energy :math:`e_{ij}`.

    See Also
    --------
    get_pair_mp2
    get_pair_iepa
    """
    g2_ab = g_ab * g_ab
    sD_ab = screen_func(-D_ab)
    e = (g2_ab / D_ab).sum()
    e_old = 1e8
    n_cycle = 0
    while abs(e_old - e) > tol and n_cycle < max_cycle:
        e_old = e
        e = scale_e * (g2_ab / (D_ab + sD_ab * e)).sum()
        n_cycle += 1
    if n_cycle >= max_cycle:
        warnings.warn(f"[WARN] Maximum cycle {n_cycle} exceeded! Pair energy error: {abs(e_old - e):12.6e}")
    return e


def get_pair_iepa(g_ab, D_ab, scale_e, tol=1e-10, max_cycle=64):
    """ Pair energy evaluation for IEPA.

    This procedure sets screen function to 1.

    See Also
    --------
    get_pair_mp2
    get_pair_iepa
    """
    g2_ab = g_ab * g_ab
    e = (g2_ab / D_ab).sum()
    e_old = 1e8
    n_cycle = 0
    while abs(e_old - e) > tol and n_cycle < max_cycle:
        e_old = e
        e = scale_e * (g2_ab / (D_ab + e)).sum()
        n_cycle += 1
    if n_cycle >= max_cycle:
        warnings.warn(f"[WARN] Maximum cycle {n_cycle} exceeded! Pair energy error: {abs(e_old - e):12.6e}")
    return e


def get_rmp2cr_norm(n2_aa, n2_ab):
    """ Comput Norm of MP2/cr (restricted). """
    nocc = n2_aa.shape[0]
    norm = np.ones((nocc, nocc))
    n2_aa_sum, n2_ab_sum = n2_aa.sum(axis=1), n2_ab.sum(axis=1)
    norm += 0.5 * (n2_ab_sum[:, None] + n2_ab_sum[None, :])
    norm += 0.25 * (n2_aa_sum[:, None] + n2_aa_sum[None, :])
    return norm


def get_rmp2cr2_norm(n2_aa, n2_ab):
    """ Comput Norm of MP2/cr II (restricted). """
    nocc = n2_aa.shape[0]
    norm = np.zeros((nocc, nocc))
    n2_aa_sum, n2_ab_sum = n2_aa.sum(axis=1), n2_ab.sum(axis=1)
    norm2 = 1 + n2_ab.sum() + 0.5 * n2_aa.sum()
    norm -= 2 * (n2_ab_sum[:, None] + n2_ab_sum[None, :])
    norm -= n2_aa_sum[:, None] + n2_aa_sum[None, :]
    norm += n2_ab.diagonal()[:, None] + n2_ab.diagonal()[None, :]
    norm += 2 * n2_ab + n2_aa
    for i in range(nocc):
        norm[i, i] /= 2
        norm[i, i] -= n2_ab[i, i] + 0.5 * n2_aa[i, i]
    norm = norm / norm2 + 1
    return norm


class RIEPAConv(EngPostSCFBase):
    """ Restricted IEPA-like class of doubly hybrid with conventional integral. """

    @property
    def restricted(self):  # type: () -> bool
        return True

    def __init__(self, mf, frozen=None, omega=0, **kwargs):
        super().__init__(mf)
        self.omega = omega
        self.frozen = frozen if frozen is not None else 0
        self.frac_num = None
        self.conv_tol = CONFIG_tol_eng_pair_iepa
        self.max_cycle = CONFIG_max_cycle_pair_iepa
        self.iepa_schemes = CONFIG_iepa_schemes
        self.siepa_screen = erfc
        self.set(**kwargs)

    kernel_energy_riepa = kernel_energy_riepa

    def driver_eng_riepa(self, **_kwargs):
        mask = self.get_frozen_mask()
        mask_occ = mask & (self.mo_occ != 0)
        mask_vir = mask & (self.mo_occ == 0)
        mo_occ_act = self.mo_occ[mask]
        mol = self.mol
        nocc_act = mask_occ.sum()
        nvir_act = mask_vir.sum()
        occ_coeff_act = self.mo_coeff[:, mask_occ]
        vir_coeff_act = self.mo_coeff[:, mask_vir]
        mo_energy_act = self.mo_energy[mask]
        # eri generator
        eri_or_mol = self.scf._eri if self.omega == 0 else mol
        eri_or_mol = eri_or_mol if eri_or_mol is not None else mol
        with mol.with_range_coulomb(self.omega):
            g_iajb = ao2mo.general(
                eri_or_mol, (occ_coeff_act, vir_coeff_act, occ_coeff_act, vir_coeff_act)) \
                .reshape(nocc_act, nvir_act, nocc_act, nvir_act)

        def gen_g_IJab(i, j):
            return g_iajb[i, :, j]

        results = kernel_energy_riepa(
            mo_energy_act, gen_g_IJab, mo_occ_act,
            iepa_schemes=self.iepa_schemes,
            screen_func=self.siepa_screen,
            tol=self.conv_tol,
            max_cycle=self.max_cycle,
            tensors=self.tensors,
            verbose=self.verbose
        )

        self.results.update(results)
        return results

    kernel = driver_eng_riepa


class RIEPARI(EngPostSCFBase):
    """ Restricted IEPA-like class of doubly hybrid with RI integral. """

    @property
    def restricted(self):  # type: () -> bool
        return True

    def __init__(self, mf, frozen=None, omega=0, with_df=None, **kwargs):
        super().__init__(mf)
        self.omega = omega
        if with_df is None:
            with_df = getattr(self.scf, "with_df", None)
        if with_df is None:
            with_df = df.DF(self.mol, auxbasis=df.make_auxbasis(self.mol, mp2fit=True))
        self.with_df = with_df
        self.frozen = frozen if frozen is not None else 0
        self.frac_num = None
        self.conv_tol = CONFIG_tol_eng_pair_iepa
        self.max_cycle = CONFIG_max_cycle_pair_iepa
        self.iepa_schemes = CONFIG_iepa_schemes
        self.siepa_screen = erfc
        self.set(**kwargs)

    kernel_energy_riepa = kernel_energy_riepa

    def driver_eng_riepa(self, **_kwargs):
        mask = self.get_frozen_mask()
        mask_occ = mask & (self.mo_occ != 0)
        mo_occ_act = self.mo_occ[mask]
        nocc_act = mask_occ.sum()
        nact = mask.sum()
        mo_coeff_act = self.mo_coeff[:, mask]
        mo_energy_act = self.mo_energy[mask]
        # eri generator
        omega = self.omega
        with_df = util.get_with_df_omega(self.with_df, omega)
        max_memory = self.max_memory - lib.current_memory()[0]
        cderi_uov = self.tensors.get("cderi_uov", None)
        if cderi_uov is None:
            cderi_uov = util.get_cderi_mo(with_df, mo_coeff_act, None, (0, nocc_act, nocc_act, nact), max_memory)
            self.tensors["cderi_uov"] = cderi_uov

        def gen_g_IJab(i, j):
            return cderi_uov[:, i].T @ cderi_uov[:, j]

        results = kernel_energy_riepa(
            mo_energy_act, gen_g_IJab, mo_occ_act,
            iepa_schemes=self.iepa_schemes,
            screen_func=self.siepa_screen,
            tol=self.conv_tol,
            max_cycle=self.max_cycle,
            tensors=self.tensors,
            verbose=self.verbose,
        )

        self.results.update(results)
        return results

    kernel = driver_eng_riepa


if __name__ == '__main__':
    def main_1():
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf_scf = scf.RHF(mol).run()
        mf_mp = RIEPAConv(mf_scf, frozen=[1, 2]).run()
        print(mf_mp.results)

    def main_2():
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf_scf = scf.RHF(mol).run()
        mf_mp = RIEPARI(mf_scf, frozen=[1, 2]).run()
        print(mf_mp.results)

    main_1()
    main_2()

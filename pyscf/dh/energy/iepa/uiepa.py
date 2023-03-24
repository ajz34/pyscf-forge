from pyscf.dh.energy import UDHBase
from pyscf.dh import util
from .riepa import get_pair_mp2, get_pair_iepa, get_pair_siepa, get_pair_dcpt2
from pyscf import lib, ao2mo
import numpy as np
from scipy.special import erfc
import warnings


class UIEPAofDH(UDHBase):
    """ Unrestricted IEPA (independent electron-pair approximation) class of doubly hybrid. """

    def __init__(self, *args, **kwargs):
        self.siepa_screen = erfc
        super().__init__(*args, **kwargs)

    def kernel(self, **kwargs):
        with self.params.temporary_flags(kwargs):
            results = driver_energy_uiepa(self)
        self.params.update_results(results)
        return results


def driver_energy_uiepa(mf_dh):
    """ Driver of unrestricted IEPA energy.

    Parameters
    ----------
    mf_dh : UDH
        Unrestricted doubly hybrid object.

    Returns
    -------
    dict

    See Also
    --------
    .riepa.driver_energy_riepa
    """
    mf_dh.build()
    log = mf_dh.log
    params = mf_dh.params
    mf_dh._flag_snapshot = mf_dh.params.flags.copy()
    results_summary = dict()
    # some results from mf_dh
    mol = mf_dh.mol
    mo_coeff_act = mf_dh.mo_coeff_act
    mo_energy_act = mf_dh.mo_energy_act
    nOcc, nVir, nact = mf_dh.nOcc, mf_dh.nVir, mf_dh.nact
    # some flags
    tol_eng_pair_iepa = params.flags["tol_eng_pair_iepa"]
    max_cycle_iepa = params.flags["max_cycle_pair_iepa"]
    iepa_schemes = params.flags["iepa_schemes"]
    integral_scheme = mf_dh.params.flags.get("integral_scheme_iepa", mf_dh.params.flags["integral_scheme"]).lower()
    # main loop
    omega_list = params.flags["omega_list_iepa"]
    for omega in omega_list:
        log.info(f"[INFO] Evaluation of IEPA at omega({omega})")
        # define g_iajb generation
        if integral_scheme.startswith("ri"):
            with_df = util.get_with_df_omega(mf_dh.with_df, omega)
            Y_OV = [params.tensors.get(util.pad_omega(f"Y_OV_{sn}", omega), None) for sn in ("a", "b")]
            if Y_OV[0] is None:
                for s, sn in [(0, "a"), (1, "b")]:
                    Y_OV[s] = params.tensors[util.pad_omega(f"Y_OV_{sn}", omega)] = util.get_cderi_mo(
                        with_df, mo_coeff_act[s], None, (0, nOcc[s], nOcc[s], nact[s]),
                        mol.max_memory - lib.current_memory()[0])
                    
            def gen_g_IJab(s0, s1, i, j):
                return Y_OV[s0][:, i].T @ Y_OV[s1][:, j]
        elif integral_scheme.startswith("conv"):
            log.warn("Conventional integral of post-SCF is not recommended!\n"
                     "Use density fitting approximation is preferred.")
            CO = [mo_coeff_act[s][:, :nOcc[s]] for s in (0, 1)]
            CV = [mo_coeff_act[s][:, nOcc[s]:] for s in (0, 1)]
            eri_or_mol = mf_dh.scf._eri if omega == 0 else mol
            if eri_or_mol is None:
                eri_or_mol = mol
            g_iajb = [np.array([])] * 3
            with mol.with_range_coulomb(omega):
                for s0, s1, ss in zip((0, 0, 1), (0, 1, 1), (0, 1, 2)):
                    g_iajb[ss] = ao2mo.general(eri_or_mol, (CO[s0], CV[s0], CO[s1], CV[s1])) \
                                      .reshape(nOcc[s0], nVir[s0], nOcc[s1], nVir[s1])
                log.debug("Spin {:}{:} ao2mo finished".format(s0, s1))

            def gen_g_IJab(s0, s1, i, j):
                if (s0, s1) == (0, 0):
                    return g_iajb[0][i, :, j]
                elif (s0, s1) == (1, 1):
                    return g_iajb[2][i, :, j]
                elif (s0, s1) == (0, 1):
                    return g_iajb[1][i, :, j]
                # elif (s0, s1) == (1, 0):
                #     return g_iajb[1][j, :, i].T
                else:
                    assert False, "Not accepted spin!"
        else:
            raise NotImplementedError

        results = kernel_energy_uiepa(
            mo_energy_act, gen_g_IJab, nOcc, iepa_schemes,
            screen_func=mf_dh.siepa_screen,
            tol=tol_eng_pair_iepa, max_cycle=max_cycle_iepa,
            tensors=params.tensors,
            verbose=mf_dh.verbose
        )
        results = {util.pad_omega(key, omega): val for (key, val) in results.items()}
        results_summary.update(results)
    return results_summary


def kernel_energy_uiepa(
        mo_energy, gen_g_IJab, nocc, iepa_schemes,
        screen_func=erfc,
        tol=1e-10, max_cycle=64,
        tensors=None,
        verbose=lib.logger.NOTE):
    """ Kernel of restricted IEPA-like methods.

    Parameters
    ----------
    mo_energy : list[np.ndarray]
        Molecular orbital energy levels.
    gen_g_IJab : callable
        Generate ERI block :math:`(ij|ab)` where :math:`i, j` is specified.
        Function signature should be ``gen_g_IJab(s0: int, s1: int, i: int, j: int) -> np.ndarray``
        with shape of returned array (a, b) and spin of s0 (i, a) and spin of s1 (j, b).
    nocc : list[int]
        Number of occupied molecular orbitals.
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

    See Also
    --------
    pyscf.dh.energy.riepa.kernel_energy_riepa_ri
    """
    log = lib.logger.new_logger(verbose=verbose)
    log.info("[INFO] Start unrestricted IEPA")

    eo = [mo_energy[s][:nocc[s]] for s in (0, 1)]
    ev = [mo_energy[s][nocc[s]:] for s in (0, 1)]

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
    if "MP2CR2" in iepa_schemes:
        raise NotImplementedError("MP2CR2 is not accepted as an unrestricted method currently!")

    # allocate pair energies
    for scheme in iepa_schemes:
        tensors.create(f"pair_{scheme}_aa", shape=(nocc[0], nocc[0]))
        tensors.create(f"pair_{scheme}_ab", shape=(nocc[0], nocc[1]))
        tensors.create(f"pair_{scheme}_bb", shape=(nocc[1], nocc[1]))
        if scheme == "MP2CR":
            if "pair_MP2_aa" not in tensors:
                tensors.create("pair_MP2_aa", shape=(nocc[0], nocc[0]))
                tensors.create("pair_MP2_ab", shape=(nocc[0], nocc[1]))
                tensors.create("pair_MP2_bb", shape=(nocc[1], nocc[1]))
            tensors.create("n2_pair_aa", shape=(nocc[0], nocc[0]))
            tensors.create("n2_pair_ab", shape=(nocc[0], nocc[1]))
            tensors.create("n2_pair_bb", shape=(nocc[1], nocc[1]))

    # In evaluation of MP2/cr, MP2 pair energy is evaluated first.
    schemes_for_pair = set(iepa_schemes)
    if "MP2CR" in schemes_for_pair:
        schemes_for_pair.difference_update(["MP2CR"])
        schemes_for_pair.add("MP2")

    for ssn, s0, s1 in [("aa", 0, 0), ("ab", 0, 1), ("bb", 1, 1)]:
        log.debug(f"In IEPA kernel, spin {ssn}")
        is_same_spin = s0 == s1
        D_ab = - ev[s0][:, None] - ev[s1][None, :]
        for I in range(nocc[s0]):
            maxJ = I if is_same_spin else nocc[s1]
            for J in range(maxJ):
                log.debug(f"In IEPA kernel, pair {I, J}")
                D_IJab = eo[s0][I] + eo[s1][J] + D_ab
                g_IJab = gen_g_IJab(s0, s1, I, J)  # Y_OV[s0][:, I].T @ Y_OV[s1][:, J]
                if is_same_spin:
                    g_IJab = g_IJab - g_IJab.T
                # evaluate pair energy for different schemes
                for scheme in schemes_for_pair:
                    pair_mat = tensors[f"pair_{scheme}_{ssn}"]
                    scale = 0.5 if is_same_spin else 1
                    if scheme == "MP2":
                        e_pair = get_pair_mp2(g_IJab, D_IJab, scale)
                    elif scheme == "DCPT2":
                        e_pair = get_pair_dcpt2(g_IJab, D_IJab, scale)
                    elif scheme == "IEPA":
                        e_pair = get_pair_iepa(g_IJab, D_IJab, scale, tol=tol, max_cycle=max_cycle)
                    elif scheme == "SIEPA":
                        e_pair = get_pair_siepa(g_IJab, D_IJab, scale,
                                                screen_func=screen_func, tol=tol, max_cycle=max_cycle)
                    else:
                        assert False
                    pair_mat[I, J] = e_pair
                    if is_same_spin:
                        pair_mat[J, I] = e_pair
                if "MP2CR" in iepa_schemes:
                    n2_mat = tensors[f"n2_pair_{ssn}"]
                    n2_val = ((g_IJab / D_IJab)**2).sum()
                    n2_mat[I, J] = n2_val
                    if is_same_spin:
                        n2_mat[J, I] = n2_val

    # process MP2/cr afterwards
    if "MP2CR" in iepa_schemes:
        n2_aa = tensors["n2_pair_aa"]
        n2_ab = tensors["n2_pair_ab"]
        n2_bb = tensors["n2_pair_bb"]
        norms = get_ump2cr_norm(n2_aa, n2_ab, n2_bb)
        tensors["norm_MP2CR_aa"], tensors["norm_MP2CR_ab"], tensors["norm_MP2CR_bb"] = norms
        tensors["pair_MP2CR_aa"] = tensors["pair_MP2_aa"] / norms[0]
        tensors["pair_MP2CR_ab"] = tensors["pair_MP2_ab"] / norms[1]
        tensors["pair_MP2CR_bb"] = tensors["pair_MP2_bb"] / norms[2]

    # Finalize energy evaluation
    results = dict()
    for scheme in iepa_schemes:
        eng_os = eng_ss = 0
        for ssn in ("aa", "ab", "bb"):
            is_same_spin = ssn[0] == ssn[1]
            scale = 0.5 if is_same_spin else 1
            eng_pair = scale * tensors[f"pair_{scheme}_{ssn}"].sum()
            results[f"eng_corr_{scheme}_{ssn}"] = eng_pair
            log.info(f"[RESULT] Energy corr {scheme} of spin {ssn}: {eng_pair:18.10f}")
            if is_same_spin:
                eng_ss += eng_pair
            else:
                eng_os += eng_pair
        eng_tot = eng_os + eng_ss
        results[f"eng_corr_{scheme}_OS"] = eng_os
        results[f"eng_corr_{scheme}_SS"] = eng_ss
        results[f"eng_corr_{scheme}"] = eng_tot
        log.info(f"[RESULT] Energy corr {scheme}_OS: {eng_os :18.10f}")
        log.info(f"[RESULT] Energy corr {scheme}_SS: {eng_ss :18.10f}")
        log.info(f"[RESULT] Energy corr {scheme}   : {eng_tot:18.10f}")
    return results


def get_ump2cr_norm(n2_aa, n2_ab, n2_bb):
    """ Comput Norm of MP2/cr (unrestricted). """
    nocc = n2_ab.shape
    # case 1: i, j -> alpha, beta
    np_ab = np.ones((nocc[0], nocc[1]))
    np_ab += 0.5 * (n2_ab.sum(axis=1)[:, None] + n2_ab.sum(axis=0)[None, :])
    np_ab += 0.25 * (n2_aa.sum(axis=1)[:, None] + n2_bb.sum(axis=0)[None, :])
    # case 2: i, j -> alpha, alpha
    np_aa = np.ones((nocc[0], nocc[0]))
    np_aa += 0.5 * (n2_ab.sum(axis=1)[:, None] + n2_ab.sum(axis=1)[None, :])
    np_aa += 0.25 * (n2_aa.sum(axis=1)[:, None] + n2_aa.sum(axis=0)[None, :])
    # case 3: i, j -> beta, beta
    np_bb = np.ones((nocc[1], nocc[1]))
    np_bb += 0.5 * (n2_ab.sum(axis=0)[:, None] + n2_ab.sum(axis=0)[None, :])
    np_bb += 0.25 * (n2_bb.sum(axis=1)[:, None] + n2_bb.sum(axis=0)[None, :])
    return np_aa, np_ab, np_bb


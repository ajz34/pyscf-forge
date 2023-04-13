from pyscf.dh import util
from pyscf.dh.energy.iepa.riepa import (
    get_pair_mp2, get_pair_iepa, get_pair_siepa, get_pair_dcpt2, RIEPAConv, RIEPARI)
from pyscf import lib, ao2mo
import numpy as np
from scipy.special import erfc


def kernel_energy_uiepa(
        mo_energy, gen_g_IJab, mo_occ, iepa_schemes,
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
    mo_occ : list[np.ndarray]
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

    See Also
    --------
    pyscf.dh.energy.riepa.kernel_energy_riepa_ri
    """
    log = lib.logger.new_logger(verbose=verbose)
    log.info("[INFO] Start unrestricted IEPA")

    mask_occ = [mo_occ[s] != 0 for s in (0, 1)]
    mask_vir = [mo_occ[s] == 0 for s in (0, 1)]
    eo = [mo_energy[s][mask_occ[s]] for s in (0, 1)]
    ev = [mo_energy[s][mask_vir[s]] for s in (0, 1)]
    nocc = tuple([mask_occ[s].sum() for s in (0, 1)])
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
    if "MP2CR2" in iepa_schemes:
        raise NotImplementedError("MP2CR2 is not accepted as an unrestricted method currently!")

    # allocate pair energies
    for scheme in iepa_schemes:
        tensors[f"pair_{scheme}_aa"] = np.zeros(shape=(nocc[0], nocc[0]))
        tensors[f"pair_{scheme}_ab"] = np.zeros(shape=(nocc[0], nocc[1]))
        tensors[f"pair_{scheme}_bb"] = np.zeros(shape=(nocc[1], nocc[1]))
        if scheme == "MP2CR":
            if "pair_MP2_aa" not in tensors:
                tensors["pair_MP2_aa"] = np.zeros(shape=(nocc[0], nocc[0]))
                tensors["pair_MP2_ab"] = np.zeros(shape=(nocc[0], nocc[1]))
                tensors["pair_MP2_bb"] = np.zeros(shape=(nocc[1], nocc[1]))
            tensors["n2_pair_aa"] = np.zeros(shape=(nocc[0], nocc[0]))
            tensors["n2_pair_ab"] = np.zeros(shape=(nocc[0], nocc[1]))
            tensors["n2_pair_bb"] = np.zeros(shape=(nocc[1], nocc[1]))

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


class UIEPAConv(RIEPAConv):
    """ Unrestricted IEPA-like class of doubly hybrid with conventional integral. """

    @property
    def restricted(self):  # type: () -> bool
        return False

    def driver_eng_iepa(self, **_kwargs):
        log = lib.logger.new_logger(verbose=self.verbose)
        mask = self.get_frozen_mask()
        mask_occ = mask & (self.mo_occ != 0)
        mask_vir = mask & (self.mo_occ == 0)
        mo_occ_act = [self.mo_occ[s][mask[s]] for s in (0, 1)]
        mol = self.mol
        nocc_act = tuple(mask_occ.sum(axis=-1))
        nvir_act = tuple(mask_vir.sum(axis=-1))
        occ_coeff_act = [self.mo_coeff[s][:, mask_occ[s]] for s in (0, 1)]
        vir_coeff_act = [self.mo_coeff[s][:, mask_vir[s]] for s in (0, 1)]
        mo_energy_act = [self.mo_energy[s][mask[s]] for s in (0, 1)]
        # eri generator
        eri_or_mol = self.scf._eri if self.omega == 0 else mol
        eri_or_mol = eri_or_mol if eri_or_mol is not None else mol
        g_iajb = [np.array([])] * 3
        with mol.with_range_coulomb(self.omega):
            for s0, s1, ss in ((0, 0, 0), (0, 1, 1), (1, 1, 2)):
                g_iajb[ss] = ao2mo.general(
                    eri_or_mol,
                    (occ_coeff_act[s0], vir_coeff_act[s0], occ_coeff_act[s1], vir_coeff_act[s1])) \
                    .reshape(nocc_act[s0], nvir_act[s0], nocc_act[s1], nvir_act[s1])
                log.debug(f"Spin {s0}{s1} ao2mo finished")

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

        results = self.kernel_energy_iepa(
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

    kernel_energy_iepa = staticmethod(kernel_energy_uiepa)
    kernel = driver_eng_iepa


class UIEPARI(RIEPARI):
    """ Unrestricted IEPA-like class of doubly hybrid with RI integral. """

    @property
    def restricted(self):  # type: () -> bool
        return False

    def driver_eng_iepa(self, **_kwargs):
        log = lib.logger.new_logger(verbose=self.verbose)
        mask = self.get_frozen_mask()
        mask_occ = mask & (self.mo_occ != 0)
        mo_occ_act = [self.mo_occ[s][mask[s]] for s in (0, 1)]
        nact = tuple(mask.sum(axis=-1))
        nocc_act = tuple(mask_occ.sum(axis=-1))
        mo_coeff_act = [self.mo_coeff[s][:, mask[s]] for s in (0, 1)]
        mo_energy_act = [self.mo_energy[s][mask[s]] for s in (0, 1)]
        # eri generator
        omega = self.omega
        with_df = util.get_with_df_omega(self.with_df, omega)
        max_memory = self.max_memory - lib.current_memory()[0]
        cderi_uov = self.tensors.get("cderi_uov", None)
        if cderi_uov is None:
            cderi_uov = [np.zeros(0)] * 2
            for s in (0, 1):
                cderi_uov[s] = util.get_cderi_mo(
                    with_df, mo_coeff_act[s], None, (0, nocc_act[s], nocc_act[s], nact[s]), max_memory)
                log.debug(f"Spin {s} ao2mo finished")
            self.tensors["cderi_uov"] = cderi_uov

        def gen_g_IJab(s0, s1, i, j):
            return cderi_uov[s0][:, i].T @ cderi_uov[s1][:, j]

        results = self.kernel_energy_iepa(
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

    kernel_energy_iepa = staticmethod(kernel_energy_uiepa)
    kernel = driver_eng_iepa


if __name__ == '__main__':
    def main_1():
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", charge=1, spin=1, basis="6-31G").build()
        mf_scf = scf.UHF(mol).run()
        mf_mp = UIEPAConv(mf_scf, frozen=[1, 2]).run()
        print(mf_mp.results)

    def main_2():
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", charge=1, spin=1, basis="6-31G").build()
        mf_scf = scf.UHF(mol).run()
        mf_mp = UIEPARI(mf_scf, frozen=[1, 2]).run()
        print(mf_mp.results)

    main_1()
    main_2()

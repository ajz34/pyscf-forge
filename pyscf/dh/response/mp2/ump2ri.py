""" RI-MP2 Response-Related Utilities. """

from pyscf.dh.response.mp2.rmp2ri import RMP2RespRI
from pyscf.dh import util, UMP2RI
from pyscf import scf, lib
from pyscf.lib.numpy_helper import ANTIHERMI
import h5py
import numpy as np

α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


def get_mp2_integrals(
        cderi_uov, mo_occ, mo_energy, mask_act,
        c_os, c_ss, incore_t_oovv,
        verbose=lib.logger.NOTE, max_memory=2000, h5file=NotImplemented):
    r""" Get 3-index transformed MP2 amplitude.

    .. math::
        t_{ij}^{ab} &= Y_{ia, P} Y_{jb, P} / D_{ij}^{ab} \\
        T_{ij}^{ab} &= 2 c_\mathrm{os} t_{ij}^{ab} - c_\mathrm{ss} t_{ij}^{ba} \\
        \Gamma_{ia, P} &= T_{ij}^{ab} Y_{jb, P} \\

    Parameters
    ----------
    cderi_uov : list[np.ndarray]
    mo_occ : np.ndarray
    mo_energy : np.ndarray
    mask_act : np.ndarray

    c_os : float
    c_ss : float
    incore_t_oovv : bool or str or float or int

    verbose : int
    max_memory : int or float
    h5file : h5py.File

    Returns
    -------
    dict, dict[str, float]
    """

    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    # dimension definition and check sanity
    assert mo_occ.ndim == mo_energy.ndim == mask_act.ndim == 2
    # todo: mask is not applied
    # mask_occ_act = (mo_occ != 0) & mask_act
    # mask_vir_act = (mo_occ == 0) & mask_act
    mask_occ_act = (mo_occ != 0)
    mask_vir_act = (mo_occ == 0)
    nocc_act = mask_occ_act.sum(axis=-1)
    nvir_act = mask_vir_act.sum(axis=-1)
    nmo = mo_occ.shape[-1]
    naux = cderi_uov[0].shape[α]
    assert (naux, nocc_act[α], nvir_act[α]) == cderi_uov[α].shape
    assert (naux, nocc_act[β], nvir_act[β]) == cderi_uov[β].shape
    assert incore_t_oovv is not None, "t_oovv must be stored for evaluation of MP2 response"

    eval_ss = abs(c_ss) > 1e-10
    eval_os = abs(c_os) > 1e-10

    so = [slice(0, nocc_act[σ]) for σ in (α, β)]  # active only
    sv = [slice(nocc_act[σ], nmo) for σ in (α, β)]  # active only

    # preparation
    eo = [mo_energy[σ][mask_occ_act[σ]] for σ in (α, β)]
    ev = [mo_energy[σ][mask_vir_act[σ]] for σ in (α, β)]

    # allocate results
    rdm1_corr = np.zeros((2, nmo, nmo))
    G_uov = [np.zeros((naux, nocc_act[σ], nvir_act[σ])) for σ in (α, β)]
    t_oovv = []
    D_ovv = []
    for (σ, ς, σς) in ((α, α, αα), (α, β, αβ), (β, β, ββ)):
        mem_avail = max_memory - lib.current_memory()[0]
        if (σ == ς and eval_ss) or (σ != ς and eval_os):
            D_ovv.append(lib.direct_sum("j - a - b -> jab", eo[ς], ev[σ], ev[ς]))
            t_oovv.append(util.allocate_array(
                incore_t_oovv, (nocc_act[σ], nocc_act[ς], nvir_act[σ], nvir_act[ς]), mem_avail,
                h5file=h5file, name=f"t_oovv_{σς}", zero_init=False, chunks=(1, 1, nvir_act[σ], nvir_act[ς])))
        else:
            D_ovv.append(None)
            t_oovv.append(None)
    eng_spin = np.array([0, 0, 0], dtype=float)

    # for async write t_oovv
    def write_t_oovv(σς, slc, buf):
        t_oovv[σς][slc] = buf

    # prepare batch
    mem_avail = max_memory - lib.current_memory()[0]
    nbatch = util.calc_batch_size(4 * max(nocc_act) * max(nvir_act)**2 + naux * max(nvir_act), mem_avail)
    with lib.call_in_background(write_t_oovv) as async_write_t_oovv:
        for σ, ς, σς in ((α, α, αα), (α, β, αβ), (β, β, ββ)):
            if (σ == ς and not eval_ss) or (σ != ς and not eval_os):
                continue
            for sI in util.gen_batch(0, nocc_act[σ], nbatch):
                log.debug(f"[DEBUG] Loop in get_mp2_integrals, spin {σ, ς} slice {sI} of {nocc_act} orbitals.")
                D_Oovv = eo[σ][sI, None, None, None] + D_ovv[σς]
                g_Oovv = lib.einsum("Pia, Pjb -> ijab", cderi_uov[σ][:, sI], cderi_uov[ς])
                t_Oovv_raw = g_Oovv / D_Oovv
                async_write_t_oovv(σς, sI, t_Oovv_raw)
                if σ == ς:  # same spin
                    # t_Oovv -= t_Oovv.swapaxes(-1, -2)
                    t_Oovv = util.hermi_sum_last2dim(t_Oovv_raw, hermi=ANTIHERMI, inplace=False)
                    eng_spin[σς] += 0.25 * np.einsum("Ijab, Ijab, Ijab ->", t_Oovv, t_Oovv, D_Oovv)
                else:
                    t_Oovv = t_Oovv_raw
                    eng_spin[σς] += np.einsum("Ijab, Ijab, Ijab ->", t_Oovv, t_Oovv, D_Oovv)

                if σ == ς:  # same spin
                    rdm1_corr[σ, so[σ], so[σ]] -= 0.5 * c_ss * lib.einsum("kiab, kjab -> ij", t_Oovv, t_Oovv)
                    rdm1_corr[σ, sv[σ], sv[σ]] += 0.5 * c_ss * lib.einsum("ijac, ijbc -> ab", t_Oovv, t_Oovv)
                    G_uov[σ][:, sI] += c_ss * lib.einsum("ijab, Pjb -> Pia", t_Oovv, cderi_uov[σ])
                else:  # spin αβ
                    for sJ in util.gen_batch(0, nocc_act[α], nbatch):
                        if sI.start < sJ.start:
                            continue
                        t_Ikab = t_Oovv
                        t_Jkab = t_Oovv if sI == sJ else t_oovv[αβ][sJ]
                        rdm1_tmp = c_os * lib.einsum("ikab, jkab -> ij", t_Ikab, t_Jkab)
                        rdm1_corr[α][sI, sJ] -= rdm1_tmp
                        if sI != sJ:
                            rdm1_corr[α][sJ, sI] -= rdm1_tmp.T
                    rdm1_corr[β, so[β], so[β]] -= c_os * lib.einsum("kiba, kjba -> ij", t_Oovv, t_Oovv)
                    rdm1_corr[α, sv[α], sv[α]] += c_os * lib.einsum("ijac, ijbc -> ab", t_Oovv, t_Oovv)
                    rdm1_corr[β, sv[β], sv[β]] += c_os * lib.einsum("jica, jicb -> ab", t_Oovv, t_Oovv)
                    G_uov[α][:, sI] += c_os * lib.einsum("ijab, Pjb -> Pia", t_Oovv, cderi_uov[β])
                    G_uov[β] += c_os * lib.einsum("jiba, Pjb -> Pia", t_Oovv, cderi_uov[α][:, sI])

    eng_mp2 = c_os * eng_spin[1] + c_ss * (eng_spin[0] + eng_spin[2])

    # results
    results = dict()
    results["eng_corr_MP2_aa"] = eng_spin[0]
    results["eng_corr_MP2_ab"] = eng_spin[1]
    results["eng_corr_MP2_bb"] = eng_spin[2]
    results["eng_corr_MP2_OS"] = eng_spin[1]
    results["eng_corr_MP2_SS"] = eng_spin[0] + eng_spin[2]
    results["eng_corr_MP2"] = eng_mp2
    log.note(f"[RESULT] Energy corr MP2 of spin aa: {eng_spin[0]:18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of spin ab: {eng_spin[1]:18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of spin bb: {eng_spin[2]:18.10f}")
    log.note(f"[RESULT] Energy corr MP2 of total  : {eng_mp2    :18.10f}")

    # tensors
    tensors = {
        "t_oovv": t_oovv,
        "G_uov": G_uov,
        "rdm1_corr": rdm1_corr}

    log.timer("get_mp2_integrals", *time0)
    return tensors, results


def get_W_I(cderi_uov, cderi_uoo, G_uov, verbose=lib.logger.NOTE):
    r""" Part I of MO-energy-weighted density matrix.

    Parameters
    ----------
    cderi_uov : list[np.ndarray]
    cderi_uoo : list[np.ndarray]
    G_uov : list[np.ndarray]
    verbose : int

    Returns
    -------
    dict[str, np.ndarray]
    """

    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    # dimension definition and check sanity
    nocc = [cderi_uov[σ].shape[1] for σ in (α, β)]
    nvir = [cderi_uov[σ].shape[2] for σ in (α, β)]

    nmo = nocc[0] + nvir[0]
    if nocc[1] + nvir[1] != nmo:
        raise NotImplementedError("Frozen core not implemented.")

    so = [slice(0, nocc[σ]) for σ in (α, β)]  # active only
    sv = [slice(nocc[σ], nmo) for σ in (α, β)]  # active only

    W_I = np.zeros((2, nmo, nmo))
    for σ in (α, β):
        W_I[σ, so[σ], so[σ]] = - lib.einsum("Pia, Pja -> ij", G_uov[σ], cderi_uov[σ])
        W_I[σ, sv[σ], sv[σ]] = - lib.einsum("Pia, Pib -> ab", G_uov[σ], cderi_uov[σ])
        W_I[σ, sv[σ], so[σ]] = - 2 * lib.einsum("Pja, Pij -> ai", G_uov[σ], cderi_uoo[σ])

    log.timer("get_W_I", *time0)
    tensors = {"W_I": W_I}
    return tensors


def get_lag_vo(
        G_uov, cderi_uaa, W_I, rdm1_corr, Ax0_Core,
        max_memory=2000, verbose=lib.logger.NOTE):
    r""" MP2 contribution to Lagrangian vir-occ block.

    Parameters
    ----------
    G_uov : list[np.ndarray]
    cderi_uaa : list[np.ndarray]
    W_I : np.ndarray
    rdm1_corr : np.ndarray
    Ax0_Core : callable
    max_memory : float
    verbose : int

    Returns
    -------
    dict[str, np.ndarray]
    """

    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    naux = G_uov[0].shape[0]
    nocc = [G_uov[σ].shape[1] for σ in (α, β)]
    nvir = [G_uov[σ].shape[2] for σ in (α, β)]
    nmo = nocc[0] + nvir[0]

    # prepare essential matrices and slices
    so = [slice(0, nocc[σ]) for σ in (α, β)]
    sv = [slice(nocc[σ], nmo) for σ in (α, β)]
    sa = [slice(0, nmo) for σ in (α, β)]

    # generate lagrangian occ-vir block
    lag_vo = Ax0_Core(sv, so, sa, sa)(rdm1_corr)
    for σ in α, β:
        lag_vo[σ] += W_I[σ][sv[σ], so[σ]]

        def load_cderi_Uaa(slc):
            return cderi_uaa[σ][slc, sv[σ], sv[σ]]

        mem_avail = max_memory - lib.current_memory()[0]
        nbatch = util.calc_batch_size(max(nvir) ** 2 + max(nocc) * max(nvir), mem_avail)
        batches = util.gen_batch(0, naux, nbatch)
        for saux, cderi_Uaa in zip(batches, lib.map_with_prefetch(load_cderi_Uaa, batches)):
            lag_vo[σ] += 2 * lib.einsum("Pib, Pab -> ai", G_uov[σ][saux], cderi_Uaa)

    log.timer("get_lag_vo", *time0)
    tensors = {"lag_vo": lag_vo}
    return tensors


class UMP2RespRI(UMP2RI, RMP2RespRI):

    def make_cderi_uaa(self):
        if "cderi_uaa" in self.tensors:
            return self.tensors["cderi_uaa"]

        log = lib.logger.new_logger(verbose=self.verbose)

        # dimension and mask
        mo_coeff = self.mo_coeff
        nmo = self.nmo

        # density fitting preparation
        omega = self.omega
        with_df = util.get_with_df_omega(self.with_df, omega)
        naux = with_df.get_naoaux()

        # array allocation
        incore_cderi_uaa = self.incore_cderi_uaa
        cderi_uaa = []
        for σ in α, β:
            mem_avail = self.max_memory - lib.current_memory()[0]
            cderi_uaa.append(util.allocate_array(
                incore_cderi_uaa, (naux, nmo, nmo), mem_avail,
                h5file=self._tmpfile, name=f"cderi_uaa_{σ}", zero_init=False))
            log.info(f"[INFO] Store type of cderi_uaa (spin {σ}): {type(cderi_uaa)}")

        # generate array
        for σ in α, β:
            util.get_cderi_mo(with_df, mo_coeff[σ], cderi_uaa[σ], max_memory=self.max_memory)

        tensors = {"cderi_uaa": cderi_uaa}
        self.tensors.update(tensors)
        return cderi_uaa

    def make_cderi_uov(self):
        if "cderi_uov" in self.tensors:
            return self.tensors["cderi_uov"]

        # dimension and mask
        mask_occ = self.mask_occ
        mask_vir = self.mask_vir

        cderi_uaa = self.tensors.get("cderi_uaa", self.make_cderi_uaa())
        cderi_uov = [cderi_uaa[σ][:, mask_occ[σ], :][:, :, mask_vir[σ]] for σ in (α, β)]

        tensors = {"cderi_uov": cderi_uov}
        self.tensors.update(tensors)
        return cderi_uov

    def make_cderi_uoo(self):
        if "cderi_uoo" in self.tensors:
            return self.tensors["cderi_uoo"]

        # dimension and mask
        mask_occ = self.mask_occ

        cderi_uaa = self.tensors.get("cderi_uaa", self.make_cderi_uaa())
        cderi_uoo = [cderi_uaa[σ][:, mask_occ[σ], :][:, :, mask_occ[σ]] for σ in (α, β)]

        tensors = {"cderi_uoo": cderi_uoo}
        self.tensors.update(tensors)
        return cderi_uoo

    def make_rdm1_corr_resp(self):
        """ Generate 1-RDM (response) of MP2 correlation :math:`D_{pq}^{(2)}`. """
        if "rdm1_corr_resp" in self.tensors:
            return self.tensors["rdm1_corr_resp"]

        # prepare input
        lag_vo = self.make_lag_vo()
        rdm1_corr = self.tensors["rdm1_corr"]

        # main function
        rdm1_corr_resp_vo = self.solve_cpks(lag_vo)
        rdm1_corr_resp = rdm1_corr.copy()
        mask_occ, mask_vir = self.mask_occ, self.mask_vir
        for σ in α, β:
            rdm1_corr_resp[np.ix_([σ], mask_vir[σ], mask_occ[σ])] = rdm1_corr_resp_vo[σ]

        self.tensors["rdm1_corr_resp"] = rdm1_corr_resp
        return rdm1_corr_resp

    def make_rdm1(self, ao_repr=False):
        r""" Generate 1-RDM (non-response) of MP2 :math:`D_{pq}^{MP2}` in MO or :math:`D_{\mu \nu}^{MP2}` in AO. """
        # prepare input
        rdm1_corr = self.tensors.get("rdm1_corr", self.make_rdm1_corr())

        rdm1 = np.array([np.diag(self.mo_occ[σ]) for σ in (α, β)])
        rdm1 += rdm1_corr
        self.tensors["rdm1"] = rdm1
        if ao_repr:
            rdm1 = np.array([self.mo_coeff[σ] @ rdm1[σ] @ self.mo_coeff[σ].T for σ in (α, β)])
        return rdm1

    def make_rdm1_resp(self, ao_repr=False):
        r""" Generate 1-RDM (response) of MP2 :math:`D_{pq}^{MP2}` in MO or :math:`D_{\mu \nu}^{MP2}` in AO. """
        # prepare input
        rdm1_corr_resp = self.tensors.get("rdm1_corr_resp", self.make_rdm1_corr_resp())

        rdm1_resp = np.array([np.diag(self.mo_occ[σ]) for σ in (α, β)])
        rdm1_resp += rdm1_corr_resp

        self.tensors["rdm1_resp"] = rdm1_resp
        if ao_repr:
            rdm1_resp = np.array([self.mo_coeff[σ] @ rdm1_resp[σ] @ self.mo_coeff[σ].T for σ in (α, β)])
        return rdm1_resp

    driver_eng_mp2 = RMP2RespRI.driver_eng_mp2
    kernel = driver_eng_mp2

    get_mp2_integrals = staticmethod(get_mp2_integrals)
    get_W_I = staticmethod(get_W_I)
    get_lag_vo = staticmethod(get_lag_vo)


if __name__ == '__main__':
    def main_1():
        from pyscf import gto, scf
        from pyscf.dh import UMP2RI
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1).build()
        mf = scf.UHF(mol).run()
        mf_mp2 = UMP2RI(mf).run()
        mf_resp = UMP2RespRI(mf)
        mf_resp.driver_eng_mp2()
        print(mf_mp2.results)
        print(mf_resp.results)

    def main_2():
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1).build()
        mf = scf.UHF(mol).run()
        mf_resp = UMP2RespRI(mf)
        print(mf_resp.make_dipole())

    def main_3():
        # test B3LYP -> CAM-B3LYP non-consistent functional dipole

        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=0, spin=0).build()
        c_os, c_ss = 1.3, 0.6

        mf = dft.UKS(mol, xc="HF").density_fit("cc-pVDZ-jkfit").run()
        mf_hdft = UMP2RespRI(mf, c_os=c_os, c_ss=c_ss)
        dip_anal = mf_hdft.make_dipole()

        # REF = np.array([0.6039302487, 0., 0.7799863729])
        # self.assertTrue(np.allclose(dip_anal, REF, atol=1e-5, rtol=1e-7))

        # generation of numerical result

        def eng_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="HF").density_fit("cc-pVDZ-jkfit").run()
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_mp = UMP2RespRI(mf_scf).run(c_os=c_os, c_ss=c_ss)
            return mf_mp.scf.e_tot + c_os * mf_mp.results["eng_corr_MP2_OS"] + c_ss * mf_mp.results["eng_corr_MP2_SS"]

        eng_array = np.zeros((2, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                eng_array[idx, t] = eng_with_dipole_field(t, h)
        dip_elec_num = - (eng_array[0] - eng_array[1]) / (2 * h)
        dip_nuc = np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())
        dip_num = dip_elec_num + dip_nuc

        print(dip_anal)
        print(dip_num)

        # self.assertTrue(np.allclose(dip_num, dip_anal, atol=1e-5, rtol=1e-7))

    def main_4():
        # test B3LYP -> CAM-B3LYP non-consistent functional dipole

        from pyscf import gto, scf, dft
        np.set_printoptions(5, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=0, spin=0).build()
        c_os, c_ss = 1.3, 0.6

        mf_r = dft.RKS(mol, xc="HF").density_fit("cc-pVDZ-jkfit").run(conv_tol=1e-12)
        mf_rmp2 = RMP2RespRI(mf_r, c_os=c_os, c_ss=c_ss)
        dip_r = mf_rmp2.make_dipole()

        mf_u = mf_r.to_uks()
        mf_u.mo_coeff = np.array(mf_u.mo_coeff)
        mf_u.mo_occ = np.array(mf_u.mo_occ)
        mf_u.mo_energy = np.array(mf_u.mo_energy)
        mf_ump2 = UMP2RespRI(mf_u, c_os=c_os, c_ss=c_ss)
        dip_u = mf_ump2.make_dipole()

        print(dip_u)
        print(dip_r)

        # print(mf_ump2.make_rdm1_corr()[0] * 2)
        # print(mf_rmp2.make_rdm1_corr())
        print(abs(mf_ump2.make_rdm1_corr()[0] * 2 - mf_rmp2.make_rdm1_corr()).max())
        print(abs(mf_ump2.make_G_uov()[0] - mf_rmp2.make_G_uov()).max())

        # self.assertTrue(np.allclose(dip_num, dip_anal, atol=1e-5, rtol=1e-7))

    main_3()

""" RI-MP2 Response-Related Utilities. """

from pyscf.dh.response.mp2.rmp2ri import RMP2RespRI
from pyscf.dh import util
from pyscf import scf, lib
from pyscf.lib.numpy_helper import ANTIHERMI
from pyscf.dh.response import RespBase
from pyscf.dh.response.respbase import get_rdm1_resp_vo_restricted
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

    so = [slice(0, nocc_act[σ]) for σ in (α, β)]  # active only
    sv = [slice(nocc_act[σ], nmo) for σ in (α, β)]  # active only

    # preparation
    eo = [mo_energy[σ][mask_occ_act[σ]] for σ in (α, β)]
    ev = [mo_energy[σ][mask_vir_act[σ]] for σ in (α, β)]
    D_ovv = [lib.direct_sum("j - a - b -> jab", eo[ς], ev[σ], ev[ς]) for (σ, ς) in ((α, α), (α, β), (β, β))]

    # allocate results
    rdm1_corr = np.zeros((2, nmo, nmo))
    G_uov = [np.zeros((naux, nocc_act[σ], nvir_act[σ])) for σ in (α, β)]
    t_oovv = []
    for (σ, ς, σς) in ((α, α, αα), (α, β, αβ), (β, β, ββ)):
        mem_avail = max_memory - lib.current_memory()[0]
        t_oovv.append(util.allocate_array(
            incore_t_oovv, (nocc_act[σ], nocc_act[ς], nvir_act[σ], nvir_act[ς]), mem_avail,
            h5file=h5file, name=f"t_oovv_{σς}", zero_init=False, chunk=(1, 1, nvir_act[σ], nvir_act[ς])))
    eng_spin = np.array([0, 0, 0], dtype=float)

    # for async write t_oovv
    def write_t_oovv(σς, slc, buf):
        t_oovv[σς][slc] = buf

    # prepare batch
    mem_avail = max_memory - lib.current_memory()[0]
    nbatch = util.calc_batch_size(4 * max(nocc_act) * max(nvir_act)**2 + naux * max(nvir_act), mem_avail)

    with lib.call_in_background(write_t_oovv) as async_write_t_oovv:
        for σ, ς, σς in ((α, α, αα), (α, β, αβ), (β, β, ββ)):
            for sI in util.gen_batch(0, nocc_act[σ], nbatch):
                log.debug(f"[DEBUG] Loop in get_mp2_integrals, spin {σ, ς} slice {sI} of {nocc_act} orbitals.")
                D_Oovv = eo[σ][sI, None, None, None] + D_ovv[σς]
                g_Oovv = lib.einsum("Pia, Pjb -> ijab", cderi_uov[σ][:, sI], cderi_uov[ς])
                t_Oovv = g_Oovv / D_Oovv
                if σ == ς:  # same spin
                    # t_Oovv -= t_Oovv.swapaxes(-1, -2)
                    util.hermi_sum_last2dim(t_Oovv, hermi=ANTIHERMI, inplace=True)
                async_write_t_oovv(σς, sI, t_Oovv)
                eng_spin[σς] += np.einsum("Ijab, Ijab, Ijab ->", t_Oovv, t_Oovv, D_Oovv)

                if σ == ς:  # same spin
                    rdm1_corr[σ, so[σ], so[σ]] -= 2 * c_ss * lib.einsum("kiab, kjab -> ij", t_Oovv, t_Oovv)
                    rdm1_corr[σ, sv[σ], sv[σ]] += 2 * c_ss * lib.einsum("ijac, ijbc -> ab", t_Oovv, t_Oovv)
                    G_uov[σ][:, sI] += 4 * c_ss * lib.einsum("ijab, Pjb -> Pia", t_Oovv, cderi_uov[σ])
                else:  # spin αβ
                    for sJ in util.gen_batch(0, nocc_act[α], nbatch):
                        t_Ikab = t_Oovv
                        t_Jkab = t_Oovv if sI == sJ else t_oovv[αβ][sJ]
                        rdm1_tmp = c_os * lib.einsum("ikab, jkab -> ij", t_Ikab, t_Jkab)
                        rdm1_corr[α][sI, sJ] -= rdm1_tmp
                        if sI != sJ:
                            rdm1_corr[α][sJ, sI] -= rdm1_tmp.T
                    rdm1_corr[β, so[β], so[β]] -= c_os * lib.einsum("kiba, kjba -> ij", t_Oovv, t_Oovv)
                    rdm1_corr[α, sv[α], sv[α]] += c_os * lib.einsum("ijac, ijbc -> ab", t_Oovv, t_Oovv)
                    rdm1_corr[β, sv[β], sv[β]] += c_os * lib.einsum("jica, jicb -> ab", t_Oovv, t_Oovv)
                    G_uov[α][:, sI] += 2 * c_os * lib.einsum("ijab, Pjb -> Pia", t_Oovv, cderi_uov[β])
                    G_uov[β] += 2 * c_os * lib.einsum("jiba, Pjb -> Pia", t_Oovv, cderi_uov[α][:, sI])

    eng_spin[0] *= 0.25 * c_ss
    eng_spin[1] *= c_os
    eng_spin[2] *= 0.25 * c_ss
    eng_mp2 = eng_spin[1] + (eng_spin[0] + eng_spin[2])

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


class UMP2RespRI(RMP2RespRI):

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
        log.info(f"[INFO] Store type of cderi_uaa: {type(cderi_uaa)}")

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
        if "cderi_uov" in self.tensors:
            return self.tensors["cderi_uov"]

        # dimension and mask
        mask_occ = self.mask_occ
        mask_vir = self.mask_vir

        cderi_uaa = self.tensors.get("cderi_uaa", self.make_cderi_uaa())
        cderi_uoo = [cderi_uaa[σ][:, mask_occ[σ], :][:, :, mask_occ[σ]] for σ in (α, β)]

        tensors = {"cderi_uoo": cderi_uoo}
        self.tensors.update(tensors)
        return cderi_uoo

    get_mp2_integrals = staticmethod(get_mp2_integrals)


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

    main_1()

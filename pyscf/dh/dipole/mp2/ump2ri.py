from pyscf.dh.dipole.dipolebase import DipoleBase, PolarBase
from pyscf.dh.response.mp2.ump2ri import UMP2RespRI
from pyscf.dh.dipole.mp2.rmp2ri import RMP2DipoleRI, RMP2PolarRI
from pyscf.dh import util
from pyscf import lib
from pyscf.lib.numpy_helper import ANTIHERMI
import numpy as np

α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


def get_pd_cderi_uov(
        cderi_uaa, U_1, mo_occ,
        verbose=lib.logger.NOTE,
        max_memory=2000):
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    nprop = U_1.shape[1]
    nocc = (mo_occ != 0).sum(axis=-1)
    nvir = (mo_occ == 0).sum(axis=-1)
    nmo = mo_occ.shape[-1]
    naux = cderi_uaa[0].shape[0]

    so = [slice(0, nocc[σ]) for σ in (α, β)]
    sv = [slice(nocc[σ], nmo) for σ in (α, β)]

    pd_cderi_uov = [np.zeros((nprop, naux, nocc[σ], nvir[σ])) for σ in (α, β)]
    for σ in α, β:

        def load_cderi_Uaa(slc):
            return cderi_uaa[σ][slc]

        mem_avail = max_memory - lib.current_memory()[0]
        nbatch = util.calc_batch_size(2 * nmo**2 + 2 * nprop * max(nocc) * max(nvir), mem_avail)
        batches = util.gen_batch(0, naux, nbatch)
        for saux, cderi_Uaa in zip(batches, lib.map_with_prefetch(load_cderi_Uaa, batches)):
            pd_cderi_uov[σ][:, saux] = (
                + lib.einsum("Ami, Pma -> APia", U_1[σ][:, :, so[σ]], cderi_Uaa[:, :, sv[σ]])
                + lib.einsum("Ama, Pmi -> APia", U_1[σ][:, :, sv[σ]], cderi_Uaa[:, :, so[σ]]))

    log.timer("get_pd_cderi_uov of dipole", *time0)
    return pd_cderi_uov


def get_mp2_integrals_deriv(
        cderi_uov, pd_cderi_uov, t_oovv, pd_fock_mo,
        mo_occ, mo_energy, c_os, c_ss,
        verbose=lib.logger.NOTE,
        max_memory=2000):
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    nprop, naux = pd_cderi_uov[0].shape[0:2]
    nocc = (mo_occ != 0).sum(axis=-1)
    nvir = (mo_occ == 0).sum(axis=-1)
    nmo = mo_occ.shape[-1]

    so = [slice(0, nocc[σ]) for σ in (α, β)]
    sv = [slice(nocc[σ], nmo) for σ in (α, β)]

    pd_G_uov = [np.zeros((nprop, naux, nocc[σ], nvir[σ])) for σ in (α, β)]
    pd_rdm1_corr = np.zeros((2, nprop, nmo, nmo))

    for σς, σ, ς in (αα, α, α), (αβ, α, β), (ββ, β, β):
        D_ovv = lib.direct_sum("j - a - b -> jab", mo_energy[ς][so[ς]], mo_energy[σ][sv[σ]], mo_energy[ς][sv[ς]])
        mem_avail = max_memory - lib.current_memory()[0]
        nbatch = util.calc_batch_size(4 * nprop * max(nocc) * max(nvir)**2, mem_avail)
        for sI in util.gen_batch(0, nocc[σ], nbatch):
            t_Oovv = np.asarray(t_oovv[σς][sI])
            D_Oovv = lib.direct_sum("i + jab -> ijab", mo_energy[σ][so[σ]], D_ovv)

            pd_t_Oovv = (
                + lib.einsum("APia, Pjb -> Aijab", pd_cderi_uov[σ][:, :, sI], cderi_uov[ς])
                + lib.einsum("APjb, Pia -> Aijab", pd_cderi_uov[ς], cderi_uov[σ][:, sI]))

            for sK in util.gen_batch(0, nocc[σ], nbatch):
                t_Ijab = t_Oovv
                t_Kjab = t_Ijab if sK == sI else t_oovv[σς][sK]
                pd_t_Oovv -= lib.einsum("Aki, kjab -> Aijab", pd_fock_mo[σ][:, sK, sI], t_Kjab)
            pd_t_Oovv -= lib.einsum("Akj, ikab -> Aijab", pd_fock_mo[ς][:, so[ς], so[ς]], t_Oovv)
            pd_t_Oovv += lib.einsum("Aca, ijcb -> Aijab", pd_fock_mo[σ][:, sv[σ], sv[σ]], t_Oovv)
            pd_t_Oovv += lib.einsum("Acb, ijac -> Aijab", pd_fock_mo[ς][:, sv[ς], sv[ς]], t_Oovv)
            pd_t_Oovv /= D_Oovv

            if σ == ς:  # same spin
                util.hermi_sum_last2dim(t_Oovv, hermi=ANTIHERMI, inplace=True)
                util.hermi_sum_last2dim(pd_t_Oovv, hermi=ANTIHERMI, inplace=True)

                pd_rdm1_corr[σ, :, so[σ], so[σ]] -= 0.5 * c_ss * lib.einsum("kiab, Akjab -> Aij", t_Oovv, pd_t_Oovv)
                pd_rdm1_corr[σ, :, sv[σ], sv[σ]] += 0.5 * c_ss * lib.einsum("ijac, Aijbc -> Aab", t_Oovv, pd_t_Oovv)
                pd_G_uov[σ][:, :, sI] += c_ss * lib.einsum("ijab, APjb -> APia", t_Oovv, pd_cderi_uov[σ])
                pd_G_uov[σ][:, :, sI] += c_ss * lib.einsum("Aijab, Pjb -> APia", pd_t_Oovv, cderi_uov[σ])
            else:  # spin αβ
                for sJ in util.gen_batch(0, nocc[α], nbatch):
                    t_Jkab = t_Oovv if sI == sJ else t_oovv[αβ][sJ]
                    pd_rdm1_corr[α, :, sI, sJ] -= c_os * lib.einsum("jkba, Aikba -> Aij", t_Jkab, pd_t_Oovv)
                pd_rdm1_corr[β, :, so[β], so[β]] -= c_os * lib.einsum("kiba, Akjba -> Aij", t_Oovv, pd_t_Oovv)
                pd_rdm1_corr[α, :, sv[α], sv[α]] += c_os * lib.einsum("ijac, Aijbc -> Aab", t_Oovv, pd_t_Oovv)
                pd_rdm1_corr[β, :, sv[β], sv[β]] += c_os * lib.einsum("jica, Ajicb -> Aab", t_Oovv, pd_t_Oovv)
                pd_G_uov[α][:, :, sI] += c_os * lib.einsum("ijab, APjb -> APia", t_Oovv, pd_cderi_uov[β])
                pd_G_uov[α][:, :, sI] += c_os * lib.einsum("Aijab, Pjb -> APia", pd_t_Oovv, cderi_uov[β])
                pd_G_uov[β] += c_os * lib.einsum("jiba, APjb -> APia", t_Oovv, pd_cderi_uov[α][:, :, sI])
                pd_G_uov[β] += c_os * lib.einsum("Ajiba, Pjb -> APia", pd_t_Oovv, cderi_uov[α][:, sI])

    pd_rdm1_corr += pd_rdm1_corr.swapaxes(-1, -2)

    tensors = {
        "pd_G_uov": pd_G_uov,
        "pd_rdm1_corr": pd_rdm1_corr,
    }

    log.timer("get_mp2_integrals_deriv of dipole", *time0)
    return tensors


def get_SCR3(
        cderi_uaa, G_uov, pd_G_uov, U_1,
        mo_occ,
        verbose=lib.logger.NOTE,
        max_memory=2000):
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    nprop = U_1.shape[1]
    nocc = (mo_occ != 0).sum(axis=-1)
    nvir = (mo_occ == 0).sum(axis=-1)
    nmo = mo_occ.shape[-1]
    naux = cderi_uaa[0].shape[0]

    so = [slice(0, nocc[σ]) for σ in (α, β)]
    sv = [slice(nocc[σ], nmo) for σ in (α, β)]

    SCR3 = [np.zeros((nprop, nvir[σ], nocc[σ])) for σ in (α, β)]

    for σ in α, β:

        def load_cderi_Uaa(slc):
            return cderi_uaa[σ][slc]

        mem_avail = max_memory - lib.current_memory()[0]
        nbatch = util.calc_batch_size(2 * nprop * nmo**2, mem_avail)
        batches = util.gen_batch(0, naux, nbatch)
        for saux, cderi_Uaa in zip(batches, lib.map_with_prefetch(load_cderi_Uaa, batches)):
            G_Uov = G_uov[σ][saux]
            pd_G_Uov = pd_G_uov[σ][:, saux]

            # occ-occ part
            pd_cderi_Uoo = lib.einsum("Ami, Pmj -> APij", U_1[σ][:, :, so[σ]], cderi_Uaa[:, :, so[σ]])
            # pd_cderi_Uoo += pd_cderi_Uoo.swapaxes(-1, -2)
            util.hermi_sum_last2dim(pd_cderi_Uoo)
            SCR3[σ] -= 2 * lib.einsum("APja, Pij -> Aai", pd_G_Uov, cderi_Uaa[:, so[σ], so[σ]])
            SCR3[σ] -= 2 * lib.einsum("Pja, APij -> Aai", G_Uov, pd_cderi_Uoo)

            # vir-vir part
            pd_cderi_Uvv = lib.einsum("Ama, Pmb -> APab", U_1[σ][:, :, sv[σ]], cderi_Uaa[:, :, sv[σ]])
            # pd_cderi_Uvv += pd_cderi_Uvv.swapaxes(-1, -2)
            util.hermi_sum_last2dim(pd_cderi_Uvv)
            SCR3[σ] += 2 * lib.einsum("APib, Pab -> Aai", pd_G_Uov, cderi_Uaa[:, sv[σ], sv[σ]])
            SCR3[σ] += 2 * lib.einsum("Pib, APab -> Aai", G_Uov, pd_cderi_Uvv)

    log.timer("get_SCR3 of dipole", *time0)
    return SCR3


class UMP2DipoleRI(UMP2RespRI, RMP2DipoleRI):

    get_pd_cderi_uov = staticmethod(get_pd_cderi_uov)
    get_mp2_integrals_deriv = staticmethod(get_mp2_integrals_deriv)
    get_SCR3 = staticmethod(get_SCR3)


class UMP2PolarRI(UMP2RespRI, PolarBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.deriv_dipole = UMP2DipoleRI.from_cls(self, self.scf, copy_all=True)


if __name__ == '__main__':

    def main_1():
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", charge=0, spin=0, basis="6-31G", verbose=0).build()
        mf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = UMP2PolarRI(mf)
        mf_pol.c_os = 0
        mf_pol.c_ss = 1
        print(mf_pol.de)

    def main_2():
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", charge=0, spin=0, basis="6-31G", verbose=0).build()
        mf = dft.RKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = RMP2PolarRI(mf)
        mf_pol.c_os = 0
        mf_pol.c_ss = 1
        print(mf_pol.de)

    main_1()
    main_2()

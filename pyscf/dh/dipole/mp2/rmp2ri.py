from pyscf.dh.dipole.dipolebase import DipoleBase, PolarBase
from pyscf.dh.response.mp2.rmp2ri import RMP2RespRI
from pyscf.dh import util
from pyscf import lib
import numpy as np


def get_pd_cderi_uov(
        cderi_uaa, U_1, mo_occ,
        verbose=lib.logger.NOTE,
        max_memory=2000):
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    nprop = len(U_1)
    nocc = (mo_occ != 0).sum()
    nvir = (mo_occ == 0).sum()
    nmo = len(mo_occ)
    naux = cderi_uaa.shape[0]
    so, sv = slice(0, nocc), slice(nocc, nmo)

    pd_cderi_uov = np.zeros((nprop, naux, nocc, nvir))

    def load_cderi_Uaa(slc):
        return cderi_uaa[slc]

    mem_avail = max_memory - lib.current_memory()[0]
    nbatch = util.calc_batch_size(2 * nmo**2 + 2 * nprop * nocc * nvir, mem_avail)
    batches = util.gen_batch(0, naux, nbatch)
    for saux, cderi_Uaa in zip(batches, lib.map_with_prefetch(load_cderi_Uaa, batches)):
        pd_cderi_uov[:, saux] = (
            + lib.einsum("Ami, Pma -> APia", U_1[:, :, so], cderi_Uaa[:, :, sv])
            + lib.einsum("Ama, Pmi -> APia", U_1[:, :, sv], cderi_Uaa[:, :, so]))

    log.timer("get_pd_cderi_uov of dipole", *time0)
    return pd_cderi_uov


def get_mp2_integrals_deriv(
        cderi_uov, pd_cderi_uov, t_oovv, pd_fock_mo,
        mo_occ, mo_energy, c_os, c_ss,
        verbose=lib.logger.NOTE,
        max_memory=2000):
    """ Get various derivative related MP2 integrals (including non-response density matrix, 3-index amplitude).

    Parameters
    ----------
    cderi_uov : np.ndarray
    pd_cderi_uov : np.ndarray
    t_oovv : np.ndarray
    pd_fock_mo : np.ndarray
    mo_occ : np.ndarray
    mo_energy : np.ndarray
    c_os : float
    c_ss : float
    verbose : int
    max_memory : float

    Returns
    -------
    dict
    """
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    nprop = len(pd_cderi_uov)
    nocc = (mo_occ != 0).sum()
    nvir = (mo_occ == 0).sum()
    nmo = len(mo_occ)
    naux = cderi_uov.shape[0]
    so, sv = slice(0, nocc), slice(nocc, nmo)

    pd_G_uov = np.zeros((nprop, naux, nocc, nvir))
    pd_rdm1_corr = np.zeros((nprop, nmo, nmo))

    D_ovv = lib.direct_sum("j - a - b -> jab", mo_energy[so], mo_energy[sv], mo_energy[sv])

    mem_avail = max_memory - lib.current_memory()[0]
    nbatch = util.calc_batch_size(4 * nprop * nocc * nvir**2, mem_avail)
    for sI in util.gen_batch(0, nocc, nbatch):
        t_Oovv = np.asarray(t_oovv[sI])
        D_Oovv = lib.direct_sum("i + jab -> ijab", mo_energy[sI], D_ovv)

        pd_t_Oovv = (
            + lib.einsum("APia, Pjb -> Aijab", pd_cderi_uov[:, :, sI], cderi_uov)
            + lib.einsum("APjb, Pia -> Aijab", pd_cderi_uov, cderi_uov[:, sI]))

        for sK in util.gen_batch(0, nocc, nbatch):
            t_Ijab = t_Oovv
            t_Kjab = t_Ijab if sK == sI else t_oovv[sK]
            pd_t_Oovv -= lib.einsum("Aki, kjab -> Aijab", pd_fock_mo[:, sK, sI], t_Kjab)
        pd_t_Oovv -= lib.einsum("Akj, ikab -> Aijab", pd_fock_mo[:, so, so], t_Oovv)
        pd_t_Oovv += lib.einsum("Aca, ijcb -> Aijab", pd_fock_mo[:, sv, sv], t_Oovv)
        pd_t_Oovv += lib.einsum("Acb, ijac -> Aijab", pd_fock_mo[:, sv, sv], t_Oovv)
        pd_t_Oovv /= D_Oovv

        T_Oovv = util.restricted_biorthogonalize(t_Oovv, 1, c_os, c_ss)
        pd_T_Oovv = util.restricted_biorthogonalize(pd_t_Oovv, 1, c_os, c_ss)

        pd_G_uov[:, :, sI] = (
            + lib.einsum("Aijab, Pjb -> APia", pd_T_Oovv, cderi_uov)
            + lib.einsum("ijab, APjb -> APia", T_Oovv, pd_cderi_uov))
        pd_rdm1_corr[:, so, so] -= 2 * lib.einsum("kiab, Akjab -> Aij", T_Oovv, pd_t_Oovv)
        pd_rdm1_corr[:, sv, sv] += 2 * lib.einsum("ijac, Aijbc -> Aab", T_Oovv, pd_t_Oovv)
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
    """ Get derivative to lagrangian (without derivative to fock response with correlated density).

    Parameters
    ----------
    cderi_uaa : np.ndarray
    G_uov : np.ndarray
    pd_G_uov : np.ndarray
    U_1 : np.ndarray
    mo_occ : np.ndarray
    verbose : int
    max_memory : float

    Returns
    -------
    np.ndarray
    """
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    nocc = (mo_occ != 0).sum()
    nvir = (mo_occ == 0).sum()
    nmo = len(mo_occ)
    naux = len(G_uov)
    nprop = len(U_1)
    so, sv = slice(0, nocc), slice(nocc, nmo)

    def load_cderi_Uaa(slc):
        return cderi_uaa[slc]

    SCR3 = np.zeros((nprop, nvir, nocc))

    mem_avail = max_memory - lib.current_memory()[0]
    nbatch = util.calc_batch_size(2 * nprop * nmo**2, mem_avail)
    batches = util.gen_batch(0, naux, nbatch)
    for saux, cderi_Uaa in zip(batches, lib.map_with_prefetch(load_cderi_Uaa, batches)):
        G_Uov = G_uov[saux]
        pd_G_Uov = pd_G_uov[:, saux]

        # occ-occ part
        pd_cderi_Uoo = lib.einsum("Ami, Pmj -> APij", U_1[:, :, so], cderi_Uaa[:, :, so])
        # pd_cderi_Uoo += pd_cderi_Uoo.swapaxes(-1, -2)
        util.hermi_sum_last2dim(pd_cderi_Uoo)
        SCR3 -= 4 * lib.einsum("APja, Pij -> Aai", pd_G_Uov, cderi_Uaa[:, so, so])
        SCR3 -= 4 * lib.einsum("Pja, APij -> Aai", G_Uov, pd_cderi_Uoo)
        del pd_cderi_Uoo

        # vir-vir part
        pd_cderi_Uvv = lib.einsum("Ama, Pmb -> APab", U_1[:, :, sv], cderi_Uaa[:, :, sv])
        # pd_cderi_Uvv += pd_cderi_Uvv.swapaxes(-1, -2)
        util.hermi_sum_last2dim(pd_cderi_Uvv)
        SCR3 += 4 * lib.einsum("APib, Pab -> Aai", pd_G_Uov, cderi_Uaa[:, sv, sv])
        SCR3 += 4 * lib.einsum("Pib, APab -> Aai", G_Uov, pd_cderi_Uvv)
        del pd_cderi_Uvv

    log.timer("get_SCR3 of dipole", *time0)
    return SCR3


class RMP2DipoleRI(DipoleBase, RMP2RespRI):

    def make_pd_cderi_uov(self):
        if self.pad_prop("cderi_uov") in self.tensors:
            return self.tensors[self.pad_prop("cderi_uov")]

        cderi_uaa = self.make_cderi_uaa()
        U_1 = self.make_U_1()
        mo_occ = self.mo_occ
        verbose = self.verbose
        max_memory = self.max_memory

        pd_cderi_uov = self.get_pd_cderi_uov(
            cderi_uaa=cderi_uaa,
            U_1=U_1,
            mo_occ=mo_occ,
            verbose=verbose,
            max_memory=max_memory)

        self.tensors[self.pad_prop("cderi_uov")] = pd_cderi_uov
        return pd_cderi_uov

    def _make_mp2_integrals_deriv(self):
        cderi_uov = self.make_cderi_uov()
        pd_cderi_uov = self.make_pd_cderi_uov()
        t_oovv = self.make_t_oovv()
        pd_fock_mo = self.scf_prop.make_pd_fock_mo()
        mo_occ = self.mo_occ
        mo_energy = self.mo_energy
        c_os = self.c_os
        c_ss = self.c_ss
        verbose = self.verbose
        max_memory = self.max_memory

        tensors = self.get_mp2_integrals_deriv(
            cderi_uov=cderi_uov,
            pd_cderi_uov=pd_cderi_uov,
            t_oovv=t_oovv,
            pd_fock_mo=pd_fock_mo,
            mo_occ=mo_occ,
            mo_energy=mo_energy,
            c_os=c_os, c_ss=c_ss,
            verbose=verbose,
            max_memory=max_memory)

        for key, val in tensors.items():
            self.tensors[self.pad_prop(key)] = val

    def make_pd_G_uov(self):
        if self.pad_prop("pd_G_uov") in self.tensors:
            return self.tensors[self.pad_prop("pd_G_uov")]

        self._make_mp2_integrals_deriv()
        return self.tensors[self.pad_prop("pd_G_uov")]

    def make_pd_rdm1_corr(self):
        if self.pad_prop("pd_rdm1_corr") in self.tensors:
            return self.tensors[self.pad_prop("pd_rdm1_corr")]

        self._make_mp2_integrals_deriv()
        return self.tensors[self.pad_prop("pd_rdm1_corr")]

    def make_SCR3(self):
        if self.pad_prop("SCR3") in self.tensors:
            return self.tensors[self.pad_prop("SCR3")]

        cderi_uaa = self.make_cderi_uaa()
        G_uov = self.make_G_uov()
        pd_G_uov = self.make_pd_G_uov()
        U_1 = self.make_U_1()
        mo_occ = self.mo_occ

        SCR3 = self.get_SCR3(
            cderi_uaa, G_uov, pd_G_uov, U_1,
            mo_occ,
            verbose=lib.logger.NOTE,
            max_memory=2000)

        self.tensors[self.pad_prop("SCR3")] = SCR3
        return SCR3

    get_pd_cderi_uov = staticmethod(get_pd_cderi_uov)
    get_mp2_integrals_deriv = staticmethod(get_mp2_integrals_deriv)
    get_SCR3 = staticmethod(get_SCR3)


class RMP2PolarRI(RMP2RespRI, PolarBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.deriv_dipole = RMP2DipoleRI.from_cls(self, self.scf, copy_all=True)


if __name__ == '__main__':
    def main_1():
        from pyscf import gto, scf
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", verbose=0).build()
        mf = scf.RHF(mol).density_fit("cc-pVDZ-jkfit").run()
        mf_pol = RMP2PolarRI(mf)
        print(mf_pol.de)

        def dipole_with_dipole_field(t, h):
            mf_scf = scf.RHF(mol).density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = RMP2RespRI(mf_scf)
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 1e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)

    main_1()

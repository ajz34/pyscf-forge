""" Hybrid-DFT Response-Related Utilities. """

from pyscf.dh import RHDFT
from pyscf.dh import util
from pyscf import gto, dft, lib, __config__, scf
from pyscf.dh.response import RespBase
from pyscf.scf import _response_functions  # this import is not unnecessary
from pyscf.dh.energy.hdft.rhdft import get_rho
import numpy as np
from functools import cached_property


CONFIG_dh_verbose = getattr(__config__, "dh_verbose", lib.logger.NOTE)
CONFIG_incore_cderi_uaa_hdft = getattr(__config__, "incore_cderi_uaa_hdft", "auto")
CONFIG_incore_eri_cpks_vovo = getattr(__config__, "incore_eri_cpks_vovo", "auto")
CONFIG_dft_gen_grid_Grids_level = getattr(__config__, 'dft_gen_grid_Grids_level', 3)
CONFIG_use_eri_cpks = getattr(__config__, "use_eri_cpks", True)


def get_eri_cpks_vovo(
        cderi_uaa, mo_occ, cj, ck, eri_cpks_vovo=None,
        max_memory=2000, verbose=CONFIG_dh_verbose):
    r""" Obtain ERI term :math:`A_{ai, bj}^\mathrm{HF}`.

    For given

    Parameters
    ----------
    cderi_uaa : np.ndarray
    mo_occ : np.ndarray
    cj : float
    ck : float
    eri_cpks_vovo : np.ndarray or None
    max_memory : float
    verbose : int

    Returns
    -------
    np.ndarray
    """
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    # dimension and sanity check
    naux = cderi_uaa.shape[0]
    nmo = len(mo_occ)
    nocc = (mo_occ != 0).sum()
    nvir = (mo_occ == 0).sum()
    so, sv = slice(0, nocc), slice(nocc, nmo)
    assert cderi_uaa.shape == (naux, nmo, nmo)

    # allocate eri_cpks_vovo if necessary
    if eri_cpks_vovo is None:
        eri_cpks_vovo = np.zeros((nvir, nocc, nvir, nocc))
    else:
        assert eri_cpks_vovo.shape == (nvir, nocc, nvir, nocc)

    # copy some tensors to memory
    cderi_uvo = np.asarray(cderi_uaa[:, sv, so])
    cderi_uoo = np.asarray(cderi_uaa[:, so, so])

    # prepare for async and batch
    mem_avail = max_memory - lib.current_memory()[0]
    nbatch = util.calc_batch_size(nvir*naux + 2*nocc**2*nvir, mem_avail)

    def load_cderi_uaa(slc):
        return np.asarray(cderi_uaa[:, slc, sv])

    def write_eri_cpks_vovo(slc, blk):
        eri_cpks_vovo[slc] += blk

    batches = util.gen_batch(nocc, nmo, nbatch)
    with lib.call_in_background(write_eri_cpks_vovo) as async_write_eri_cpks_vovo:
        for sA, cderi_uAa in zip(batches, lib.map_with_prefetch(load_cderi_uaa, batches)):
            log.debug(f"[DEBUG] in get_eri_cpks_vovo, slice {sA} of virtual orbitals {nocc, nmo}")
            sAvir = slice(sA.start - nocc, sA.stop - nocc)
            blk = np.zeros((sA.stop - sA.start, nocc, nvir, nocc))
            if abs(cj) > 1e-10:
                blk += 4 * cj * lib.einsum("Pai, Pbj -> aibj", cderi_uvo[:, sAvir], cderi_uvo)
            if abs(ck) > 1e-10:
                blk -= ck * (
                    + lib.einsum("Paj, Pbi -> aibj", cderi_uvo[:, sAvir], cderi_uvo)
                    + lib.einsum("Pij, Pab -> aibj", cderi_uoo, cderi_uAa))
            async_write_eri_cpks_vovo(sAvir, blk)

    log.timer("get_eri_cpks_vovo", *time0)
    return eri_cpks_vovo


def get_Ax0_cpks_HF(eri_cpks_vovo, max_memory=2000, verbose=CONFIG_dh_verbose):
    r""" Convenient function for evaluation of HF contribution of Fock response in MO basis
    :math:`\sum_{rs} A_{ai, bj} X_{bj}^\mathbb{A}` by explicitly contraction to MO ERI :math:`(ai|bj)`.
    
    Parameters
    ----------
    eri_cpks_vovo : np.ndarray
    max_memory : float
    verbose : int

    Returns
    -------
    callable
    """
    nvir, nocc = eri_cpks_vovo.shape[:2]

    def Ax0_cpks_HF_inner(X):
        log = lib.logger.new_logger(verbose=verbose)
        time0 = lib.logger.process_clock(), lib.logger.perf_counter()
        
        X_shape = X.shape
        X = X.reshape((-1, X_shape[-2], X_shape[-1]))
        res = np.zeros_like(X)

        def load_eri_cpks_vovo(slc):
            return eri_cpks_vovo[slc]

        mem_avail = max_memory - lib.current_memory()[0]
        nbatch = util.calc_batch_size(nocc**2 * nvir, mem_avail)
        batches = util.gen_batch(0, nvir, nbatch)
        
        for sA, eri_cpks_Vovo in zip(batches, lib.map_with_prefetch(load_eri_cpks_vovo, batches)):
            res[:, sA] = lib.einsum("aibj, Abj -> Aai", eri_cpks_Vovo, X)
        
        res.shape = list(X_shape[:-2]) + [res.shape[-2], res.shape[-1]]

        log.timer("Ax0_cpks_HF_inner", *time0)
        return res
    
    return Ax0_cpks_HF_inner


def get_Ax0_Core_KS(
        sp, sq, sr, ss,
        mo_coeff, xc_setting, xc_kernel,
        max_memory=2000, verbose=CONFIG_dh_verbose):
    r""" Convenient function for evaluation of pure DFT contribution of Fock response in MO basis
    :math:`\sum_{rs} A_{pq, rs} X_{rs}^\mathbb{A}`.
    
    Parameters
    ----------
    sp, sq, sr, ss : slice or list
    mo_coeff : np.ndarray
    xc_setting : tuple[dft.numint.NumInt, gto.Mole, dft.Grids, str, np.ndarray]
    xc_kernel : tuple[np.ndarray, np.ndarray, np.ndarray]
    max_memory : float
    verbose : int

    Returns
    -------
    callable
    """
    C = mo_coeff
    ni, mol, grids, xc, dm = xc_setting
    rho, _, fxc = xc_kernel
    
    def Ax0_Core_KS_inner(X):
        log = lib.logger.new_logger(verbose=verbose)
        time0 = lib.logger.process_clock(), lib.logger.perf_counter()
        mem_avail = max_memory - lib.current_memory()[0]

        X_shape = X.shape
        X = X.reshape((-1, X_shape[-2], X_shape[-1]))
        dmX = C[:, sr] @ X @ C[:, ss].T
        dmX += dmX.swapaxes(-1, -2)
        ax_ao = ni.nr_rks_fxc(mol, grids, xc, dm, dmX, hermi=1, rho0=rho, fxc=fxc, max_memory=mem_avail)
        res = 2 * C[:, sp].T @ ax_ao @ C[:, sq]
        res.shape = list(X_shape[:-2]) + [res.shape[-2], res.shape[-1]]

        log.timer("Ax0_Core_KS_inner", *time0)
        return res

    return Ax0_Core_KS_inner


def get_Ax0_Core_resp(
        sp, sq, sr, ss, vresp, mo_coeff,
        verbose=CONFIG_dh_verbose):
    r""" Convenient function for evaluation of Fock response in MO basis
    :math:`\sum_{rs} A_{pq, rs} X_{rs}^\mathbb{A}` by PySCF's response function.

    Parameters
    ----------
    sp, sq, sr, ss : slice or list
        Slice of molecular orbital indices.
    vresp : callable
        Fock response function in AO basis (generated by ``mf.scf.gen_response``).
    mo_coeff : np.ndarray
        Molecular orbital coefficients.
    verbose : int
        Print verbosity.

    Returns
    -------
    callable
        A function where input is :math:`X_{rs}^\mathbb{A}`, and output is
        :math:`\sum_{rs} A_{pq, rs} X_{rs}^\mathbb{A}`.
    """
    C = mo_coeff

    def Ax0_Core_resp_inner(X):
        log = lib.logger.new_logger(verbose=verbose)
        time0 = lib.logger.process_clock(), lib.logger.perf_counter()

        X_shape = X.shape
        X = X.reshape((-1, X_shape[-2], X_shape[-1]))
        dmX = C[:, sr] @ X @ C[:, ss].T
        dmX += dmX.swapaxes(-1, -2)
        ax_ao = vresp(dmX)
        res = 2 * C[:, sp].T @ ax_ao @ C[:, sq]
        res.shape = list(X_shape[:-2]) + [res.shape[-2], res.shape[-1]]

        log.timer("Ax0_Core_resp_inner", *time0)
        return res

    return Ax0_Core_resp_inner


def get_xc_integral(ni, mol, grids, xc, dm):
    rho = get_rho(mol, grids, dm)
    try:
        exc, vxc, fxc, kxc = ni.eval_xc_eff(xc, rho, deriv=3)
        tensors = {
            "rho": rho,
            f"exc_{xc}": exc,
            f"vxc_{xc}": vxc,
            f"fxc_{xc}": fxc,
            f"kxc_{xc}": kxc,
        }
    except NotImplementedError:
        exc, vxc, fxc, _ = ni.eval_xc_eff(xc, rho, deriv=2)
        tensors = {
            "rho": rho,
            f"exc_{xc}": exc,
            f"vxc_{xc}": vxc,
            f"fxc_{xc}": fxc,
        }
    return tensors


class RHDFTResp(RHDFT, RespBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.incore_cderi_uaa = CONFIG_incore_cderi_uaa_hdft
        self.incore_eri_cpks_vovo = CONFIG_incore_eri_cpks_vovo
        self.use_eri_cpks = CONFIG_use_eri_cpks

        grid_level_cpks = max(CONFIG_dft_gen_grid_Grids_level - 1, 1)
        self.grids_cpks = dft.Grids(self.mol)
        self.grids_cpks.level = grid_level_cpks
        self.grids_cpks.build()

    @cached_property
    def vresp(self):
        """ Fock response function (derivative w.r.t. molecular coefficient in AO basis). """
        try:
            return self.scf.gen_response()
        except ValueError:
            # case that have customized xc
            # should pass check of ni.libxc.test_deriv_order and ni.libxc.is_hybrid_xc
            ni = self.scf._numint
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.scf.xc, self.mol.spin)
            if abs(hyb) > 1e-10 or abs(omega) > 1e-10:
                fake_xc = "B3LYPg"
            else:
                fake_xc = "PBE"
            actual_xc = self.scf.xc
            self.scf.xc = fake_xc
            resp = self.scf.gen_response()
            self.scf.xc = actual_xc
            return resp

    def make_cderi_uaa(self, omega=0):
        """ Generate cholesky decomposed ERI (all block, full orbitals, s1 symm, in memory/disk). """
        if util.pad_omega("cderi_uaa", omega) in self.tensors:
            return self.tensors[util.pad_omega("cderi_uaa", omega)]

        log = lib.logger.new_logger(verbose=self.verbose)

        # dimension and mask
        mo_coeff = self.mo_coeff
        nmo = self.nmo

        # density fitting preparation
        with_df = util.get_with_df_omega(self.scf.with_df, omega)
        naux = with_df.get_naoaux()

        # array allocation
        mem_avail = self.max_memory - lib.current_memory()[0]
        incore_cderi_uaa = self.incore_cderi_uaa
        cderi_uaa = util.allocate_array(
            incore_cderi_uaa, (naux, nmo, nmo), mem_avail,
            h5file=self._tmpfile, name=util.pad_omega("cderi_uaa", omega), zero_init=False)
        log.info(f"[INFO] Store type of cderi_uaa: {type(cderi_uaa)}")

        # generate array
        util.get_cderi_mo(with_df, mo_coeff, cderi_uaa, max_memory=self.max_memory)

        tensors = {util.pad_omega("cderi_uaa", omega): cderi_uaa}
        self.tensors.update(tensors)
        return cderi_uaa

    def make_Ax0_Core(self, sp, sq, sr, ss):
        r""" Convenient function for evaluation of Fock response in MO basis
        :math:`\sum_{rs} A_{pq, rs} X_{rs}^\mathbb{A}`.

        Parameters
        ----------
        sp, sq, sr, ss : slice or list
            Slice of molecular orbital indices.

        Returns
        -------
        callable
            A function where input is :math:`X_{rs}^\mathbb{A}`, and output is
            :math:`\sum_{rs} A_{pq, rs} X_{rs}^\mathbb{A}`.

        Notes
        -----
        This function acts as a wrapper of various possible Fock response algorithms.

        The exch-corr functional of this function refers to `self.hdft` (energy evaluation) instead of `self.scf` (SCF).
        """
        # if not RI, then use general Ax0_Core_resp
        if not hasattr(self.scf, "with_df") or not self.use_eri_cpks:
            return self.make_Ax0_Core_resp(sp, sq, sr, ss)

        # try if satisfies CPKS evaluation (vovo)
        lst_nmo = np.arange(self.nmo)
        lst_occ = lst_nmo[:self.nocc]
        lst_vir = lst_nmo[self.nocc:]
        try:
            if (
                    np.all(lst_nmo[sp] == lst_vir) and np.all(lst_nmo[sq] == lst_occ) and
                    np.all(lst_nmo[sr] == lst_vir) and np.all(lst_nmo[ss] == lst_occ)):
                return self.make_Ax0_cpks()
            else:
                # otherwise, use response by PySCF
                return self.make_Ax0_Core_resp(sp, sq, sr, ss)
        except ValueError:
            # dimension not match
            return self.make_Ax0_Core_resp(sp, sq, sr, ss)

    def make_Ax0_Core_resp(self, sp, sq, sr, ss, vresp=None, mo_coeff=None):
        r""" Convenient function for evaluation of Fock response in MO basis
        :math:`\sum_{rs} A_{pq, rs} X_{rs}^\mathbb{A}` by PySCF's response function.

        Parameters
        ----------
        sp, sq, sr, ss : slice or list
            Slice of molecular orbital indices.
        vresp : callable
            Fock response function in AO basis (generated by ``mf.scf.gen_response``).
        mo_coeff : np.ndarray
            Molecular orbital coefficients.

        Returns
        -------
        callable
            A function where input is :math:`X_{rs}^\mathbb{A}`, and output is
            :math:`\sum_{rs} A_{pq, rs} X_{rs}^\mathbb{A}`.

        Notes
        -----
        This function calls PySCF's gen_response function. For cases that requires large virtual contraction
        (such as :math:`A_{ai, pq} X_{pq}`), this function should be somehow quicker.
        """
        vresp = vresp if vresp is not None else self.vresp
        mo_coeff = mo_coeff if mo_coeff is not None else self.mo_coeff
        return self.get_Ax0_Core_resp(sp, sq, sr, ss, vresp, mo_coeff)

    def make_eri_cpks_vovo(self):
        r""" Generate ERI for CP-KS evaluation :math:`(ai, bj)` for current exch-corr setting. """
        if "eri_cpks_vovo" in self.tensors:
            return self.tensors["eri_cpks_vovo"]

        # This function will utilize mf.scf._numint to specify hybrid and omega coefficients

        if not hasattr(self.scf, "xc"):
            # not a DFT instance, configure as HF instance
            omega, alpha, hyb = 0, 0, 1
        else:
            ni = self.scf._numint  # type: dft.numint.NumInt
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.scf.xc)

        nvir, nocc = self.nvir, self.nocc
        cderi_uaa = self.tensors.get("cderi_uaa", self.make_cderi_uaa())

        eri_cpks_vovo = util.allocate_array(
            incore=self.incore_eri_cpks_vovo,
            shape=(nvir, nocc, nvir, nocc),
            max_memory=self.max_memory,
            h5file=self._tmpfile,
            name="eri_cpks_vovo",
            chunk=(1, nocc, nvir, nocc))

        self.get_eri_cpks_vovo(
            cderi_uaa=cderi_uaa,
            mo_occ=self.mo_occ,
            cj=1, ck=hyb,
            eri_cpks_vovo=eri_cpks_vovo,
            max_memory=self.max_memory,
            verbose=self.verbose
        )

        if abs(omega) > 1e-10:
            cderi_uaa = self.make_cderi_uaa(omega=omega)
            self.get_eri_cpks_vovo(
                cderi_uaa=cderi_uaa,
                mo_occ=self.mo_occ,
                cj=0, ck=alpha-hyb,
                eri_cpks_vovo=eri_cpks_vovo,
                max_memory=self.max_memory,
                verbose=self.verbose
            )

        self.tensors["eri_cpks_vovo"] = eri_cpks_vovo
        return eri_cpks_vovo

    def make_Ax0_cpks_HF(self):
        eri_cpks_vovo = self.tensors.get("eri_cpks_vovo", self.make_eri_cpks_vovo())
        ax0_cpks_hf = self.get_Ax0_cpks_HF(eri_cpks_vovo, self.max_memory, self.verbose)
        return ax0_cpks_hf

    def make_xc_integral(self):
        ni = self.scf._numint  # type: dft.numint.NumInt
        mol = self.mol
        grids = self.scf.grids
        xc_token = self.xc.token
        dm = self.scf.make_rdm1()
        tensors = self.get_xc_integral(ni, mol, grids, xc_token, dm)
        self.tensors.update(tensors)
        return tensors

    def make_xc_integral_cpks(self):
        ni = self.scf._numint  # type: dft.numint.NumInt
        mol = self.mol
        grids = self.grids_cpks
        xc_token = self.xc.token
        dm = self.scf.make_rdm1()
        tensors = self.get_xc_integral(ni, mol, grids, xc_token, dm)
        tensors = {key + "/cpks": val for key, val in tensors.items()}
        self.tensors.update(tensors)
        return tensors

    def make_Ax0_Core_KS(self, sp, sq, sr, ss):
        if not hasattr(self.scf, "xc"):
            # not a DFT instance, KS contribution is zero
            return lambda *args, **kwargs: 0

        ni = self.scf._numint
        mol = self.mol
        grids = self.grids_cpks
        xc_token = self.xc.token
        dm = self.scf.make_rdm1()
        if f"fxc_{xc_token}" not in self.tensors:
            self.make_xc_integral_cpks()
        rho = self.tensors[f"rho/cpks"]
        vxc = self.tensors[f"vxc_{xc_token}/cpks"]
        fxc = self.tensors[f"fxc_{xc_token}/cpks"]

        xc_setting = ni, mol, grids, xc_token, dm
        xc_kernel = rho, vxc, fxc
        mo_coeff = self.mo_coeff

        ax0_core_ks = self.get_Ax0_Core_KS(
            sp, sq, sr, ss,
            mo_coeff, xc_setting, xc_kernel,
            max_memory=self.max_memory,
            verbose=self.verbose)
        return ax0_core_ks

    def make_Ax0_cpks(self):
        so, sv = self.mask_occ, self.mask_vir
        ax0_core_ks = self.make_Ax0_Core_KS(sv, so, sv, so)
        ax0_cpks_hf = self.make_Ax0_cpks_HF()

        def Ax0_cpks_inner(X):
            res = ax0_cpks_hf(X) + ax0_core_ks(X)
            return res
        return Ax0_cpks_inner

    def make_lag_vo(self):
        r""" Generate hybrid DFT contribution to Lagrangian vir-occ block :math:`L_{ai}`. """
        if "lag_vo" in self.tensors:
            return self.tensors["lag_vo"]

        nocc, nmo = self.nocc, self.nmo
        so, sv = slice(0, nocc), slice(nocc, nmo)
        mo_coeff = self.mo_coeff
        lag_vo = 4 * mo_coeff[:, sv].T @ self.hdft.get_fock(dm=self.scf.make_rdm1()) @ mo_coeff[:, so]
        self.tensors["lag_vo"] = lag_vo
        return lag_vo

    def make_rdm1_resp_vo(self):
        r""" Generate 1-RDM (response) of hybrid DFT contribution :math:`D_{ai}`. """
        if "rdm1_resp_vo" in self.tensors:
            return self.tensors["rdm1_resp_vo"]

        # prepare input
        lag_vo = self.tensors.get("lag_vo", self.make_lag_vo())

        rdm1_resp_vo = self.solve_cpks(lag_vo)
        self.tensors["rdm1_resp_vo"] = rdm1_resp_vo
        return rdm1_resp_vo

    def make_rdm1_resp(self, ao=False):
        r""" Generate 1-RDM (response) of hybrid DFT :math:`D_{pq}` in MO or :math:`D_{\mu \nu}` in AO. """

        nocc, nmo = self.nocc, self.nmo
        so, sv = slice(0, nocc), slice(nocc, nmo)
        rdm1 = np.diag(self.mo_occ)
        rdm1[sv, so] = self.tensors.get("rdm1_resp_vo", self.make_rdm1_resp_vo())
        self.tensors["rdm1_resp"] = rdm1
        if ao:
            rdm1 = self.mo_coeff @ rdm1 @ self.mo_coeff.T
        return rdm1

    get_Ax0_Core_resp = staticmethod(get_Ax0_Core_resp)
    get_Ax0_cpks_HF = staticmethod(get_Ax0_cpks_HF)
    get_Ax0_Core_KS = staticmethod(get_Ax0_Core_KS)
    get_eri_cpks_vovo = staticmethod(get_eri_cpks_vovo)
    get_xc_integral = staticmethod(get_xc_integral)


if __name__ == '__main__':
    pass

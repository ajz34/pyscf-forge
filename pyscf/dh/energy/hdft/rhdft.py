from pyscf.dh import util
from pyscf.dh.util import XCList, XCInfo
from pyscf.dh.energy import EngBase
from pyscf import dft, lib, scf, df, __config__
import numpy as np
import copy

from pyscf.dh.util.numint_addon import numint_customized


def get_energy_restricted_exactx(mf, dm, omega=None):
    """ Evaluate exact exchange energy (for either HF and long-range).

    Parameters
    ----------
    mf : dft.rks.RKS
        SCF object that have ``get_k`` member function.
    dm : np.ndarray
        Density matrix.
    omega : float or None
        Parameter of long-range ERI integral :math:`\\mathrm{erfc} (\\omega r_{12}) / r_{12}`.
    """
    hermi = 1 if np.allclose(dm, dm.T.conj()) else 0
    if omega == 0:
        vk = mf.get_k(dm=dm, hermi=hermi)
    else:
        vk = mf.get_k(dm=dm, hermi=hermi, omega=omega)
    ex = - 0.25 * np.einsum('ij, ji ->', dm, vk)
    ex = util.check_real(ex)
    # results
    result = dict()
    omega = omega if omega is not None else 0
    result[util.pad_omega("eng_exx_HF", omega)] = ex
    return result


def get_energy_restricted_noxc(mf, dm):
    """ Evaluate energy contributions that is not exchange-correlation.

    Note that some contributions (such as vdw) is not considered.

    Parameters
    ----------
    mf : dft.rks.RKS
        SCF object that have ``get_hcore``, ``get_j`` member functions.
    dm : np.ndarray
        Density matrix.
    """
    hermi = 1 if np.allclose(dm, dm.T.conj()) else 0
    hcore = mf.get_hcore()
    vj = mf.get_j(dm=dm, hermi=hermi)
    eng_nuc = mf.mol.energy_nuc()
    eng_hcore = np.einsum('ij, ji ->', dm, hcore)
    eng_j = 0.5 * np.einsum('ij, ji ->', dm, vj)
    eng_hcore = util.check_real(eng_hcore)
    eng_j = util.check_real(eng_j)
    eng_noxc = eng_hcore + eng_nuc + eng_j
    # results
    results = dict()
    results["eng_nuc"] = eng_nuc
    results["eng_hcore"] = eng_hcore
    results["eng_j"] = eng_j
    results["eng_noxc"] = eng_noxc
    return results


def get_rho(mol, grids, dm):
    """ Obtain density on DFT grids.

    Note that this function always give density ready for meta-GGA.
    Thus, returned density grid dimension is (6, ngrid).
    For more details, see docstring in ``pyscf.dft.numint.eval_rho``.

    This function accepts either restricted or unrestricted density matrices.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    grids : dft.grid.Grids
        DFT grids object. Dimension (nao, nao) or (nset, nao, nao).
    dm : np.ndarray
        Density matrix.

    Returns
    -------
    np.ndarray
        Density grid of dimension (5, ngrid) or (nset, 5, ngrid).
    """
    ngrid = grids.weights.size
    ni = dft.numint.NumInt()
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, with_lapl=False)
    rho = np.empty((nset, 5, ngrid))
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, deriv=1, max_memory=2000):
        p0, p1 = p1, p1 + weight.size
        for idm in range(nset):
            rho[idm, :, p0:p1] = make_rho(idm, ao, mask, 'MGGA')
    if nset == 1:
        rho = rho[0]
    return rho


def get_energy_purexc(xc_lists, rho, weights, restricted, numint=None):
    """ Evaluate energy contributions of pure (DFT) exchange-correlation effects.

    Note that this kernel does not count HF, LR_HF and advanced correlation into account.
    To evaluate exact exchange (HF or LR_HF), use ``kernel_energy_restricted_exactx``.

    Parameters
    ----------
    xc_lists : str or XCInfo or XCList or list[str or XCInfo or XCList]
        List of xc codes.
    rho : np.ndarray
        Full list of density grids. Dimension (>1, ngrid) or (nset, >1, ngrid).
    weights : np.ndarray
        Full list of DFT grid weights.
    restricted : bool
        Indicator of restricted or unrestricted of incoming rho.
    numint : dft.numint.NumInt
        Special numint item if required.

    Returns
    -------
    dict
        Since realization of dictionary in python is ordered, use `result.values()` gives xc energy ordered by input
        xc functional tokens.

    See Also
    --------
    kernel_energy_restricted_exactx
    """
    if not isinstance(xc_lists, list):
        xc_lists = [xc_lists]
    # parse xc_lists
    xc_lists, xc_lists_ = [], xc_lists
    for xc_list in xc_lists_:
        if isinstance(xc_list, str):
            xc_lists.append(XCList(xc_list, code_scf=False))
        elif isinstance(xc_list, XCInfo):
            xc_lists.append((XCList.build_from_list([xc_list])))
        elif isinstance(xc_list, XCList):
            xc_lists.append(xc_list)
        else:
            assert False, "Type of input must be str, XCInfo or XCList."

    results = {}
    for xc_list in xc_lists:
        if numint is None:
            ni = dft.numint.NumInt()
            try:
                ni._xc_type(xc_list.token)
            except (ValueError, KeyError):
                ni = numint_customized(xc_list)
        else:
            ni = numint

        if restricted:
            wrho0 = rho[0] * weights
        else:
            wrho0 = rho[:, 0].sum(axis=0) * weights

        rho_to_eval = rho
        if ni._xc_type(xc_list.token) == "LDA":
            if restricted:
                rho_to_eval = rho[0]
            else:
                rho_to_eval = rho[:, 0]

        exc = ni.eval_xc_eff(xc_list.token, rho_to_eval, deriv=0)[0]
        results[f"eng_purexc_{xc_list.token}"] = exc @ wrho0
    return results


def get_energy_vv10(mol, dm, nlc_pars, grids=None, nlcgrids=None, verbose=lib.logger.NOTE):
    log = lib.logger.new_logger(verbose=verbose)
    if grids is None:
        log.warn("VV10 grids not found. Use default grids of PySCF for VV10.")
        grids = dft.Grids(mol).build()
    rho = get_rho(mol, grids, dm)
    if nlcgrids is None:
        nlcgrids = grids
        vvrho = rho
    else:
        nlcgrids.build()
        vvrho = get_rho(mol, nlcgrids, dm)
    # handle unrestricted case
    if len(rho.shape) == 3:
        rho = rho[0] + rho[1]
        vvrho = vvrho[0] + vvrho[1]
    exc_vv10, _ = dft.numint._vv10nlc(rho, grids.coords, vvrho, nlcgrids.weights, nlcgrids.coords, nlc_pars)
    eng_vv10 = (rho[0] * grids.weights * exc_vv10).sum()
    result = dict()
    result["eng_VV10({:}; {:})".format(*nlc_pars)] = eng_vv10
    return result


def custom_mf(mf, xc, auxbasis_or_with_df=None):
    """ Customize options of PySCF's mf object.

    - Check and customize numint if necessary
    - Check and customize density fitting object if necessary

    Parameters
    ----------
    mf : dft.rks.RKS or dft.uks.UKS
        SCF object to be customized.
    xc : XCList
        Exchange-correlation list.
    auxbasis_or_with_df : str or df.DF
        Auxiliary basis definition.

    Returns
    -------
    dft.rks.RKS or dft.uks.UKS

    Notes
    -----
    Note that if no with_df object passed in, the density-fitting setting of an SCF object is left as is.
    So leaving option ``auxbasis_or_with_df=None`` will not convert density-fitting SCF to conventional SCF.
    """
    mf = copy.copy(mf)
    verbose = mf.verbose
    log = lib.logger.new_logger(verbose=verbose)
    restricted = isinstance(mf, scf.hf.RHF)

    # transform to dft class if necessary
    if not hasattr(mf, "xc"):
        log.note("[INFO] Input SCF instance is not KS. Transfer to KS instance.")
        converged = mf.converged
        if restricted:
            mf = mf.to_rks()
        else:
            mf = mf.to_uks()
        mf.converged = converged

    # check whether xc code is the same to SCF object; if not, substitute it
    if XCList(mf.xc, code_scf=False).token != xc.token:
        log.note("[INFO] Exchange-correlation is not the same to SCF object. Change xc of SCF.")
        mf.xc = xc.token
        mf.converged = False
        mf._numint = dft.numint.NumInt()  # refresh numint

    # try PySCF parsing of xc code; if not PySCF parsable, customize numint first
    try:
        ni = mf._numint  # type: dft.numint.NumInt
        ni._xc_type(mf.xc)
    except (KeyError, ValueError):
        mf._numint = numint_customized(xc)
        mf.converged = False

    # change to with_df object
    if auxbasis_or_with_df is not None:
        if not isinstance(auxbasis_or_with_df, df.DF):
            with_df = df.DF(mf.mol, auxbasis=auxbasis_or_with_df)
        else:
            with_df = auxbasis_or_with_df
        if not hasattr(mf, "with_df"):
            mf = mf.density_fit(with_df=with_df)
            mf.converged = False
        else:
            if mf.with_df.auxbasis != with_df.auxbasis:
                mf.with_df = with_df
                mf.converged = False

    return mf


class RSCF(EngBase):
    """ Restricted SCF hybrid (low-rung) DFT wrapper class of convenience.

    Notes
    -----
    This class is an extension to original PySCF's RKS/UKS.

    - Only for further usage of response properties.

    Warnings
    --------
    This class may change the underlying SCF object.
    It's better to initialize this object first, before actually running SCF iterations.
    """

    def __init__(self, mf):
        super().__init__(mf)
        if not hasattr(mf, "xc"):
            self.xc = "HF"
        else:
            self.xc = mf.xc
        xc_scf = XCList(self.xc, code_scf=True)
        xc_eng = XCList(self.xc, code_scf=False)
        if xc_scf != xc_eng:
            raise ValueError("Given energy functional contains part that could not handle with SCF!")
        self.xc = xc_scf

    @property
    def restricted(self):
        return True

    @property
    def e_tot(self) -> float:
        return self.scf.e_tot

    def make_energy_purexc(self, xc_lists, numint=None, dm=None):
        """ Evaluate energy contributions of pure (DFT) exchange-correlation effects.

        Parameters
        ----------
        xc_lists : str or XCInfo or XCList or list[str or XCInfo or XCList]
            List of xc codes.
        numint : dft.numint.NumInt
            Special numint item if required.
        dm : np.ndarray
            Density matrix in AO basis.

        See Also
        --------
        get_energy_purexc
        """
        grids = self.scf.grids
        if dm is None:
            dm = self.scf.make_rdm1()
        dm = np.asarray(dm)
        if (self.restricted and dm.ndim != 2) or (not self.restricted and (dm.ndim != 3 or dm.shape[0] != 2)):
            raise ValueError("Dimension of input density matrix is not correct.")
        rho = self.get_rho(self.mol, grids, dm)
        return self.get_energy_purexc(
            xc_lists, rho, grids.weights, self.restricted, numint=numint)

    def kernel(self, *args, **kwargs):
        if not self.scf.converged:
            self.scf.kernel(*args, **kwargs)
        return self.e_tot

    def to_resp(self, key):
        from pyscf.dh.response.hdft.rhdft import RSCFResp
        from pyscf.dh.dipole.hdft.rhdft import RSCFDipole
        resp_dict = {
            "resp": RSCFResp,
            "dipole": RSCFDipole,
        }
        return resp_dict[key].from_cls(self, self.scf, copy_all=True)

    get_energy_exactx = staticmethod(get_energy_restricted_exactx)
    get_energy_noxc = staticmethod(get_energy_restricted_noxc)
    get_energy_vv10 = staticmethod(get_energy_vv10)
    get_energy_purexc = staticmethod(get_energy_purexc)
    get_rho = staticmethod(get_rho)


class RHDFT(RSCF):
    """ Restricted hybrid (low-rung) DFT wrapper class of convenience.

    Notes
    -----
    This class is an extension to original PySCF's RKS/UKS.

    - Evaluation of various low-rung energy component results
    - Modification to NumInt class

    Warnings
    --------
    This class may change the underlying SCF object.
    It's better to initialize this object first, before actually running SCF iterations.
    """

    def __init__(self, mf, xc=None):
        super().__init__(mf)
        if xc is not None:
            self.xc = xc
        elif not hasattr(mf, "xc"):
            self.xc = "HF"
        else:
            self.xc = self.scf.xc
        if isinstance(self.xc, str):
            xc_scf = XCList(self.xc, code_scf=True)
            xc_eng = XCList(self.xc, code_scf=False)
            if xc_scf != xc_eng:
                raise ValueError("Given energy functional contains part that could not handle with SCF!")
            self.xc = xc_scf
        else:
            xc_scf = self.xc

        self.hdft = custom_mf(mf, xc_scf)

    @property
    def e_tot(self) -> float:
        key = f"eng_tot_{self.xc.token}"
        if key in self.results:
            return self.results[key]

        self.results[key] = self.hdft.energy_tot()
        return self.results[key]

    def kernel(self, *args, **kwargs):
        if not self.scf.converged:
            self.scf.kernel(*args, **kwargs)
        if self.hdft.mo_coeff is None:
            self.hdft.mo_coeff = self.scf.mo_coeff
            self.hdft.mo_occ = self.scf.mo_occ
            self.hdft.mo_energy = self.scf.mo_energy
        return self.e_tot

    def to_resp(self, key):
        from pyscf.dh.response.hdft.rhdft import RHDFTResp
        from pyscf.dh.dipole.hdft.rhdft import RHDFTDipole
        resp_dict = {
            "resp": RHDFTResp,
            "dipole": RHDFTDipole,
        }
        return resp_dict[key].from_cls(self, self.scf, copy_all=True)


if __name__ == '__main__':
    def main_1():
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5").build()
        mf_s = scf.RHF(mol)
        mf = RHDFT(mf_s, xc="HF, LYP").run()
        res = mf.make_energy_purexc([", LYP", "B88, ", "HF", "LR_HF(0.5)", "SSR(GGA_X_B88, 0.5), "])
        print(res)

    main_1()

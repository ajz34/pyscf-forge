from pyscf.dh.dipole.dipolebase import DipoleBase, PolarBase
from pyscf.dh.response.hdft.rhdft import RSCFResp, RHDFTResp
from pyscf.dh.dipole.hdft.rhdft import RSCFDipole, RHDFTDipole, RHDFTPolar
from pyscf import lib, dft, gto
import numpy as np

from pyscf.dh.response.hdft.uhdft import USCFResp, UHDFTResp

α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


def get_pd_fock_mo(
        fock_mo, hcore_1_mo, mo_occ,
        U_1, Ax0_Core,
        verbose=lib.logger.NOTE):
    """ Full derivative of Fock matrix in MO-basis by perturbation of dipole electric field.

    Note that this function is only used for electric properties, since derivative of ERI contribution is not counted.

    Parameters
    ----------
    fock_mo : np.ndarray
    hcore_1_mo : np.ndarray
    mo_occ : np.ndarray
    U_1 : np.ndarray
    Ax0_Core : callable
    verbose : int

    Returns
    -------
    np.ndarray
    """
    log = lib.logger.new_logger(verbose=verbose)
    time0 = lib.logger.process_clock(), lib.logger.perf_counter()

    mask_occ = [mo_occ[σ] != 0 for σ in (α, β)]
    sa, so = np.ones_like(mo_occ, dtype=bool), mask_occ
    U_1_occ = [U_1[σ][:, :, so[σ]] for σ in (α, β)]

    pd_fock_mo = (
        + hcore_1_mo
        + lib.einsum("sAmp, smq -> sApq", U_1, fock_mo)
        + lib.einsum("sAmq, spm -> sApq", U_1, fock_mo)
        + np.array(Ax0_Core(sa, sa, sa, so)(U_1_occ)))

    log.timer("get_pd_fock_mo of dipole", *time0)
    return pd_fock_mo


class USCFDipole(USCFResp, RSCFDipole):

    def make_hcore_1_mo(self):
        if self.pad_prop("hcore_1_mo") in self.tensors:
            return self.tensors[self.pad_prop("hcore_1_mo")]

        hcore_1_ao = self.make_hcore_1_ao()
        hcore_1_mo = lib.einsum("sup, tuv, svq -> stpq", self.mo_coeff, hcore_1_ao, self.mo_coeff)

        self.tensors[self.pad_prop("hcore_1_mo")] = hcore_1_mo
        return hcore_1_mo

    def make_pd_fock_mo(self):
        if self.pad_prop("pd_fock_mo") in self.tensors:
            return self.tensors[self.pad_prop("pd_fock_mo")]

        mo_coeff = self.mo_coeff
        fock_mo = lib.einsum("sup, suv, svq -> spq", mo_coeff, self.scf.get_fock(), mo_coeff)
        hcore_1_mo = self.make_hcore_1_mo()
        mo_occ = self.mo_occ
        U_1 = self.make_U_1()
        Ax0_Core = self.Ax0_Core
        verbose = self.verbose
        pd_fock_mo = self.get_pd_fock_mo(
            fock_mo=fock_mo,
            hcore_1_mo=hcore_1_mo,
            mo_occ=mo_occ,
            U_1=U_1,
            Ax0_Core=Ax0_Core,
            verbose=verbose)

        self.tensors[self.pad_prop("pd_fock_mo")] = pd_fock_mo
        return pd_fock_mo

    def make_U_1(self):
        """ Generate first-order derivative of molecular coefficient. """
        if self.pad_prop("U_1") in self.tensors:
            return self.tensors[self.pad_prop("U_1")]

        nocc, nmo = self.nocc, self.nmo
        so = [slice(0, nocc[σ]) for σ in (α, β)]
        sv = [slice(nocc[σ], nmo) for σ in (α, β)]

        hcore_1_mo = self.make_hcore_1_mo()
        hcore_1_vo = [hcore_1_mo[σ][:, sv[σ], so[σ]] for σ in (α, β)]
        U_1_vo = self.solve_cpks(hcore_1_vo)

        U_1 = np.zeros_like(hcore_1_mo)

        for σ in α, β:
            U_1[σ][:, sv[σ], so[σ]] = U_1_vo[σ]
            U_1[σ][:, so[σ], sv[σ]] = - U_1_vo[σ].swapaxes(-1, -2)

        self.tensors[self.pad_prop("U_1")] = U_1
        return U_1

    def make_SCR3(self):
        return 0

    def make_pd_rdm1_corr(self):
        return 0

    get_pd_fock_mo = staticmethod(get_pd_fock_mo)


class UHDFTDipole(DipoleBase, UHDFTResp):

    def make_pd_fock_mo(self):
        if self.pad_prop("pd_fock_mo") in self.tensors:
            return self.tensors[self.pad_prop("pd_fock_mo")]

        mo_coeff = self.mo_coeff
        fock_mo = lib.einsum("sup, suv, svq -> spq", mo_coeff, self.hdft.get_fock(), mo_coeff)
        hcore_1_mo = self.make_hcore_1_mo()
        mo_occ = self.mo_occ
        U_1 = self.make_U_1()
        Ax0_Core = self.make_Ax0_Core
        verbose = self.verbose
        pd_fock_mo = self.get_pd_fock_mo(
            fock_mo=fock_mo,
            hcore_1_mo=hcore_1_mo,
            mo_occ=mo_occ,
            U_1=U_1,
            Ax0_Core=Ax0_Core,
            verbose=verbose)

        self.tensors[self.pad_prop("pd_fock_mo")] = pd_fock_mo
        return pd_fock_mo

    def make_SCR3(self):
        if self.pad_prop("SCR3") in self.tensors:
            return self.tensors[self.pad_prop("SCR3")]

        nocc, nmo, nprop = self.nocc, self.nmo, self.nprop
        so = [slice(0, nocc[σ]) for σ in (α, β)]
        sv = [slice(nocc[σ], nmo) for σ in (α, β)]
        SCR3 = [2 * self.make_pd_fock_mo()[σ][:, sv[σ], so[σ]] for σ in (α, β)]

        self.tensors[self.pad_prop("SCR3")] = SCR3
        return SCR3

    def make_pd_rdm1_corr(self):
        return np.zeros((2, self.nprop, self.nmo, self.nmo))

    get_pd_fock_mo = staticmethod(get_pd_fock_mo)


class USCFPolar(USCFResp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deriv_dipole = USCFDipole.from_cls(self, self.scf, copy_all=True)

    def make_polar(self):
        if "polar" in self.tensors:
            return self.tensors["polar"]

        nprop = self.deriv_dipole.nprop
        nocc = self.nocc
        so = [slice(0, nocc[σ]) for σ in (α, β)]
        hcore_1_mo = self.deriv_dipole.make_hcore_1_mo()
        U_1 = self.deriv_dipole.make_U_1()
        pol_scf = np.zeros((nprop, nprop))
        for σ in α, β:
            pol_scf -= 2 * lib.einsum("Api, Bpi -> AB", hcore_1_mo[σ][:, :, so[σ]], U_1[σ][:, :, so[σ]])

        self.tensors["polar"] = pol_scf
        return pol_scf

    @property
    def de(self):
        return self.make_polar()

    kernel = make_polar


class UHDFTPolar(PolarBase, UHDFTResp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deriv_dipole = UHDFTDipole.from_cls(self, self.scf, copy_all=True)


if __name__ == '__main__':
    def main_1():
        from pyscf import gto, scf, dft
        np.set_printoptions(10, suppress=True, linewidth=150)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=0, spin=0, verbose=0).build()
        mf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit").run()
        mf_pol = UHDFTPolar(mf, xc="TPSS0")
        print(mf_pol.de)
        REF = np.array(
            [[5.6996863302, 0., -0.9570246008],
             [0., 1.4947978961, -0.],
             [-0.9570246008, -0., 5.2046766405]])
        # self.assertTrue(np.allclose(mf_pol.de, REF, atol=1e-6, rtol=1e-4))


        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = UHDFTResp(mf_scf, xc="TPSS0")
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 3e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)
        print(mf_pol.de - pol_num)
        print(np.allclose(mf_pol.de, pol_num, atol=1e-6, rtol=1e-4))

        # self.assertTrue(np.allclose(mf_pol.de, pol_num, atol=1e-6, rtol=1e-4))

    main_1()

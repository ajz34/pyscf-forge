from dftd3.parameters import get_damping_param
from dftd3.interface import (
    DispersionModel, DampingParam, RationalDampingParam, ZeroDampingParam,
    ModifiedRationalDampingParam, ModifiedZeroDampingParam, OptimizedPowerDampingParam)

from pyscf.dh.energy import EngBase
from pyscf.dh import util
from pyscf import lib


class DFTD3Eng(EngBase):

    def __init__(self, mf, version, atm=False, xc=None, XC=None, **kwargs):
        super().__init__(mf)

        self.param_cls = NotImplemented
        self.param = NotImplemented  # type: DampingParam
        self.version = NotImplemented  # type: str

        # upper case input XC steams from upper case transformation in xccode
        assert not (xc is not None and XC is not None), "Only one of xc or XC is None"
        xc = XC if xc is None else xc
        xc = mf.xc if xc is None else xc

        self.parse_param(version, xc=xc, atm=atm, param_input=kwargs)

    def parse_param(self, version, xc=None, atm=False, param_input=None):
        """ Parse DFT-D3 parameters.

        Parameters
        ----------
        version : int or str
        xc : str
        atm : bool
        param_input : None or dict
        """

        log = lib.logger.new_logger(verbose=self.verbose)

        # parse version
        version = str(version).lower()
        if version in ["4", "bj", "d3bj"]:
            self.param_cls = RationalDampingParam
            self.version = "bj"
        elif version in ["3", "zero", "d3zero"]:
            self.param_cls = ZeroDampingParam
            self.version = "zero"
        elif version in ["6", "bjm", "mbj", "d3mbj"]:
            self.param_cls = ModifiedRationalDampingParam
            self.version = "bjm"
        elif version in ["5", "zerom", "mzero", "d3mzero"]:
            self.param_cls = ModifiedZeroDampingParam
            self.version = "zerom"
        elif version in ["op", "d3op"]:
            self.param_cls = OptimizedPowerDampingParam
            self.version = "op"
        else:
            raise ValueError("DFTD3 version is not recognized!")
        log.debug(f"[DEBUG] Version of DFT-D3: {self.version}")

        # parse param
        param_input = {} if param_input is None else param_input
        param_input = {key.lower(): val for key, val in param_input.items()}
        if len(param_input) == 0:
            # no user-input, then try to parse DFT-D3 parameter from xc-code
            # special case for B3LYP
            if xc.lower() == "b3lypg":
                xc = "b3lyp"
            self.param = self.param_cls(method=xc, atm=atm)
            param_output = get_damping_param(method=xc, defaults=[self.version])
            log.debug(f"[DEBUG] XC for DFT-D3: {xc}")
            log.debug(f"[DEBUG] Parameters for DFT-D3: {param_output}")
        else:
            # use input parameters instead of guess
            if "s9" not in param_input and not atm:
                param_input["s9"] = 0.
            self.param = self.param_cls(**param_input)
            log.debug(f"[DEBUG] Using customize parameter for DFT-D3.")
            log.debug(f"[DEBUG] Parameters for DFT-D3: {param_input}")

    @property
    def restricted(self):
        return None

    def driver_eng_dftd3(self):
        log = lib.logger.new_logger(verbose=self.verbose)
        model = DispersionModel(self.mol.atom_charges(), self.mol.atom_coords())
        results = model.get_dispersion(self.param, grad=False)
        results = {key + "_dftd3": val for key, val in results.items()}
        util.update_results(self.results, results)
        self.e_corr = self.results["energy_dftd3"]
        log.info(f"[INFO] Energy of DFT-D3 correction: {self.e_corr}")
        return self.e_tot

    kernel = driver_eng_dftd3


if __name__ == '__main__':
    from pyscf import gto, dft
    import numpy as np

    mol = gto.Mole(atom="""
    C   1.40000000   0.00000000   0.00000000
    C   0.70000000   1.21243557   0.00000000
    C  -0.70000000   1.21243557   0.00000000
    C  -1.40000000   0.00000000   0.00000000
    C  -0.70000000  -1.21243557   0.00000000
    C   0.70000000  -1.21243557   0.00000000
    H   2.49000000   0.00000000   0.00000000
    H   1.24500000   2.15640326   0.00000000
    H  -1.24500000   2.15640326   0.00000000
    H  -2.49000000   0.00000000   0.00000000
    H  -1.24500000  -2.15640326   0.00000000
    H   1.24500000  -2.15640326   0.00000000
    C   1.40000000   0.00000000   2.00000000
    C   0.70000000   1.21243557   2.00000000
    C  -0.70000000   1.21243557   2.00000000
    C  -1.40000000   0.00000000   2.00000000
    C  -0.70000000  -1.21243557   2.00000000
    C   0.70000000  -1.21243557   2.00000000
    H   2.49000000   0.00000000   2.00000000
    H   1.24500000   2.15640326   2.00000000
    H  -1.24500000   2.15640326   2.00000000
    H  -2.49000000   0.00000000   2.00000000
    H  -1.24500000  -2.15640326   2.00000000
    H   1.24500000  -2.15640326   2.00000000
    """, basis="6-31G", unit="Angstrom", verbose=4).build()

    mf = dft.RKS(mol, xc="PBE0")
    mf_dftd3 = DFTD3Eng(mf, "bj").run()

    assert np.allclose(
        DFTD3Eng(mf, "bj").run().e_corr,
        -2.9109185161490E-02)
    assert np.allclose(
        DFTD3Eng(mf, "bj", **{"s8": 1.2177, "a1": 0.4145, "a2": 4.8593}).run().e_corr,
        -2.9109185161490E-02)

from pyscf.dh.energy import RDHBase, EngBase
from typing import Tuple, List
from pyscf.dh.util import XCList, XCDH
from pyscf import lib, scf, gto, dft, df, __config__
from pyscf.dh.energy.rdft import custom_mf

CONFIG_etb_first = getattr(__config__, "etb_first", False)
CONFIG_route_scf = getattr(__config__, "route_scf", "ri")
CONFIG_route_mp2 = getattr(__config__, "route_mp2", "ri")
CONFIG_route_iepa = getattr(__config__, "route_iepa", "ri")
CONFIG_frozen = getattr(__config__, "frozen", 0)


class DH(EngBase):
    """ Doubly hybrid object for energy evaluation.

    Attributes
    ----------
    inherited : List[Tuple[RDHBase, XCList]]
        Advanced correlation methods by inherited instances, accompanied with related exchange-correlation list.
    xc : XCDH
        Exchange-correlation object that represents both SCF part and energy part.
    log : lib.logger.Logger
         PySCF Logger.
    flags : dict
        Dictionary of additional options.
    """

    @property
    def restricted(self):
        return isinstance(self.scf, scf.rhf.RHF)

    def __init__(self, mf_or_mol, xc, flags=None):
        # cheat to generate someting by __init__ from base class
        mol = mf_or_mol if isinstance(mf_or_mol, gto.Mole) else mf_or_mol.mol
        super().__init__(scf.HF(mol))
        self.inherited = []  # type: List[Tuple[RDHBase, XCList]]
        self.xc = NotImplemented
        self.log = NotImplemented
        self.flags = flags = flags if flags is not None else dict()
        self.instantiate(mf_or_mol, xc, flags)

    def instantiate(self, mf_or_mol, xc, flags):
        """ Prepare essential objects.

        Objects to be generated in this function:
        - ``mol`` molecule object
        - ``xc`` Exchange-correlation token
        - ``_scf`` SCF object
        - ``with_df`` Density fitting object for post-SCF
        - ``frozen`` Frozen object for post-SCF

        Parameters
        ----------
        mf_or_mol : scf.hf.SCF or gto.Mole
            Molecule or SCF object.
        xc : XCDH or XCList or str
            Exchange-correlation token
        flags : dict
            Additional options
        """
        # set molecule
        mol = mf_or_mol if isinstance(mf_or_mol, gto.Mole) else mf_or_mol.mol
        self.mol = mol

        # frozen
        self.frozen = flags.get("frozen", CONFIG_frozen)

        # logger
        self.verbose = mol.verbose
        self.log = lib.logger.new_logger(verbose=self.verbose)

        # xc/xclist
        if isinstance(xc, str):
            self.xc = XCDH(xc)
        elif isinstance(xc, XCDH):
            self.xc = xc
        elif isinstance(xc, XCList):
            self.xc = XCDH(xc.token)
        else:
            assert False
        xc_scf = self.xc.xc_scf
        self.log.info(f"[INFO] Found xc_scf {self.xc.xc_scf.token}.")
        self.log.info(f"[INFO] Found xc_eng {self.xc.xc_eng.token}.")

        # build mf
        route_scf = flags.get("route_scf", CONFIG_route_scf)
        etb_first = flags.get("etb_first", CONFIG_etb_first)

        if isinstance(mf_or_mol, gto.Mole):
            if self.restricted:
                mf = dft.RKS(mol, xc=xc_scf.token)
            else:
                mf = dft.UKS(mol, xc=xc_scf.token)
            self.log.info(f"[INFO] Generate SCF object with xc {xc_scf.token}.")
        else:
            mf = mf_or_mol  # type: dft.rks.RKS or dft.uks.UKS
        do_jk = route_scf.lower() == "ri"
        if do_jk:
            auxbasis_jk = flags.get("auxbasis_jk", None)
            if auxbasis_jk is None:
                auxbasis_jk = df.aug_etb(mol) if etb_first else df.make_auxbasis(mol, mp2fit=False)
                self.log.info("[INFO] Generate auxbasis_jk ...")
                self.log.info(str(auxbasis_jk))
        else:
            auxbasis_jk = None
        mf = custom_mf(mf, xc_scf, auxbasis_or_with_df=auxbasis_jk)
        self._scf = mf

        # generate with_df (for post-SCF)
        etb_first = flags.get("etb_first", CONFIG_etb_first)

        auxbasis_ri = flags.get("auxbasis_ri", None)
        if auxbasis_ri in [None, True]:  # build auxbasis_ri anyway
            auxbasis_ri = df.aug_etb(mol) if etb_first else df.make_auxbasis(mol, mp2fit=True)
            self.log.info("[INFO] Generate auxbasis_ri ...")
            self.log.info(str(auxbasis_ri))
        if auxbasis_ri == auxbasis_jk and hasattr(self.scf, "with_df"):
            self.with_df = self.scf.with_df
            self.log.info("[INFO] Found same SCF auxiliary basis.")
        else:
            self.with_df = df.DF(mol, auxbasis=auxbasis_ri)

    def build(self):
        """ Build essential parts of doubly hybrid instance.

        Build process should be performed only once. Rebuild this instance shall not cost any time.
        """
        if self.scf.mo_coeff is None:
            self.log.info("[INFO] Molecular coefficients not found. Run SCF first.")
            self.scf.run()
        if self.scf.e_tot != 0 and not self.scf.converged:
            self.log.warn("SCF not converged!")
        if self.scf.grids.weights is None:
            self.scf.initialize_grids(dm=self.scf.make_rdm1())
        return self

    def kernel(self, **kwargs):
        with self.params.temporary_flags(kwargs):
            results = self.driver_energy_dh()
        self.params.update_results(results)
        return results

    def to_mp2(self):
        # import
        if self.restricted:
            from pyscf.dh import RMP2Conv as MP2Conv
            from pyscf.dh import RMP2RI as MP2RI
        else:
            from pyscf.dh import UMP2Conv as MP2Conv
            from pyscf.dh import UMP2RI as MP2RI

        # configurations
        route_mp2 = self.flags.get("route_mp2", CONFIG_route_mp2)
        incore_t_oovv_mp2 = self.flags.get("incore_t_oovv_mp2", NotImplemented)

        # generate instance
        if route_mp2.lower().startswith("ri"):
            mf = MP2RI.from_rdh(self, self.scf)
        elif route_mp2.lower().startswith("conv"):
            mf = MP2Conv.from_rdh(self, self.scf)
        else:
            assert False, "Not recognized route_mp2."

        # fill configurations
        if incore_t_oovv_mp2 is not NotImplemented:
            mf.incore_t_oovv_mp2 = incore_t_oovv_mp2

        return mf

    def to_iepa(self):
        # import
        if self.restricted:
            from pyscf.dh import RIEPAConv as IEPAConv
            from pyscf.dh import RIEPARI as IEPARI
        else:
            from pyscf.dh import UIEPAConv as IEPAConv
            from pyscf.dh import UIEPARI as IEPARI

        # configurations
        route_iepa = self.flags.get("route_iepa", CONFIG_route_iepa)
        tol_eng_pair_iepa = self.flags.get("tol_eng_pair_iepa", NotImplemented)
        max_cycle_iepa = self.flags.get("max_cycle_iepa", NotImplemented)
        iepa_schemes = self.flags.get("iepa_schemes", NotImplemented)

        # generate instance
        if route_iepa.lower().startswith("ri"):
            mf = IEPARI.from_rdh(self, self.scf)
        elif route_iepa.lower().startswith("conv"):
            mf = IEPAConv.from_rdh(self, self.scf)
        else:
            assert False, "Not recognized route_iepa."

        # fill configurations
        if tol_eng_pair_iepa is not NotImplemented:
            mf.conv_tol = tol_eng_pair_iepa
        if max_cycle_iepa is not NotImplemented:
            mf.max_cycle = max_cycle_iepa
        if iepa_schemes is not NotImplemented:
            mf.iepa_schemes = iepa_schemes

        return mf

    def to_ring_ccd(self):
        # import
        if self.restricted:
            from pyscf.dh import RRingCCDConv as RingCCDConv
        else:
            raise NotImplementedError

        # configurations
        tol_eng_ring_ccd = self.flags.get("tol_eng_ring_ccd", NotImplemented)
        tol_amp_ring_ccd = self.flags.get("tol_amp_ring_ccd", NotImplemented)
        max_cycle_ring_ccd = self.flags.get("max_cycle_ring_ccd", NotImplemented)

        # generated instance
        mf = RingCCDConv.from_rdh(self, self.scf)

        # fill configurations
        if tol_eng_ring_ccd is not NotImplemented:
            mf.conv_tol = tol_eng_ring_ccd
        if tol_amp_ring_ccd is not NotImplemented:
            mf.conv_tol_amp = tol_amp_ring_ccd
        if max_cycle_ring_ccd is not NotImplemented:
            mf.max_cycle = max_cycle_ring_ccd

        return mf


if __name__ == '__main__':
    def main_1():
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5").build()
        mf = DH(mol, xc="XYG3", flags={"etb_first": True, "frozen": "FreezeNobleGasCore"}).build()
        print(mf.scf.e_tot)
        print(dft.RKS(mol, xc="B3LYPg").density_fit().run().e_tot)
        print(mf.to_mp2().run().results)
        print(mf.to_iepa().run().results)
        print(mf.to_ring_ccd().run().results)

    main_1()


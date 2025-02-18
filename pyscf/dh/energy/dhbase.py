from pyscf import lib, dh
from abc import ABC, abstractmethod
import numpy as np
from pyscf import gto, dft, df, scf


class EngBase(lib.StreamObject, ABC):
    """ Base class of post-SCF energy component evaluation.

    Attributes
    ----------
    mol : gto.Mole
        Molecular instance.
    _scf : scf.hf.SCF or dft.rks.KohnShamDFT
        Self-consistent object. Compatibility to mp.mp2.MP2.
        For usual case call ``mf.scf`` instead of ``mf._scf`` is better.
    frozen : dh.util.FrozenCore or list[int] or int
        Frozen core instance that can give active orbital mask array.
        If integer or list of integers are given, then it will be considered as usual PySCF settings of frozen core.
    verbose : int
        Print level.
    max_memory : float or int
        Allowed memory in MB.
    with_df : df.DF
        Density fitting object.
    results : dict
        Saved results during computation. Not for modification.
    tensors : dict
        Saved intermediate matrices and tensors.
    _tmpfile : lib.misc.H5TmpFile
        Temporary file for storing large tensors.
    e_corr : float
        Evaluated post-SCF correlation energy.

    Notes
    -----
    This base class is designed to meet the standards of `pyscf.lib.StreamObject` and `pyscf.mp.mp2.MP2` without
    heavy modification. These classes may be sutiable to contribute PySCF with only a little change.

    What's more than general post-SCF classes in PySCF:
    - Enhanced frozen core. However, legacy frozen core settings is also acceptable.
    - Attribute omega, _results, _tensors, _tmpfile
    - Property restricted, scf
    - Some utility functions

    What's less than general post-SCF classes in PySCF:
    - We do not accept the case where SCF molecular orbitals are not the same to post-SCF orbitals.
    """
    def __init__(self, mf=None):
        self._scf = NotImplemented
        self.mol = NotImplemented
        self.with_df = NotImplemented
        self.frozen = 0
        self.verbose = NotImplemented
        self.max_memory = NotImplemented
        self.results = dict()
        self.tensors = dict()
        self._tmpfile = lib.H5TmpFile()
        self.e_corr = NotImplemented

        if mf is not None:
            self._scf = mf
            self.mol = mf.mol
            self.verbose = self.mol.verbose
            self.max_memory = self.mol.max_memory

    @property
    @abstractmethod
    def restricted(self):
        raise NotImplementedError

    @property
    def scf(self):
        # type: () -> dft.rks.RKS
        """ A more linting favourable replacement of attribute ``_scf``. """
        return self._scf

    @scf.setter
    def scf(self, mf_scf):
        self._scf = mf_scf

    @property
    def mo_coeff(self):
        # type: () -> np.ndarray
        """ Molecular orbital coefficient. """
        return self.scf.mo_coeff

    @mo_coeff.setter
    def mo_coeff(self, mo_coeff):
        if not np.allclose(mo_coeff, self.scf.mo_coeff):
            raise ValueError()

    @property
    def mo_occ(self):
        # type: () -> np.ndarray
        """ Molecular orbital occupation number. """
        return self.scf.mo_occ

    @mo_occ.setter
    def mo_occ(self, mo_occ):
        if not np.allclose(mo_occ, self.scf.mo_occ):
            raise ValueError()

    @property
    def mo_energy(self):
        # type: () -> np.ndarray
        """ Molecular orbital energy. """
        return self.scf.mo_energy

    @mo_energy.setter
    def mo_energy(self, mo_energy):
        if not np.allclose(mo_energy, self.scf.mo_energy):
            raise ValueError()

    @property
    def nmo(self):
        # type: () -> int
        """ Number of molecular orbitals. """
        return self.mo_coeff.shape[-1]

    @property
    def nocc(self):
        # type: () -> int or tuple
        """ Number of occupied orbitals. """
        nocc = (self.mo_occ > 0).sum(axis=-1)
        if isinstance(nocc, np.ndarray):
            nocc = tuple(nocc)
        return nocc

    @property
    def nvir(self):
        # type: () -> int or tuple
        """ Number of unoccupied (virtual) orbitals. """
        nvir = (self.mo_occ <= 0).sum(axis=-1)
        if isinstance(nvir, np.ndarray):
            nvir = tuple(nvir)
        return nvir

    @property
    def mask_occ(self):
        """ Mask list of occupied orbitals. """
        return self.mo_occ != 0

    @property
    def mask_occ_act(self):
        """ Mask list of occupied active orbitals. """
        return self.get_frozen_mask() & (self.mo_occ != 0)

    @property
    def mask_vir(self):
        """ Mask list of virtual orbitals. """
        return self.mo_occ == 0

    @property
    def mask_vir_act(self):
        """ Mask list of virtual active orbitals. """
        return self.get_frozen_mask() & (self.mo_occ == 0)

    @property
    def e_tot(self):
        # type: () -> float
        return self.scf.e_tot + self.e_corr

    def get_frozen_mask(self):
        # type: () -> np.ndarray
        """ Get boolean mask for the restricted reference orbitals.

        See Also
        --------
        pyscf.mp.mp2.get_frozen_mask, pyscf.mp.ump2.get_frozen_mask
        """
        if isinstance(self.frozen, dh.util.FrozenCore):
            return self.frozen.mask
        else:
            if isinstance(self.frozen, int):
                frozen = np.arange(self.frozen)
            else:
                frozen = self.frozen
            frozen_core = dh.util.FrozenCore(self.mol, self.mo_occ, rule=frozen)
            return frozen_core.mask

    @classmethod
    def from_cls(cls, mf_dh: "EngBase", *args, copy_all=False, **kwargs):
        """ Build a new (inherited) instance by a base EngBase instance.

        By default only some attributes are copied. Excluded attributes including
        - ``results``
        - ``tensors``
        - ``_tmpfile``
        - ``e_corr``
        """
        mf_new = cls(*args, **kwargs)
        mf_new.__dict__.update(mf_dh.__dict__)
        if not copy_all:
            mf_new.results = dict()
            mf_new.tensors = dict()
            mf_new._tmpfile = lib.H5TmpFile()
            mf_new.e_corr = NotImplemented
        return mf_new

    def to_resp(self, key):
        """ Transform base class instance to derived (response property) class.

        To perform transformation, one need to fill values in class dictionary.
        Currently, derived class can be

        - `resp`: First-order response class.
        """
        raise NotImplementedError(f"{key} of {self.__class__} is not implemented.")

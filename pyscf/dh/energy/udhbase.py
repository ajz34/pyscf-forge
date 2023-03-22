from typing import List

import numpy as np

from pyscf import lib
from pyscf.dh import util
from pyscf.dh.energy import RDHBase


class UDHBase(RDHBase):
    """ Unrestricted doubly hybrid class. """

    @property
    def restricted(self) -> bool:
        return False

    @property
    def nocc(self) -> List[int]:
        """ Number of occupied (alpha, beta) molecular orbitals. """
        return list(self.mol.nelec)

    @property
    def nvir(self) -> List[int]:
        """ Number of virtual (alpha, beta) molecular orbitals. """
        return [self.nmo - self.nocc[0], self.nmo - self.nocc[1]]

    def get_mask_act(self) -> np.ndarray:
        """ Get mask of active orbitals.

        Dimension: (2, nmo), boolean array
        """
        frozen_rule = self.params.flags["frozen_rule"]
        frozen_list = self.params.flags["frozen_list"]
        mask_act = util.parse_frozen_list(self.mol, self.nmo, frozen_list, frozen_rule)
        if len(mask_act.shape) == 1:
            # enforce mask to be [mask_act_alpha, mask_act_beta]
            mask_act = np.array([mask_act, mask_act])
        return mask_act

    def get_shuffle_frz(self) -> np.ndarray:
        mask_act = self.get_mask_act()
        mo_idx = np.arange(self.nmo)
        nocc = self.nocc
        shuffle_frz_occ = [np.concatenate([mo_idx[~mask_act[s]][:nocc[s]], mo_idx[mask_act[s]][:nocc[s]]])
                           for s in (0, 1)]
        shuffle_frz_vir = [np.concatenate([mo_idx[mask_act[s]][nocc[s]:], mo_idx[~mask_act[s]][nocc[s]:]])
                           for s in (0, 1)]
        shuffle_frz = np.array([np.concatenate([shuffle_frz_occ[s], shuffle_frz_vir[s]]) for s in (0, 1)])
        if not np.allclose(shuffle_frz[0], np.arange(self.nmo)) \
                or not np.allclose(shuffle_frz[1], np.arange(self.nmo)):
            self.log.warn("MO orbital indices will be shuffled.")
        return shuffle_frz

    @property
    def nCore(self) -> List[int]:
        """ Number of frozen occupied orbitals. """
        mask_act = self.get_mask_act()
        return [(~mask_act[s][:self.nocc[s]]).sum() for s in (0, 1)]

    @property
    def nOcc(self) -> List[int]:
        """ Number of active occupied orbitals. """
        mask_act = self.get_mask_act()
        return [mask_act[s][:self.nocc[s]].sum() for s in (0, 1)]

    @property
    def nVir(self) -> List[int]:
        """ Number of active virtual orbitals. """
        mask_act = self.get_mask_act()
        return [mask_act[s][self.nocc[s]:].sum() for s in (0, 1)]

    @property
    def nFrzvir(self) -> List[int]:
        """ Number of inactive virtual orbitals. """
        mask_act = self.get_mask_act()
        return [(~mask_act[s][self.nocc[s]:]).sum() for s in (0, 1)]

    @property
    def nact(self) -> List[int]:
        return [self.get_mask_act()[s].sum() for s in (0, 1)]

    def get_idx_frz_categories(self) -> List[tuple]:
        """ Get indices of molecular orbital categories.

        This function returns 4 numbers:
        [(nCore, nCore + nOcc, nCore + nOcc + nVir, nmo)_alpha, ..._beta]
        """
        return [(self.nCore[s], self.nocc[s], self.nocc[s] + self.nVir[s], self.nmo) for s in (0, 1)]

    @property
    def mo_coeff(self) -> np.ndarray:
        """ Molecular orbital coefficient. """
        shuffle_frz = self.get_shuffle_frz()
        return np.array([self.scf.mo_coeff[s][:, shuffle_frz[s]] for s in (0, 1)])

    @property
    def mo_occ(self) -> np.ndarray:
        """ Molecular orbital occupation number. """
        shuffle_frz = self.get_shuffle_frz()
        return np.array([self.scf.mo_occ[s][shuffle_frz[s]] for s in (0, 1)])

    @property
    def mo_energy(self) -> np.ndarray:
        """ Molecular orbital energy. """
        shuffle_frz = self.get_shuffle_frz()
        return np.array([self.scf.mo_energy[s][shuffle_frz[s]] for s in (0, 1)])

    @property
    def mo_coeff_act(self) -> List[np.ndarray]:
        return [self.mo_coeff[s][:, self.nCore[s]:self.nCore[s]+self.nact[s]].copy() for s in (0, 1)]

    @property
    def mo_energy_act(self) -> List[np.ndarray]:
        return [self.mo_energy[s][self.nCore[s]:self.nCore[s]+self.nact[s]].copy() for s in (0, 1)]

    def get_Y_OV(self, regenerate=False) -> List[np.ndarray]:
        """ Get cholesky decomposed ERI in MO basis (occ-vir part with frozen core).

        Dimension: (naux, nOcc, nVir) for each spin.
        """
        nact, nOcc = self.nact, self.nOcc
        if regenerate or "Y_OV" not in self.params.tensors:
            Y_OV = [util.get_cderi_mo(
                self.with_df, self.mo_coeff_act[s], None, (0, nOcc[s], nOcc[s], nact[s]),
                self.mol.max_memory - lib.current_memory()[0]
            ) for s in (0, 1)]
            self.params.tensors["Y_OV_a"] = Y_OV[0]
            self.params.tensors["Y_OV_b"] = Y_OV[1]
        else:
            Y_OV = [self.params.tensors["Y_OV_a"], self.params.tensors["Y_OV_b"]]
        return Y_OV

import re
import typing
import numpy as np
from pyscf.data import elements
from pyscf.lib import logger

if typing.TYPE_CHECKING:
    from pyscf import gto

MAX_LEVEL = 7
MAX_ANG = 4

CONFIGURATION_FC = [
    [ 0, 0, 0, 0],     # 000  GHOST
    [ 1, 0, 0, 0],     # 001  H
    [ 2, 0, 0, 0],     # 002  He
    [ 3, 0, 0, 0],     # 003  Li
    [ 4, 0, 0, 0],     # 004  Be
    [ 4, 1, 0, 0],     # 005  B
    [ 4, 2, 0, 0],     # 006  C
    [ 4, 3, 0, 0],     # 007  N
    [ 4, 4, 0, 0],     # 008  O
    [ 4, 5, 0, 0],     # 009  F
    [ 4, 6, 0, 0],     # 010  Ne
    [ 5, 6, 0, 0],     # 011  Na
    [ 6, 6, 0, 0],     # 012  Mg
    [ 6, 7, 0, 0],     # 013  Al
    [ 6, 8, 0, 0],     # 014  Si
    [ 6, 9, 0, 0],     # 015  P
    [ 6,10, 0, 0],     # 016  S
    [ 6,11, 0, 0],     # 017  Cl
    [ 6,12, 0, 0],     # 018  Ar
    [ 7,12, 0, 0],     # 019  K
    [ 8,12, 0, 0],     # 020  Ca
    [ 8,12, 1, 0],     # 021  Sc
    [ 8,12, 2, 0],     # 022  Ti
    [ 8,12, 3, 0],     # 023  V
    [ 8,12, 4, 0],     # 024  Cr
    [ 8,12, 5, 0],     # 025  Mn
    [ 8,12, 6, 0],     # 026  Fe
    [ 8,12, 7, 0],     # 027  Co
    [ 8,12, 8, 0],     # 028  Ni
    [ 8,12, 9, 0],     # 029  Cu
    [ 8,12,10, 0],     # 030  Zn
    [ 8,13,10, 0],     # 031  Ga
    [ 8,14,10, 0],     # 032  Ge
    [ 8,15,10, 0],     # 033  As
    [ 8,16,10, 0],     # 034  Se
    [ 8,17,10, 0],     # 035  Br
    [ 8,18,10, 0],     # 036  Kr
    [ 9,18,10, 0],     # 037  Rb
    [10,18,10, 0],     # 038  Sr
    [10,18,11, 0],     # 039  Y
    [10,18,12, 0],     # 040  Zr
    [10,18,13, 0],     # 041  Nb
    [10,18,14, 0],     # 042  Mo
    [10,18,15, 0],     # 043  Tc
    [10,18,16, 0],     # 044  Ru
    [10,18,17, 0],     # 045  Rh
    [10,18,28, 0],     # 046  Pd
    [10,18,19, 0],     # 047  Ag
    [10,18,20, 0],     # 048  Cd
    [10,19,20, 0],     # 049  In
    [10,20,20, 0],     # 050  Sn
    [10,21,20, 0],     # 051  Sb
    [10,22,20, 0],     # 052  Te
    [10,23,20, 0],     # 053  I
    [10,24,20, 0],     # 054  Xe
    [11,24,20, 0],     # 055  Cs
    [12,24,20, 0],     # 056  Ba
    [12,24,20, 1],     # 057  La
    [12,24,20, 2],     # 058  Ce
    [12,24,20, 3],     # 059  Pr
    [12,24,20, 4],     # 060  Nd
    [12,24,20, 5],     # 061  Pm
    [12,24,20, 6],     # 062  Sm
    [12,24,20, 7],     # 063  Eu
    [12,24,20, 8],     # 064  Gd
    [12,24,20, 9],     # 065  Tb
    [12,24,20,10],     # 066  Dy
    [12,24,20,11],     # 067  Ho
    [12,24,20,12],     # 068  Er
    [12,24,20,13],     # 069  Tm
    [12,24,20,14],     # 070  Yb
    [12,24,21,14],     # 071  Lu
    [12,24,22,14],     # 072  Hf
    [12,24,23,14],     # 073  Ta
    [12,24,24,14],     # 074  W
    [12,24,25,14],     # 075  Re
    [12,24,26,14],     # 076  Os
    [12,24,27,14],     # 077  Ir
    [12,24,28,14],     # 078  Pt
    [12,24,29,14],     # 079  Au
    [12,24,30,14],     # 080  Hg
    [12,25,30,14],     # 081  Tl
    [12,26,30,14],     # 082  Pb
    [12,27,30,14],     # 083  Bi
    [12,28,30,14],     # 084  Po
    [12,29,30,14],     # 085  At
    [12,30,30,14],     # 086  Rn
    [13,30,30,14],     # 087  Fr
    [14,30,30,14],     # 088  Ra
    [14,30,30,15],     # 089  Ac
    [14,30,30,16],     # 090  Th
    [14,30,30,17],     # 091  Pa
    [14,30,30,18],     # 092  U
    [14,30,30,19],     # 093  Np
    [14,30,30,20],     # 094  Pu
    [14,30,30,21],     # 095  Am
    [14,30,30,22],     # 096  Cm
    [14,30,30,23],     # 097  Bk
    [14,30,30,24],     # 098  Cf
    [14,30,30,25],     # 099  Es
    [14,30,30,26],     # 100  Fm
    [14,30,30,27],     # 101  Md
    [14,30,30,28],     # 102  No
    [14,30,31,28],     # 103  Lr
    [14,30,32,28],     # 104  Rf
    [14,30,33,28],     # 105  Db
    [14,30,34,28],     # 106  Sg
    [14,30,35,28],     # 107  Bh
    [14,30,36,28],     # 108  Hs
    [14,30,37,28],     # 109  Mt
    [14,30,38,28],     # 110  Ds
    [14,30,39,28],     # 111  Rg
    [14,30,40,28],     # 112  Cn
    [14,31,40,28],     # 113  Nh
    [14,32,40,28],     # 114  Fl
    [14,33,40,28],     # 115  Mc
    [14,34,40,28],     # 116  Lv
    [14,35,40,28],     # 117  Ts
    [14,36,40,28],     # 118  Og
]


class ElementConfiguration:
    """ Class to parse electronic configuration table of an chemical element. """

    element: int
    """ Element charge of concern. Should between 0 to 118; no g orbitals accepted. """

    def __init__(self, element):
        self.element = elements.charge(element)
        if self.element > len(CONFIGURATION_FC) - 1:
            raise ValueError(f"Element {self.element} too large.")

    def get_configuration_table(self):
        """ Return table of configuration of 7x4 (7 levels, spdf types of angular moments). """
        table = np.zeros((MAX_LEVEL, MAX_ANG), dtype=int)
        # start index of different angular moment
        angular_start = [l for l in range(MAX_ANG)]
        angular_num = [2 + 4 * l for l in range(MAX_ANG)]
        for angular in range(MAX_ANG):
            angular_elec = CONFIGURATION_FC[self.element][angular]
            if angular_elec == 0:
                continue
            n_start = angular_start[angular]
            n_end = n_start + angular_elec // angular_num[angular]
            table[n_start:n_end, angular] = angular_num[angular]
            assert n_end <= MAX_LEVEL, "possibly program bug"
            if n_end < MAX_LEVEL:
                table[n_end, angular] = angular_elec % angular_num[angular]
        return table

    def get_active_table(self, level, is_active):
        """ Table of configuration for active electrons.

        Parameters
        ----------
        level : int or list[int]
            Level of electrons to be active or frozen.

        is_active : bool
            Is this an active or frozen level.

        Returns
        -------
        np.ndarray
            Table of configuration for active electrons.
        """
        table = self.get_configuration_table().copy()
        MAX_LEVEL, MAX_ANG = table.shape
        if isinstance(level, int):
            active_level = [level] * MAX_ANG
        assert len(level) == MAX_ANG, "dim of active_level must be the same with number of angular momemtums"
        assert min(level) >= 0, "level must be non-negative integers"
        for angular, active in enumerate(level):
            if is_active is False:
                # check sanity
                if table[:active, angular].sum() % (2 + 4 * angular) != 0:
                    raise ValueError("Possibly active orbital electrons are frozen by current setting.")
                table[:active, angular] = 0
            else:
                non_empty_level = np.arange(MAX_LEVEL)[table[:, 0] > 0]
                if len(non_empty_level) == 0:
                    continue
                non_empty_largest = max(non_empty_level)
                freeze_level = non_empty_largest - active + 1
                if freeze_level <= 0:
                    continue
                # check sanity
                if table[:freeze_level, angular].sum() % (2 + 4 * angular) != 0:
                    raise ValueError("Possibly active orbital electrons are frozen by current setting.")
                table[:freeze_level, angular] = 0
        return table

    def get_num_core_electrons(self, active_level, by_valence=True):
        """ Get number of frozen core electrons.

        See Also
        -------
        get_active_table
        """
        return self.get_configuration_table().sum() - self.get_active_table(active_level, by_valence).sum()


class FrozenRuleBase:
    """ Base class for frozen core rules. """

    active_levels: np.ndarray
    """ Electrons to be active for each element and each shell (when ``is_active`` is True). """
    is_active: bool
    """ Is this an active or frozen level. """

    def __init__(self):
        self.define_active_levels()

    def define_active_levels(self):
        """ Defines frozen rule by implementing ``active_levels`` (119x4 array) and ``is_active`` (usually True).

        Inherited classes should implement this function. """
        raise NotImplementedError("As a base class this should be implemented, or either override get_num_core.")

    def get_num_core(self, elem):
        """ Get number of frozen electrons for an element.

        Parameters
        ----------
        elem : str or int
            Indicator of an element.

        Returns
        -------
        int
        """
        chrg = ElementConfiguration(elem).element
        return ElementConfiguration(elem).get_num_core_electrons(self.active_levels[chrg], self.is_active)

    def print_frozen_core_electrons(self):
        assert self.active_levels.shape == (len(CONFIGURATION_FC), MAX_ANG)
        list_num_core = [self.get_num_core(n) for n in range(len(CONFIGURATION_FC))]
        str_num_core = [f" {s:>3d}" for s in list_num_core]
        str_elem = [f" {elements._atom_symbol(s):>3}" for s in range(len(CONFIGURATION_FC))]
        token = ""
        # layer 0
        token += str_elem[0] + "\n" + str_num_core[0] + "\n"
        # layer 1
        token += "".join([str_elem[1], " " * 16 * 4, str_elem[2], "\n"])
        token += "".join([str_num_core[1], " " * 16 * 4, str_num_core[2], "\n"])
        # layer 2-3
        token += "".join([*str_elem[3:5], " " * 10 * 4, *str_elem[5:11], "\n"])
        token += "".join([*str_num_core[3:5], " " * 10 * 4, *str_num_core[5:11], "\n"])
        token += "".join([*str_elem[11:13], " " * 10 * 4, *str_elem[13:19], "\n"])
        token += "".join([*str_num_core[11:13], " " * 10 * 4, *str_num_core[13:19], "\n"])
        # layer 4-5
        token += "".join([*str_elem[19:37], "\n"])
        token += "".join([*str_num_core[19:37], "\n"])
        token += "".join([*str_elem[37:55], "\n"])
        token += "".join([*str_num_core[37:55], "\n"])
        # layer 6-7
        token += "".join([*str_elem[55:57], " " * 4, *str_elem[72:87], "\n"])
        token += "".join([*str_num_core[55:57], " " * 4, *str_num_core[72:87], "\n"])
        token += "".join([*str_elem[87:89], " " * 4, *str_elem[104:119], "\n"])
        token += "".join([*str_num_core[87:89], " " * 4, *str_num_core[104:119], "\n"])
        # layer La, Ac
        token += "\n"
        token += "".join([" " * 3 * 4, *str_elem[57:72], "\n"])
        token += "".join(["Lanthanides ", *str_num_core[57:72], "\n"])
        token += "".join([" " * 3 * 4, *str_elem[89:104], "\n"])
        token += "".join(["  Actinides ", *str_num_core[89:104], "\n"])
        print(token)
        return token

    @staticmethod
    def idx_alk_metal():
        """ Index list of alkali metals and alkaline earth metals. """
        return np.array([3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88])

    @staticmethod
    def idx_transition():
        """ Index of transition metals (d only, without La and Ac). """
        return np.array(list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)) + list(range(104, 113)))

    @staticmethod
    def idx_p_main():
        """ Index of p main group elements. """
        return np.array(
            list(range(5, 11)) + list(range(13, 19)) + list(range(31, 37)) + list(range(49, 55))
            + list(range(81, 87)) + list(range(113, 119))
        )

    @staticmethod
    def idx_la():
        """ Index of Lanthanum group elements. """
        return np.array(range(57, 72))

    @staticmethod
    def idx_ac():
        """ Index of Actinium group elements. """
        return np.array(range(89, 104))

    @staticmethod
    def idx_p_metal_loid():
        """ Index of metals and metalloids elements in p main group. """
        return np.array([5, 13, 14, 31, 32, 33, 49, 50, 51, 52, 81, 82, 83, 84, 113, 114, 115, 116])


class FrozenRuleNone(FrozenRuleBase):
    """ Frozen rule that no electron will be frozen. """

    def define_active_levels(self):
        self.active_levels = np.zeros((len(CONFIGURATION_FC), MAX_ANG), dtype=int)
        self.is_active = False


class FrozenRuleORCA(FrozenRuleBase):
    """ Frozen rule convention from ORCA.

    .. code-block::

        #    H                                                                  He
        #    0                                                                   0
        #   Li  Be                                           B   C   N   O   F  Ne
        #    0   0                                           2   2   2   2   2   2
        #   Na  Mg                                          Al  Si   P   S  Cl  Ar
        #    2   2                                          10  10  10  10  10  10
        #    K  Ca  Sc  Ti   V  Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
        #   10  10  10  10  10  10  10  10  10  10  10  10  18  18  18  18  18  18
        #   Rb  Sr   Y  Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te   I  Xe
        #   18  18  28  28  28  28  28  28  28  28  28  28  36  36  36  36  36  36
        #   Cs  Ba      Hf  Ta   W  Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
        #   36  36      46  46  46  46  46  46  46  46  46  68  68  68  68  68  68
        #   Fr  Ra      Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
        #   68  68     100 100 100 100 100 100 100 100 100 100 100 100 100 100 100
        #
        #               La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu
        # Lanthanides   36  36  36  36  36  36  36  36  36  36  36  36  36  36  36
        #               Ac  Th  Pa   U  Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr
        #   Actinides   68  68  68  68  68  68  68  68  68  68  68  68  68  68  68
    """
    def define_active_levels(self):
        self.active_levels = np.zeros((len(CONFIGURATION_FC), MAX_ANG), dtype=int)
        self.is_active = True
        self.active_levels[0:3] = [1, 1, 1, 1]
        self.active_levels[self.idx_alk_metal()] = [2, 2, 3, 3]
        self.active_levels[self.idx_la()] = [2, 2, 3, 3]
        self.active_levels[self.idx_ac()] = [2, 2, 3, 3]
        self.active_levels[self.idx_transition()] = [2, 2, 2, 3]
        self.active_levels[self.idx_p_main()] = [1, 1, 2, 2]
        self.active_levels[104:113] = [1, 1, 2, 2]  # post Ac transition metel


class FrozenRuleNobleGasCore(FrozenRuleBase):
    """ Frozen rule that the first layer noble gas is frozen.

    .. code-block::

        #    H                                                                  He
        #    0                                                                   0
        #   Li  Be                                           B   C   N   O   F  Ne
        #    2   2                                           2   2   2   2   2   2
        #   Na  Mg                                          Al  Si   P   S  Cl  Ar
        #   10  10                                          10  10  10  10  10  10
        #    K  Ca  Sc  Ti   V  Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
        #   18  18  18  18  18  18  18  18  18  18  18  18  18  18  18  18  18  18
        #   Rb  Sr   Y  Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te   I  Xe
        #   36  36  36  36  36  36  36  36  36  36  36  36  36  36  36  36  36  36
        #   Cs  Ba      Hf  Ta   W  Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
        #   54  54      54  54  54  54  54  54  54  54  54  54  54  54  54  54  54
        #   Fr  Ra      Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
        #   86  86      86  86  86  86  86  86  86  86  86  86  86  86  86  86  86
        #
        #               La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu
        # Lanthanides   54  54  54  54  54  54  54  54  54  54  54  54  54  54  54
        #               Ac  Th  Pa   U  Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr
        #   Actinides   86  86  86  86  86  86  86  86  86  86  86  86  86  86  86
    """
    def define_active_levels(self):
        self.active_levels = np.zeros((len(CONFIGURATION_FC), MAX_ANG), dtype=int)
        self.is_active = True
        self.active_levels[:] = [1, 1, 2, 3]


class FrozenRuleInnerNobleGasCore(FrozenRuleBase):
    r""" Frozen rule that the first layer noble gas is frozen.

    .. code-block::

        #    H                                                                  He
        #    0                                                                   0
        #   Li  Be                                           B   C   N   O   F  Ne
        #    0   0                                           0   0   0   0   0   0
        #   Na  Mg                                          Al  Si   P   S  Cl  Ar
        #    2   2                                           2   2   2   2   2   2
        #    K  Ca  Sc  Ti   V  Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
        #   10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10
        #   Rb  Sr   Y  Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te   I  Xe
        #   18  18  18  18  18  18  18  18  18  18  18  18  18  18  18  18  18  18
        #   Cs  Ba      Hf  Ta   W  Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
        #   36  36      36  36  36  36  36  36  36  36  36  36  36  36  36  36  36
        #   Fr  Ra      Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
        #   54  54      54  54  54  54  54  54  54  54  54  54  54  54  54  54  54
        #
        #               La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu
        # Lanthanides   36  36  36  36  36  36  36  36  36  36  36  36  36  36  36
        #               Ac  Th  Pa   U  Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr
        #   Actinides   54  54  54  54  54  54  54  54  54  54  54  54  54  54  54
    """

    def define_active_levels(self):
        self.active_levels = np.zeros((len(CONFIGURATION_FC), MAX_ANG), dtype=int)
        self.is_active = True
        self.active_levels[:] = [2, 2, 3, 4]


class FrozenRuleG2(FrozenRuleBase):
    """ Frozen rule of G2, should be the same to Gaussian's FreezeG2.

    .. code-block::

        #    H                                                                  He
        #    0                                                                   0
        #   Li  Be                                           B   C   N   O   F  Ne
        #    2   2                                           2   2   2   2   2   2
        #   Na  Mg                                          Al  Si   P   S  Cl  Ar
        #   10  10                                          10  10  10  10  10  10
        #    K  Ca  Sc  Ti   V  Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
        #   10  10  18  18  18  18  18  18  18  18  18  18  28  28  28  28  28  28
        #   Rb  Sr   Y  Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te   I  Xe
        #   28  28  36  36  36  36  36  36  36  36  36  36  46  46  46  46  46  46
        #   Cs  Ba      Hf  Ta   W  Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
        #   46  46      54  54  54  54  54  54  54  54  54  78  78  78  78  78  78
        #   Fr  Ra      Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
        #   78  78      86  86  86  86  86  86  86  86  86 110 110 110 110 110 110
        #
        #               La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu
        # Lanthanides   54  54  54  54  54  54  54  54  54  54  54  54  54  54  54
        #               Ac  Th  Pa   U  Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr
        #   Actinides   86  86  86  86  86  86  86  86  86  86  86  86  86  86  86
    """
    def define_active_levels(self):
        self.active_levels = np.zeros((len(CONFIGURATION_FC), MAX_ANG), dtype=int)
        self.is_active = True
        self.active_levels[:] = [1, 1, 2, 3]
        self.active_levels[self.idx_alk_metal()[self.idx_alk_metal() > 18]] = [2, 2, 2, 3]
        self.active_levels[self.idx_p_main()] = [1, 1, 1, 2]


class FrozenRuleG3(FrozenRuleBase):
    """ Frozen rule of G3, should be the same to Gaussian's FreezeG3.

    .. code-block::

        #    H                                                                  He
        #    0                                                                   0
        #   Li  Be                                           B   C   N   O   F  Ne
        #    2   2                                           2   2   2   2   2   2
        #   Na  Mg                                          Al  Si   P   S  Cl  Ar
        #   10  10                                          10  10  10  10  10  10
        #    K  Ca  Sc  Ti   V  Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
        #   10  10  18  18  18  18  18  18  18  18  18  18  18  18  18  18  18  18
        #   Rb  Sr   Y  Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te   I  Xe
        #   36  36  36  36  36  36  36  36  36  36  36  36  36  36  36  36  36  36
        #   Cs  Ba      Hf  Ta   W  Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
        #   54  54      54  54  54  54  54  54  54  54  54  54  54  54  54  54  54
        #   Fr  Ra      Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
        #   86  86      86  86  86  86  86  86  86  86  86  86  86  86  86  86  86
        #
        #               La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu
        # Lanthanides   54  54  54  54  54  54  54  54  54  54  54  54  54  54  54
        #               Ac  Th  Pa   U  Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr
        #   Actinides   86  86  86  86  86  86  86  86  86  86  86  86  86  86  86
    """
    def define_active_levels(self):
        self.active_levels = np.zeros((len(CONFIGURATION_FC), MAX_ANG), dtype=int)
        self.is_active = True
        self.active_levels[:] = [1, 1, 2, 3]
        self.active_levels[[19, 20]] = [2, 2, 2, 3]


class FrozenRuleG4(FrozenRuleBase):
    """ Frozen rule of G4, should be the same to Gaussian's FreezeG4.

    .. code-block::

        #    H                                                                  He
        #    0                                                                   0
        #   Li  Be                                           B   C   N   O   F  Ne
        #    2   2                                           2   2   2   2   0   0
        #   Na  Mg                                          Al  Si   P   S  Cl  Ar
        #   10  10                                          10  10  10  10  10  10
        #    K  Ca  Sc  Ti   V  Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
        #   10  10  18  18  18  18  18  18  18  18  18  18  18  18  18  18  18  18
        #   Rb  Sr   Y  Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te   I  Xe
        #   36  36  36  36  36  36  36  36  36  36  36  36  36  36  36  36  36  36
        #   Cs  Ba      Hf  Ta   W  Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
        #   54  54      54  54  54  54  54  54  54  54  54  54  54  54  54  54  54
        #   Fr  Ra      Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
        #   86  86      86  86  86  86  86  86  86  86  86  86  86  86  86  86  86
        #
        #               La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu
        # Lanthanides   54  54  54  54  54  54  54  54  54  54  54  54  54  54  54
        #               Ac  Th  Pa   U  Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr
        #   Actinides   86  86  86  86  86  86  86  86  86  86  86  86  86  86  86
    """
    def define_active_levels(self):
        self.active_levels = np.zeros((len(CONFIGURATION_FC), MAX_ANG), dtype=int)
        self.is_active = True
        self.active_levels[:] = [1, 1, 2, 3]
        self.active_levels[[9, 10, 19, 20]] = [2, 2, 2, 3]


class FrozenRuleSmallCore(FrozenRuleBase):
    """ Frozen rule of small core.

    .. code-block::

        #    H                                                                  He
        #    0                                                                   0
        #   Li  Be                                           B   C   N   O   F  Ne
        #    2   2                                           2   2   2   2   2   2
        #   Na  Mg                                          Al  Si   P   S  Cl  Ar
        #    2   2                                          10  10  10  10  10  10
        #    K  Ca  Sc  Ti   V  Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
        #   10  10  18  18  18  18  18  18  18  18  18  18  18  18  18  18  18  18
        #   Rb  Sr   Y  Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te   I  Xe
        #   28  28  36  36  36  36  36  36  36  36  36  36  36  36  36  36  36  36
        #   Cs  Ba      Hf  Ta   W  Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
        #   46  46      54  54  54  54  54  54  54  54  54  54  54  54  54  54  54
        #   Fr  Ra      Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
        #   78  78      86  86  86  86  86  86  86  86  86  86  86  86  86  86  86
        #
        #               La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu
        # Lanthanides   54  54  54  54  54  54  54  54  54  54  54  54  54  54  54
        #               Ac  Th  Pa   U  Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr
        #   Actinides   86  86  86  86  86  86  86  86  86  86  86  86  86  86  86

    Notes
    -----
    See original article of [1]_.

    .. [1] Rassolov, Vitaly A, John A Pople, Paul C Redfern, and Larry A Curtiss. “The Definition of Core Electrons.”
           Chem. Phys. Lett. 350, (5–6), 573–76. https://doi.org/10.1016/S0009-2614(01)01345-8.
    """
    def define_active_levels(self):
        self.active_levels = np.zeros((len(CONFIGURATION_FC), MAX_ANG), dtype=int)
        self.is_active = True
        self.active_levels[:] = [1, 1, 2, 3]
        self.active_levels[self.idx_alk_metal()] = [2, 2, 2, 3]
        self.active_levels[3:5] = [1, 1, 2, 3]


class FrozenRuleLargeCore(FrozenRuleBase):
    """ Frozen rule of small core.

    .. code-block::

        #    H                                                                  He
        #    0                                                                   0
        #   Li  Be                                           B   C   N   O   F  Ne
        #    2   2                                           2   2   2   2   2   2
        #   Na  Mg                                          Al  Si   P   S  Cl  Ar
        #   10  10                                          10  10  10  10  10  10
        #    K  Ca  Sc  Ti   V  Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
        #   18  18  18  18  18  18  18  18  18  18  18  18  28  28  28  28  28  28
        #   Rb  Sr   Y  Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te   I  Xe
        #   36  36  36  36  36  36  36  36  36  36  36  36  46  46  46  46  46  46
        #   Cs  Ba      Hf  Ta   W  Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
        #   54  54      68  68  68  68  68  68  68  68  68  78  78  78  78  78  78
        #   Fr  Ra      Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
        #   86  86     100 100 100 100 100 100 100 100 100 110 110 110 110 110 110
        #
        #               La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu
        # Lanthanides   54  54  54  54  54  54  54  54  54  54  54  54  54  54  54
        #               Ac  Th  Pa   U  Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr
        #   Actinides   86  86  86  86  86  86  86  86  86  86  86  86  86  86  86

    Notes
    -----
    See original article of [1]_.

    .. [1] Rassolov, Vitaly A, John A Pople, Paul C Redfern, and Larry A Curtiss. “The Definition of Core Electrons.”
           Chem. Phys. Lett. 350, (5–6), 573–76. https://doi.org/10.1016/S0009-2614(01)01345-8.
    """
    def define_active_levels(self):
        self.active_levels = np.zeros((len(CONFIGURATION_FC), MAX_ANG), dtype=int)
        self.is_active = True
        self.active_levels[:] = [1, 1, 2, 3]
        self.active_levels[self.idx_p_main()] = [1, 1, 1, 2]
        self.active_levels[self.idx_transition()] = [1, 1, 2, 2]


FrozenRules = {
    "none": FrozenRuleNone(),
    # orca/pyscf rule
    "pyscf": FrozenRuleORCA(),
    "orca": FrozenRuleORCA(),
    # gaussian rule
    # default for other basis: FreezeNobleGasCore
    # default for 6-31G: FreezeG2
    "freezenoblegascore": FrozenRuleNobleGasCore(),
    "frzngc": FrozenRuleNobleGasCore(),
    "freezeinnernoblegascore": FrozenRuleInnerNobleGasCore(),
    "frzingc": FrozenRuleInnerNobleGasCore(),
    "fc1": FrozenRuleInnerNobleGasCore(),
    "freezeg2": FrozenRuleG2(),
    "freezeg3": FrozenRuleG3(),
    "freezeg4": FrozenRuleG4(),
    "frzg2": FrozenRuleG2(),
    "frzg3": FrozenRuleG3(),
    "frzg4": FrozenRuleG4(),
    # need double check?
    "smallcore": FrozenRuleSmallCore(),
    "largecore": FrozenRuleLargeCore(),
}


class FrozenCore:
    """ Frozen core class that gives mask of active orbitals. """

    def __init__(
            self, mol, mo_occ,
            rule=None, ecp_only=False, mo_energy=None):
        """
        Parameters
        ----------
        mol : gto.Mole
            Molecular object.
        mo_occ : np.ndarray
            Molecular orbital occupation numbers.
        rule : str or int or tuple or dict or list
            Rules of frozen core.
        ecp_only : bool
            True if not counting additional frozen occupied electrons when atom have ECP electrons.
            This option is only applied for rule-based frozen electron selection. If total number is given by
            tuple of integers, this option is not applied.
        mo_energy : np.ndarray
            Molecular orbital energies. Required if `rule="window"`.

        Notes
        -----
        Format of rules accepted here is
        - list: Index of frozen orbitals (the same convention to PySCF)
        - string: String rule that applies a complete resolution (such as ORCA, FreezeNobleGasCore, etc.)
        - int, tuple[int, int]: Number of electrons of occupied and virtual (**not spacial orbitals**)
        - dict[int or str, Any]: Element-wise string rule or tuple of integers
        - ``("EnergyWindow", (eng_min, eng_max))``: Molecular orbital energy window selection
        """
        # input attributes
        self.mol = mol
        self.mo_occ = mo_occ
        self.mo_energy = mo_energy
        self.rule = rule
        self.ecp_only = ecp_only
        # derived attributes
        self._mask = NotImplemented
        self._frozen = NotImplemented

    def reset(self):
        self._mask = NotImplemented
        self._frozen = NotImplemented

    def check_sanity(self):
        # check dimensions
        assert self.mo_occ.ndim in (1, 2)
        assert self.mo_occ.sum() == self.mol.nelectron
        if self.mo_energy is not None:
            assert self.mo_occ.shape == self.mo_energy.shape
        # check occupation number is not floats
        assert np.allclose(self.mo_occ.round(), self.mo_occ)
        # check if occupation number is sorted large->small
        mo_occ_sorted = self.mo_occ.copy()
        mo_occ_sorted.sort(axis=-1)  # reversed order
        if mo_occ_sorted.ndim == 1:
            assert np.allclose(mo_occ_sorted[::-1], self.mo_occ)
        else:
            assert np.allclose(mo_occ_sorted[:, ::-1], self.mo_occ)

    # region Mask Generation

    @staticmethod
    def parse_frozen_numbers(mol, rule, ecp_only=False):
        """ Parse frozen orbital numbers by existing rule.

        Parameters
        ----------
        mol : gto.Mole
        rule : str or int or tuple[int, int] or dict
        ecp_only : bool

        Returns
        -------
        tuple[int, int]
        """
        if rule is None:
            return 0, 0
        if isinstance(rule, tuple):
            assert len(rule) == 2 and isinstance(rule[0], int) and isinstance(rule[1], int)
            return rule
        if isinstance(rule, dict):
            rule = {elements.charge(key): val for (key, val) in rule.items()}

        f_occ, f_vir = 0, 0
        for idx in range(mol.natm):
            # basic information of atom
            nelec = mol.atom_charge(idx)
            nchrg = elements.charge(mol.atom_symbol(idx))
            is_ecp = nelec != nchrg
            f_occ_atom, f_vir_atom = 0, 0
            # determine rule of current atom
            if isinstance(rule, dict):
                rule_atom = rule[nchrg]
            else:
                rule_atom = rule
            # determine frozen-core number of atom
            if isinstance(rule_atom, str):
                f_occ_atom = FrozenRules[re.sub("[-_ ]", "", rule_atom.lower())].get_num_core(nchrg)
            elif isinstance(rule_atom, FrozenRuleBase):
                f_occ_atom = rule_atom.get_num_core(nchrg)
            elif isinstance(rule_atom, tuple):
                f_occ_atom, f_vir_atom = rule_atom
            # finalize frozen core evaluation of atom
            if is_ecp and ecp_only:
                f_occ_atom = 0
            elif is_ecp:
                f_occ_atom -= min(f_occ_atom, nchrg - nelec)
            f_occ += f_occ_atom
            f_vir += f_vir_atom

        return f_occ, f_vir

    @staticmethod
    def parse_active_mask_by_elec(mo_occ, f_occ, f_vir):
        """ Given MO occupation numbers and frozen electrons, parse active mask array.

        This function try to handle number of electrons that could be filled in each orbital.
        This information will be utilized to freeze virtual orbitals.

        Parameters
        ----------
        mo_occ : np.ndarray
            MO occupation numbers. Must be two dimensions (if RHF or GHF, make first dimension to be one).
        f_occ : int
            Number of frozen occupied electrons to be masked.
        f_vir : int
            Number of frozen virtual "electrons" to be masked.

        Returns
        -------
        np.ndarray
            Mask of active MO orbital.
        """
        assert mo_occ.ndim == 2, "Must transform mo_occ to 2 dimension first, see `parse_active_mask`"
        orb_elec = int(np.max(mo_occ))  # number of electron in one MO
        nset = mo_occ.shape[0]
        if not (f_occ % (nset * orb_elec) == 0 and f_vir % (nset * orb_elec) == 0):
            raise ValueError("Could not partially freeze orbitals with several spins or half-frozen.")
        n_occ_frz = f_occ // (nset * orb_elec)
        n_vir_frz = f_vir // (nset * orb_elec)
        mask = np.ones(mo_occ.shape, dtype=bool)
        mask[:, :n_occ_frz] = False
        if n_vir_frz > 0:
            mask[:, -n_vir_frz:] = False
        return mask

    @staticmethod
    def parse_active_mask_by_energy_window(mo_occ, mo_energy, energy_window):
        """ Given MO energies, parse active mask array.

        Parameters
        ----------
        mo_occ : np.ndarray
            MO occupation numbers.
        mo_energy : np.ndarray
            MO energies. Must be the same dimension to `mo_occ`.
        energy_window : tuple[float, float]
            Energies window. Orbital energies falls into this region considered as active orbital.

        Returns
        -------
        np.ndarray
            Mask of active MO orbital.
        """
        assert mo_occ.shape == mo_energy.shape
        assert len(energy_window) == 2, "Energy window should be two floats denoting energy range."
        return (mo_energy >= min(energy_window)) & (mo_energy <= max(energy_window))

    @staticmethod
    def parse_active_mask_by_gaussian_window(mo_occ, gaussian_window):
        assert mo_occ.ndim == 2, "Must transform mo_occ to 2 dimension first, see `parse_active_mask`"
        raise NotImplementedError

    def get_active_mask(self):
        """ Parse active mask from attributes in instance.

        Note that this mask array is based on input `mo_occ`, which has not been shuffled.

        Returns
        -------
        np.ndarray
            Mask of active MO orbital. Shape is the same to `mo_occ`.
        """
        self.check_sanity()

        nset = self.mo_occ.shape[0] if self.mo_occ.ndim == 2 else 1
        nmo = self.mo_occ.shape[-1]
        mo_occ = self.mo_occ if self.mo_occ.ndim == 2 else self.mo_occ.reshape((1, -1))

        if self.mo_energy is not None:
            mo_energy = self.mo_energy if self.mo_energy.ndim == 1 else self.mo_energy.reshape((1, -1))
            assert mo_occ.shape == mo_energy.shape
        else:
            mo_energy = None

        mask = np.ones((nset, nmo), dtype=bool)

        if isinstance(self.rule, list or np.ndarray):
            # 1. case of frozen/active list defined, highest priority
            # first perform mask; if self.act, then reverse mask in order to make active orbitals to be True
            arr = self.rule
            if not hasattr(arr[0], "__iter__"):
                # one list of frozen orbitals:
                mask[:, arr] = False
            else:
                # multiple lists of frozen orbitals
                assert len(arr) == nset, "Number of frozen lists is not the same to number of occupation sets."
                for i, arr_i in enumerate(arr):
                    mask[i, arr_i] = False
        elif isinstance(self.rule, tuple) and isinstance(self.rule[0], "str"):
            # 2. case of special treatments
            assert len(self.rule)
            if re.sub("[-_ ]", "", self.rule[0]).lower() == "energywindow":
                # 2.1 energy window
                mask = self.parse_active_mask_by_energy_window(mo_occ, mo_energy, self.rule[1])
            elif re.sub("[-_ ]", "", self.rule[2]).lower() == "gaussianwindow":
                # 2.2 Gaussian handling of window
                mask = self.parse_active_mask_by_gaussian_window(mo_occ, self.rule[1])
        else:
            # 3. case of rule-based frozen core
            f_occ, f_vir = self.parse_frozen_numbers(self.mol, self.rule, self.ecp_only)
            mask = self.parse_active_mask_by_elec(mo_occ, f_occ, f_vir)

        # reshape mask to original form
        mask.shape = self.mo_occ.shape
        return mask

    # endregion

    @property
    def mask(self):
        """ Mask of molecular occupied orbitals. """
        if self._mask is NotImplemented:
            self._mask = self.get_active_mask()
        return self._mask

    def frozen(self):
        """ Index of frozen core orbitals. """
        if self._frozen is NotImplemented:
            if self.mask.ndim == 1:
                frozen = np.arange(self.mask.shape[0])[self.mask]
            else:
                nset, nmo = self.mask.shape
                frozen = [np.arange(nmo)[self.mask[n]] for n in range(nset)]
            self._frozen = frozen
        return self._frozen


if __name__ == '__main__':
    def main_1():
        from pyscf import mp, gto, scf
        elemcfg = ElementConfiguration("Pm")
        mol = gto.Mole(atom="I 0 0 0; H 0 0 1; O 0 1 0", basis="def2-TZVP", ecp="def2-TZVP").build()
        mf = scf.RHF(mol).run()
        fc = FrozenCore(mol, mf.mo_occ, rule="FreezeNobleGasCore", ecp_only=False)
        fc = FrozenCore(mol, mf.mo_occ, rule=[1, 2], ecp_only=False)
        print(fc.mask)

    def main_2():

        # FrozenRuleNone().print_frozen_core_electrons()
        # FrozenRuleORCA().print_frozen_core_electrons()
        # FrozenRuleNobleGasCore().print_frozen_core_electrons()
        # FrozenRuleInnerNobleGasCore().print_frozen_core_electrons()
        # FrozenRuleLargeCore().print_frozen_core_electrons()
        # FrozenRuleLargeCore().print_frozen_core_electrons()
        FrozenRuleLargeCore().print_frozen_core_electrons()

    main_2()

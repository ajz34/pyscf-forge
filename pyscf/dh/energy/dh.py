from pyscf.dh.energy import EngBase
from typing import Tuple, List
from pyscf.dh import util
from pyscf.dh.util import XCType, XCList, XCDH, update_results, pad_omega
from pyscf.dh.energy.hdft.rhdft import get_rho, numint_customized
from pyscf import lib, scf, gto, dft, df, __config__
from pyscf.dh.energy.hdft.rhdft import custom_mf


CONFIG_etb_first = getattr(__config__, "etb_first", False)
CONFIG_route = getattr(__config__, "route", "ri")
CONFIG_route_scf = getattr(__config__, "route_scf", CONFIG_route)
CONFIG_route_mp2 = getattr(__config__, "route_mp2", CONFIG_route)
CONFIG_route_iepa = getattr(__config__, "route_iepa", CONFIG_route)
CONFIG_frozen = getattr(__config__, "frozen", 0)


def _process_xc_split(xc_list):
    """ Split exch-corr list by certain type of xc.

    Exch-corr list of input should be energy functional.

    Parameters
    ----------
    xc_list : XCList

    Returns
    -------
    dict[str, XCList]
    """

    result = dict()
    # 1. low-rung
    xc_extracted = xc_list.copy()
    xc_low_rung = xc_extracted.extract_by_xctype(XCType.RUNG_LOW)
    xc_extracted = xc_extracted.remove(xc_low_rung)
    result["low_rung"] = xc_low_rung
    # 2. advanced correlation
    # 2.1 IEPA (may also resolve evaluation of MP2)
    xc_iepa = xc_extracted.extract_by_xctype(XCType.IEPA)
    if len(xc_iepa) != 0:
        xc_iepa = xc_extracted.extract_by_xctype(XCType.IEPA | XCType.MP2)
        xc_extracted = xc_extracted.remove(xc_iepa)
    result["iepa"] = xc_iepa
    # 2.2 MP2
    xc_mp2 = xc_extracted.extract_by_xctype(XCType.MP2 | XCType.RSMP2)
    xc_extracted = xc_extracted.remove(xc_mp2)
    result["mp2"] = xc_mp2
    # 2.3 Ring CCD
    xc_ring_ccd = xc_extracted.extract_by_xctype(XCType.RS_RING_CCD)
    xc_extracted = xc_extracted.remove(xc_ring_ccd)
    result["ring_ccd"] = xc_ring_ccd

    # finalize
    if len(xc_extracted) > 0:
        raise RuntimeError(f"Some xc terms not evaluated! Possibly bg of program.\nXC not evaluated: {xc_extracted}")
    return result


def _process_energy_exx(mf_dh, xc_list, force_evaluate=False):
    """ Evaluate exact exchange energy.

    Parameters
    ----------
    mf_dh : DH
    xc_list : XCList
    force_evaluate : bool

    Returns
    -------
    tuple[XCList, float]
    """
    mf_scf = mf_dh.to_scf()
    log = lib.logger.new_logger(verbose=mf_dh.verbose)
    xc_exx = xc_list.extract_by_xctype(XCType.EXX)
    if len(xc_exx) == 0:
        return xc_list, 0
    xc_extracted = xc_list.remove(xc_exx, inplace=False)
    log.info(f"[INFO] XCList extracted by process_energy_exx: {xc_exx.token}")
    log.info(f"[INFO] XCList remains   by process_energy_exx: {xc_extracted.token}")
    results = dict()
    eng_tot = 0
    for info in xc_exx:
        log.info(f"[INFO] EXX to be evaluated: {info.token}")
        # determine omega
        if info.name == "LR_HF":
            assert len(info.parameters) == 1
            omega = info.parameters[0]
        elif info.name == "HF":
            omega = 0
        else:
            assert False, "Only accept LR_HF or HF, no SR_HF or anything else like second-order exchange, etc"
        name_eng_exx_HF = util.pad_omega("eng_exx_HF", omega)
        if force_evaluate or name_eng_exx_HF not in mf_dh.results:
            log.info(f"[INFO] Evaluate {name_eng_exx_HF}")
            results.update(mf_scf.get_energy_exactx(mf_scf.scf, mf_scf.scf.make_rdm1(), omega=omega))
            eng = results[name_eng_exx_HF]
        else:
            log.info(f"[INFO] {name_eng_exx_HF} is evaluated. Take previous evaluated value.")
            eng = mf_dh.results[name_eng_exx_HF]
        eng = info.fac * eng
        log.note(f"[RESULT] Energy of exchange {info.token}: {eng:20.12f}")
        eng_tot += eng

    update_results(mf_dh.results, results)
    return xc_extracted, eng_tot


def _process_energy_mp2(mf_dh, xc_mp2, force_evaluate=False):
    """ Evaluate MP2 correlation energy.

    Parameters
    ----------
    mf_dh : DH
    xc_mp2 : XCList
    force_evaluate : bool

    Returns
    -------
    tuple[XCList, float]
    """
    if len(xc_mp2) == 0:
        return 0

    log = lib.logger.new_logger(verbose=mf_dh.verbose)
    log.info(f"[INFO] XCList to be evaluated by process_energy_mp2: {xc_mp2.token}")

    def comput_mp2():
        eng_tot = 0
        results = mf_dh.results
        for info in xc_mp2:
            if XCType.MP2 in info.type:
                c_os, c_ss = info.parameters
                omega = 0
            else:
                omega, c_os, c_ss = info.parameters
            eng = info.fac * (
                + c_os * results[pad_omega("eng_corr_MP2_OS", omega)]
                + c_ss * results[pad_omega("eng_corr_MP2_SS", omega)])
            log.note(f"[RESULT] Energy of correlation {info.token}: {eng:20.12f}")
            eng_tot += eng
        return eng_tot

    def force_comput_mp2():
        for info in xc_mp2:
            if XCType.MP2 in info.type:
                c_os, c_ss = info.parameters
                omega = 0
            else:
                omega, c_os, c_ss = info.parameters
            if pad_omega("eng_corr_MP2_SS", omega) in mf_dh.results and not force_evaluate:
                continue  # results have been evaluated
            mf_mp2 = mf_dh.to_mp2(omega=omega, c_os=c_os, c_ss=c_ss).run()
            update_results(mf_dh.results, mf_mp2.results)
            mf_dh.inherited["mp2"][1].append(mf_mp2)
        eng_tot = comput_mp2()
        return eng_tot

    if force_evaluate:
        eng_tot = force_comput_mp2()
    else:
        try:
            eng_tot = comput_mp2()
        except KeyError:
            eng_tot = force_comput_mp2()

    return eng_tot


def _process_energy_iepa(mf_dh, xc_iepa, force_evaluate=False):
    """ Evaluate IEPA-like correlation energy.

    Parameters
    ----------
    mf_dh : DH
    xc_iepa : XCList
    force_evaluate : bool

    Returns
    -------
    float
    """
    log = lib.logger.new_logger(verbose=mf_dh.verbose)
    if len(xc_iepa) == 0:
        return 0

    # run MP2 while in IEPA if found
    log.info(f"[INFO] XCList to be evaluated by process_energy_iepa: {xc_iepa.token}")

    # prepare IEPA
    iepa_schemes = [info.name for info in xc_iepa]
    log.info(f"[INFO] Detected IEPAs: {iepa_schemes}")

    def comput_iepa():
        eng_tot = 0
        results = mf_dh.results
        for info in xc_iepa:
            eng = info.fac * (
                + info.parameters[0] * results[f"eng_corr_{info.name}_OS"]
                + info.parameters[1] * results[f"eng_corr_{info.name}_SS"])
            log.note(f"[RESULT] Energy of correlation {info.token}: {eng:20.12f}")
            eng_tot += eng
        return eng_tot

    def force_comput_iepa():
        mf_iepa = mf_dh.to_iepa().run(iepa_schemes=iepa_schemes)
        update_results(mf_dh.results, mf_iepa.results)
        eng_tot = comput_iepa()
        mf_dh.inherited["iepa"][1].append(mf_iepa)
        return eng_tot

    if force_evaluate:
        eng_tot = force_comput_iepa()
    else:
        try:
            eng_tot = comput_iepa()
        except KeyError:
            eng_tot = force_comput_iepa()

    return eng_tot


def _process_energy_ring_ccd(mf_dh, xc_ring_ccd, force_evaluate=False):
    """ Evaluate Ring-CCD-like correlation energy.

    Parameters
    ----------
    mf_dh : DH
    xc_ring_ccd : XCList
    force_evaluate : bool

    Returns
    -------
    tuple[XCList, float]
    """
    if len(xc_ring_ccd) == 0:
        return 0

    log = lib.logger.new_logger(verbose=mf_dh.verbose)
    log.info(f"[INFO] XCList to be evaluated by process_energy_drpa: {xc_ring_ccd.token}")

    # generate omega list
    # parameter of RS-Ring-CCD: omega, c_os, c_ss
    omega_list = []
    for xc_info in xc_ring_ccd:
        omega_list.append(xc_info.parameters[0])

    def comput_ring_ccd():
        eng_tot = 0
        results = mf_dh.results
        for xc_info in xc_ring_ccd:
            omega, c_os, c_ss = xc_info.parameters
            eng = xc_info.fac * (
                + c_os * results[pad_omega("eng_corr_RING_CCD_OS", omega)]
                + c_ss * results[pad_omega("eng_corr_RING_CCD_SS", omega)])
            log.note(f"[RESULT] Energy of correlation {xc_info.token}: {eng:20.12f}")
            eng_tot += eng
        return eng_tot

    def force_comput_ring_ccd():
        for omega in omega_list:
            if pad_omega("eng_corr_RING_CCD_SS", omega) in mf_dh.results and not force_evaluate:
                continue  # results have been evaluated
            mf_ring_ccd = mf_dh.to_ring_ccd(omega=omega).run()
            update_results(mf_dh.results, mf_ring_ccd.results)
            mf_dh.inherited["ring_ccd"][1].append(mf_ring_ccd)
        eng_tot = comput_ring_ccd()
        return eng_tot

    if force_evaluate:
        eng_tot = force_comput_ring_ccd()
    else:
        try:
            eng_tot = comput_ring_ccd()
        except KeyError:
            eng_tot = force_comput_ring_ccd()

    return eng_tot


def _process_energy_low_rung(mf_dh, xc_list):
    """ Evaluate low-rung pure DFT exchange/correlation energy.

    Parameters
    ----------
    mf_dh : DH
    xc_list : XCList

    Returns
    -------
    float
    """
    # logic of this function
    # 1. first try the whole low_rung token
    # 2. if low_rung failed, split pyscf_parsable and other terms
    # 2.1 try if customized numint is possible (this makes evaluation of energy faster)
    # 2.2 pyscf_parsable may fail when different omegas occurs
    #     split exx and pure dft evaluation (TODO: how rsh pure functionals to be evaluated?)
    # 2.3 even if this fails, split evaluate components term by term (XCInfo by XCInfo)
    log = lib.logger.new_logger(verbose=mf_dh.verbose)
    mf_scf = mf_dh.to_scf()
    ni = dft.numint.NumInt()
    log.info(f"[INFO] XCList to be parsed in process_energy_low_rung: {xc_list.token}")

    if len(xc_list) == 0:
        return 0

    def handle_multiple_omega():
        log.warn(
            "Low-rung DFT has different values of omega found for RSH functional.\n"
            "We evaluate EXX and pure DFT in separate.\n"
            "This may cause problem that if some pure DFT components is omega-dependent "
            "and our program currently could not handle it.\n"
            "Anyway, this is purely experimental and use with caution.")
        eng_tot = 0
        xc_exx = xc_list.extract_by_xctype(XCType.EXX)
        xc_non_exx = xc_list.remove(xc_exx, inplace=False)
        log.info(f"[INFO] XCList extracted by process_energy_low_rung (handle_multiple_omega, "
                 f"exx): {xc_exx.token}")
        log.info(f"[INFO] XCList extracted by process_energy_low_rung (handle_multiple_omega, "
                 f"non_exx): {xc_non_exx.token}")
        if len(xc_exx) != 0:
            xc_exx_extracted, eng_exx = _process_energy_exx(mf_dh, xc_exx)
            eng_non_exx = _process_energy_low_rung(mf_dh, xc_non_exx)
            assert len(xc_exx_extracted) == 0
            eng_tot += eng_exx + eng_non_exx
            log.note(f"[RESULT] Energy of process_energy_low_rung (handle_multiple_omega): {eng_tot}")
            return eng_tot
        else:
            log.warn(
                "Seems pure-DFT part has different values of omega.\n"
                "This may caused by different hybrids with different omegas.\n"
                "We evaluate each term one-by-one.")
            xc_non_exx_list = [XCList().build_from_list([xc_info]) for xc_info in xc_non_exx]
            for xc_non_exx_item in xc_non_exx_list:
                eng_item = _process_energy_low_rung(mf_dh, xc_non_exx_item)
                eng_tot += eng_item
            log.note(f"[RESULT] Energy of process_energy_low_rung (handle_multiple_omega): {eng_tot}")
            return eng_tot

    numint = None
    # perform the logic of function
    try:
        ni._xc_type(xc_list.token)
    except (ValueError, KeyError) as err:
        if isinstance(err, ValueError) and "Different values of omega" in err.args[0]:
            return handle_multiple_omega()
        elif isinstance(err, KeyError) and "Unknown" in err.args[0] \
                or isinstance(err, ValueError) and "too many values to unpack" in err.args[0]:
            log.info("[INFO] Unknown functional to PySCF. Try build custom numint.")
            try:
                numint = numint_customized(xc_list)
            except ValueError as err_numint:
                if "Different values of omega" in err_numint.args[0]:
                    log.warn("We first resolve different omegas, and we later move on numint.")
                    return handle_multiple_omega()
                else:
                    raise err_numint
        else:
            raise err

    eng_tot = 0
    if numint is None:
        numint = dft.numint.NumInt()
    # exx part
    xc_exx = xc_list.extract_by_xctype(XCType.EXX)
    xc_non_exx = xc_list.remove(xc_exx, inplace=False)
    log.info(f"[INFO] XCList extracted by process_energy_low_rung (exx): {xc_exx.token}")
    log.info(f"[INFO] XCList extracted by process_energy_low_rung (non_exx): {xc_non_exx.token}")
    xc_exx_extracted, eng_exx = _process_energy_exx(mf_dh, xc_exx)
    eng_tot += eng_exx
    assert len(xc_exx_extracted) == 0
    # dft part with additional exx
    if not (hasattr(numint, "custom") and numint.custom):
        omega, alpha, hyb = numint.rsh_and_hybrid_coeff(xc_non_exx.token)
    else:
        log.info("[INFO] Custom numint detected. We assume exx has already evaluated and no hybrid parameters.")
        omega, alpha, hyb = 0, 0, 0
    hyb_token = ""
    if abs(hyb) > 1e-10:
        hyb_token += f"+{hyb}*HF"
    if abs(omega) > 1e-10:
        hyb_token += f"+{alpha - hyb}*LR_HF({omega})"
    if len(hyb_token) > 0:
        log.info(f"[INFO] Evaluate exx token {hyb_token} from functional {xc_non_exx.token}.")
        _, eng_add_exx = _process_energy_exx(mf_dh, XCList().build_from_token(hyb_token, code_scf=True))
        eng_tot += eng_add_exx
    # pure part
    grids = mf_scf.scf.grids
    rho = get_rho(mf_scf.mol, grids, mf_scf.scf.make_rdm1())
    results = mf_scf.get_energy_purexc([xc_non_exx.token], rho, grids.weights, mf_scf.restricted, numint=numint)
    update_results(mf_dh.results, results)
    eng_tot += results["eng_purexc_" + xc_non_exx.token]
    log.note(f"[RESULT] Energy of process_energy_low_rung (non_exx): {results['eng_purexc_' + xc_non_exx.token]}")
    log.note(f"[RESULT] Energy of process_energy_low_rung (total): {eng_tot}")
    return eng_tot


def driver_energy_dh(mf_dh, xc=None, force_evaluate=False):
    """ Driver of multiple exchange-correlation energy component evaluation.

    Parameters
    ----------
    mf_dh : DH
        Object of doubly hybrid (restricted).
    xc : XCList or str
        Token of exchange and correlation.
    """
    log = mf_dh.log

    # prepare xc tokens
    mf_dh.build()
    if xc is None:
        xc = mf_dh.xc.xc_eng
    elif isinstance(xc, str):
        xc = XCList(xc, code_scf=False)
    assert isinstance(xc, XCList)
    xc = xc.copy()
    xc_splitted = _process_xc_split(xc)
    for key, xc_list in xc_splitted.items():
        mf_dh.inherited[key] = (xc_list, [])

    result = dict()
    eng_tot = 0.
    # 1. general xc
    xc_low_rung = xc_splitted["low_rung"]
    if xc_low_rung == mf_dh.xc.xc_scf and not force_evaluate:
        # If low-rung of xc_eng and xc_scf shares the same xc formula,
        # then we just use the SCF energy to evaluate low-rung part of total doubly hybrid energy.
        log.info("[INFO] xc of SCF is the same to xc of energy in rung-low part. Add SCF energy to total energy.")
        eng_tot += mf_dh.scf.e_tot
    else:
        eng_low_rung = _process_energy_low_rung(mf_dh, xc_low_rung)
        eng_tot += eng_low_rung
        result.update(mf_dh.to_scf().get_energy_noxc(mf_dh.scf, mf_dh.scf.make_rdm1()))
        eng_tot += result["eng_noxc"]

    # 2. other correlation
    # 2.1 IEPA (may also resolve evaluation of MP2)
    xc_iepa = xc_splitted["iepa"]
    eng_iepa = _process_energy_iepa(mf_dh, xc_iepa)
    eng_tot += eng_iepa
    # 2.2 MP2
    xc_mp2 = xc_splitted["mp2"]
    eng_mp2 = _process_energy_mp2(mf_dh, xc_mp2)
    eng_tot += eng_mp2
    # 2.3 Ring CCD
    xc_ring_ccd = xc_splitted["ring_ccd"]
    eng_ring_ccd = _process_energy_ring_ccd(mf_dh, xc_ring_ccd)
    eng_tot += eng_ring_ccd
    # # 2.4 VV10
    # xc_extracted, eng_vdw = _process_energy_vdw(mf_dh, xc_extracted)
    # eng_tot += eng_vdw

    result[f"eng_dh_{xc.token}"] = eng_tot
    update_results(mf_dh.results, result)
    log.note(f"[RESULT] Energy of {xc.token}: {eng_tot:20.12f}")
    return eng_tot


class DH(EngBase):
    """ Doubly hybrid object for energy evaluation.

    Attributes
    ----------
    inherited : List[Tuple[EngBase, XCList]]
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

    def __init__(self, mf_or_mol, xc, flags=None, **kwargs):
        # cheat to generate someting by __init__ from base class
        mol = mf_or_mol if isinstance(mf_or_mol, gto.Mole) else mf_or_mol.mol
        super().__init__(scf.HF(mol))
        self.inherited = dict()  # type: dict[str, tuple[XCList, list[EngBase]]]
        self.xc = NotImplemented  # type: XCDH
        self.log = NotImplemented  # type: lib.logger.Logger
        self.flags = flags if flags is not None else dict()
        self.flags.update(kwargs)
        self.instantiate(mf_or_mol, xc, self.flags)

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
        if isinstance(xc, (str, tuple)):
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
        do_jk = route_scf.lower().startswith("ri")
        do_only_dfj = route_scf.lower().replace("-", "").replace("_", "") in ["rijonx", "rij"]
        if do_jk:
            auxbasis_jk = flags.get("auxbasis_jk", None)
            if auxbasis_jk is None:
                auxbasis_jk = df.aug_etb(mol) if etb_first else df.make_auxbasis(mol, mp2fit=False)
                self.log.info("[INFO] Generate auxbasis_jk ...")
                self.log.info(str(auxbasis_jk))
        else:
            auxbasis_jk = None
        mf = custom_mf(mf, xc_scf, auxbasis_or_with_df=auxbasis_jk)
        if do_jk:
            mf.only_dfj = do_only_dfj
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

    def build_scf(self, **kwargs):
        """ Build SCF object ``mf._scf``.

        SCF object will be built upon molecule object and passed flags. Pre-instantiated SCF object will be discarded.

        Passed optional parameters will be written to ``self.flags``.
        """
        self.flags.update(kwargs)
        self.instantiate(self.mol, self.xc, self.flags)
        return self

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

    def kernel(self, xc=None, **kwargs):

        if xc is None:
            xc = self.xc.xc_eng
        elif isinstance(xc, str):
            xc = XCList(xc, code_scf=False)
        assert isinstance(xc, XCList)

        self.flags.update(kwargs)
        force_evaluate = self.flags.get("force_evaluate", False)
        eng_tot = driver_energy_dh(self, xc, force_evaluate)
        return eng_tot

    @property
    def e_tot(self):
        return self.results[f"eng_dh_{self.xc.xc_eng.token}"]

    def to_scf(self, **kwargs):
        # import
        if self.restricted:
            from pyscf.dh import RHDFT as HDFT
        else:
            from pyscf.dh import UHDFT as HDFT

        # generate instance
        mf = HDFT.from_rdh(self, self.scf, self.xc.xc_scf, **kwargs)

        return mf

    def to_mp2(self, **kwargs):
        # import
        if self.restricted:
            from pyscf.dh import RMP2Conv as MP2Conv
            from pyscf.dh.energy.mp2 import RMP2RI as MP2RI
        else:
            from pyscf.dh import UMP2Conv as MP2Conv
            from pyscf.dh import UMP2RI as MP2RI

        # configurations
        route_mp2 = self.flags.get("route_mp2", CONFIG_route_mp2)
        incore_t_oovv_mp2 = self.flags.get("incore_t_oovv_mp2", NotImplemented)

        # generate instance
        if route_mp2.lower().startswith("ri"):
            mf = MP2RI.from_rdh(self, self.scf, **kwargs)
        elif route_mp2.lower().startswith("conv"):
            mf = MP2Conv.from_rdh(self, self.scf, **kwargs)
        else:
            assert False, "Not recognized route_mp2."

        # fill configurations
        if incore_t_oovv_mp2 is not NotImplemented:
            mf.incore_t_oovv_mp2 = incore_t_oovv_mp2

        return mf

    def to_iepa(self, **kwargs):
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
            mf = IEPARI.from_rdh(self, self.scf, **kwargs)
        elif route_iepa.lower().startswith("conv"):
            mf = IEPAConv.from_rdh(self, self.scf, **kwargs)
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

    def to_ring_ccd(self, **kwargs):
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
        mf = RingCCDConv.from_rdh(self, self.scf, **kwargs)

        # fill configurations
        if tol_eng_ring_ccd is not NotImplemented:
            mf.conv_tol = tol_eng_ring_ccd
        if tol_amp_ring_ccd is not NotImplemented:
            mf.conv_tol_amp = tol_amp_ring_ccd
        if max_cycle_ring_ccd is not NotImplemented:
            mf.max_cycle = max_cycle_ring_ccd

        return mf


RDH = DH
UDH = DH  # currently workflow of unrestricted is exactly the same to restricted


if __name__ == '__main__':
    def main_1():
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5").build()
        mf = DH(mol, xc="XYG3", flags={"etb_first": True, "frozen": "FreezeNobleGasCore"}).build()
        print(mf.scf.e_tot)
        print(dft.RKS(mol, xc="B3LYPg").density_fit().run().e_tot)
        print(mf.to_mp2().run().results)
        print(mf.to_iepa().run().results)
        print(mf.to_ring_ccd().run().results)
        print(mf.run().e_tot)

    main_1()


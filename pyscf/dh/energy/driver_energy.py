from pyscf import dft
from pyscf.dh import util
from pyscf.dh.energy.rdft import get_rho, numint_customized
from pyscf.dh.util import XCType, XCList
import typing

if typing.TYPE_CHECKING:
    from pyscf.dh import RDH


def _process_energy_exx(mf_dh: "RDH", xc_list: XCList, force_evaluate=False):
    log = mf_dh.log
    xc_exx = xc_list.extract_by_xctype(XCType.EXX)
    if len(xc_exx) == 0:
        return xc_list, 0
    xc_extracted = xc_list.remove(xc_exx, inplace=False)
    log.info(f"[INFO] XCList extracted by process_energy_exx: {xc_exx.token}")
    log.info(f"[INFO] XCList remains   by process_energy_exx: {xc_extracted.token}")
    result = dict()
    eng_tot = 0
    for info in xc_exx:
        log.info(f"[INFO] EXX to be evaluated: {info.token}")
        # determine omega
        omega = 0
        if info.name == "LR_HF":
            assert len(info.parameters) == 1
            omega = info.parameters[0]
        elif info.name == "HF":
            omega = 0
        else:
            assert False, "Only accept LR_HF or HF, no SR_HF or anything else like second-order exchange, etc"
        name_eng_exx_HF = util.pad_omega("eng_exx_HF", omega)
        if force_evaluate or name_eng_exx_HF not in mf_dh.params.results:
            log.info(f"[INFO] Evaluate {name_eng_exx_HF}")
            result.update(mf_dh.get_energy_exactx(mf_dh.scf, mf_dh.make_rdm1_scf(), omega=omega))
            eng = result[name_eng_exx_HF]
        else:
            log.info(f"[INFO] {name_eng_exx_HF} is evaluated. Take previous evaluated value.")
            eng = mf_dh.params.results[name_eng_exx_HF]
        eng = info.fac * eng
        log.note(f"[RESULT] Energy of exchange {info.token}: {eng:20.12f}")
        eng_tot += eng
    mf_dh.params.update_results(result)
    return xc_extracted, eng_tot


def _process_energy_iepa(mf_dh: "RDH", xc_list: XCList, force_evaluate=False):
    log = mf_dh.log
    xc_iepa = xc_list.extract_by_xctype(XCType.IEPA)
    if len(xc_iepa) == 0:
        return xc_list, 0

    # run MP2 while in IEPA if found
    xc_iepa = xc_list.extract_by_xctype(XCType.IEPA | XCType.MP2)
    xc_extracted = xc_list.remove(xc_iepa, inplace=False)
    log.info(f"[INFO] XCList extracted by process_energy_iepa: {xc_iepa.token}")
    log.info(f"[INFO] XCList remains   by process_energy_iepa: {xc_extracted.token}")

    # prepare IEPA
    iepa_schemes = [info.name for info in xc_iepa]
    log.info(f"[INFO] Detected IEPAs: {iepa_schemes}")

    def comput_iepa():
        eng_tot = 0
        results = mf_dh.params.results
        for info in xc_iepa:
            eng = info.fac * (
                + info.parameters[0] * results[f"eng_corr_{info.name}_OS"]
                + info.parameters[1] * results[f"eng_corr_{info.name}_SS"])
            log.note(f"[RESULT] Energy of correlation {info.token}: {eng:20.12f}")
            eng_tot += eng
        return eng_tot

    def force_comput_iepa():
        mf_iepa = mf_dh.to_iepa()
        mf_iepa.kernel(iepa_schemes=iepa_schemes)
        eng_tot = comput_iepa()
        mf_dh.inherited.append((mf_iepa, xc_iepa))
        return eng_tot

    if force_evaluate:
        eng_tot = force_comput_iepa()
    else:
        try:
            eng_tot = comput_iepa()
        except KeyError:
            eng_tot = force_comput_iepa()

    return xc_extracted, eng_tot


def _process_energy_mp2(mf_dh: "RDH", xc_list: XCList, force_evaluate=False):
    log = mf_dh.log
    xc_mp2 = xc_list.extract_by_xctype(XCType.MP2 | XCType.RSMP2)
    if len(xc_mp2) == 0:
        return xc_list, 0
    xc_extracted = xc_list.remove(xc_mp2, inplace=False)
    log.info(f"[INFO] XCList extracted by process_energy_mp2: {xc_mp2.token}")
    log.info(f"[INFO] XCList remains   by process_energy_mp2: {xc_extracted.token}")
    log.info("[INFO] MP2 detected")
    # generate omega list
    # parameter of RSMP2: omega, c_os, c_ss
    omega_list = []
    for info in xc_mp2:
        if XCType.MP2 in info.type:
            omega_list.append(0)
        else:
            assert XCType.RSMP2 in info.type
            omega_list.append(info.parameters[0])
    assert len(set(omega_list)) == len(omega_list)

    def comput_mp2():
        eng_tot = 0
        results = mf_dh.params.results
        for info in xc_mp2:
            if XCType.MP2 in info.type:
                c_os, c_ss = info.parameters
                omega = 0
            else:
                omega, c_os, c_ss = info.parameters
            eng = info.fac * (
                + c_os * results[util.pad_omega("eng_corr_MP2_OS", omega)]
                + c_ss * results[util.pad_omega("eng_corr_MP2_SS", omega)])
            log.note(f"[RESULT] Energy of correlation {info.token}: {eng:20.12f}")
            eng_tot += eng
        return eng_tot

    def force_comput_mp2():
        mf_mp2 = mf_dh.to_mp2()
        mf_mp2.kernel(omega_list_mp2=omega_list)
        eng_tot = comput_mp2()
        mf_dh.inherited.append((mf_mp2, xc_mp2))
        return eng_tot

    if force_evaluate:
        eng_tot = force_comput_mp2()
    else:
        try:
            eng_tot = comput_mp2()
        except KeyError:
            eng_tot = force_comput_mp2()

    return xc_extracted, eng_tot


def _process_energy_drpa(mf_dh: "RDH", xc_list: XCList, force_evaluate=False):
    log = mf_dh.log
    xc_ring_ccd = xc_list.extract_by_xctype(XCType.RS_RING_CCD)
    xc_extracted = xc_list.remove(xc_ring_ccd, inplace=False)
    if len(xc_ring_ccd) == 0:
        return xc_list, 0
    log.info(f"[INFO] XCList extracted by process_energy_drpa (RS_RING_CCD): {xc_ring_ccd.token}")
    log.info(f"[INFO] XCList remains   by process_energy_drpa (RS_RING_CCD): {xc_extracted.token}")
    log.info(f"[INFO] Ring-CCD detected")

    # generate omega list
    # parameter of RSMP2: omega, c_os, c_ss
    omega_list = []
    for xc_info in xc_ring_ccd:
        omega_list.append(xc_info.parameters[0])

    def comput_ring_ccd():
        eng_tot = 0
        for xc_info in xc_ring_ccd:
            omega, c_os, c_ss = xc_info.parameters
            eng = xc_info.fac * (
                + c_os * mf_dh.params.results[util.pad_omega("eng_corr_RING_CCD_OS", omega)]
                + c_ss * mf_dh.params.results[util.pad_omega("eng_corr_RING_CCD_SS", omega)])
            log.note(f"[RESULT] Energy of correlation {xc_info.token}: {eng:20.12f}")
            eng_tot += eng
        return eng_tot

    def force_comput_ring_ccd():
        mf_ring_ccd = mf_dh.to_ring_ccd()
        mf_ring_ccd.kernel(omega_list_ring_ccd=omega_list)
        eng_tot = comput_ring_ccd()
        mf_dh.inherited.append((mf_ring_ccd, xc_ring_ccd))
        return eng_tot

    if force_evaluate:
        eng_tot = force_comput_ring_ccd()
    else:
        try:
            eng_tot = comput_ring_ccd()
        except KeyError:
            eng_tot = force_comput_ring_ccd()

    return xc_extracted, eng_tot


def _process_energy_vdw(mf_dh: "RDH", xc_list: XCList):
    log = mf_dh.log
    xc_vdw = xc_list.extract_by_xctype(XCType.VDW)
    if len(xc_vdw) == 0:
        return xc_list, 0
    xc_extracted = xc_list.remove(xc_vdw, inplace=False)
    log.info("[INFO] XCList extracted by process_energy_vdw: {:}".format(xc_vdw.token))
    log.info("[INFO] XCList remains   by process_energy_vdw: {:}".format(xc_extracted.token))
    eng_tot = 0
    result = dict()
    for info in xc_vdw:
        # currently only implement VV10
        if info.name == "VV10":
            nlc_pars = info.parameters
            assert len(nlc_pars) == 2
            grids = mf_dh.scf.grids
            nlcgrids = mf_dh.scf.nlcgrids
            res = mf_dh.get_energy_vv10(
                mf_dh.mol, mf_dh.make_rdm1_scf(), nlc_pars, grids, nlcgrids,
                verbose=mf_dh.verbose)
            result.update(res)
            eng = info.fac * result["eng_VV10({:}; {:})".format(*nlc_pars)]
            log.info("[RESULT] energy of VV10({:}; {:}): {:20.12f}".format(*nlc_pars, eng))
            eng_tot += eng
        else:
            raise NotImplementedError("Currently VDW only accepts VV10!")
    mf_dh.params.update_results(result)
    return xc_extracted, eng_tot


def _process_energy_low_rung(mf_dh: "RDH", xc_list: XCList, xc_to_parse: XCList = None):
    # logic of this function
    # 1. first try the whole low_rung token
    # 2. if low_rung failed, split pyscf_parsable and other terms
    # 2.1 try if customized numint is possible (this makes evaluation of energy faster)
    # 2.2 pyscf_parsable may fail when different omegas occurs
    #     split exx and pure dft evaluation (TODO: how rsh pure functionals to be evaluated?)
    # 2.3 even if this fails, split evaluate components term by term (XCInfo by XCInfo)
    log = mf_dh.log
    ni = dft.numint.NumInt()
    if xc_to_parse is None:
        xc_to_parse = xc_list.extract_by_xctype(XCType.RUNG_LOW)
    xc_extracted = xc_list.remove(xc_to_parse, inplace=False)
    log.info(f"[INFO] XCList to be parsed in process_energy_low_rung: {xc_to_parse.token}")

    if len(xc_to_parse) == 0:
        return xc_list, 0

    def handle_multiple_omega():
        log.warn(
            "Low-rung DFT has different values of omega found for RSH functional.\n"
            "We evaluate EXX and pure DFT in separate.\n"
            "This may cause problem that if some pure DFT components is omega-dependent "
            "and our program currently could not handle it.\n"
            "Anyway, this is purely experimental and use with caution.")
        eng_tot = 0
        xc_exx = xc_to_parse.extract_by_xctype(XCType.EXX)
        xc_non_exx = xc_to_parse.remove(xc_exx, inplace=False)
        log.info(f"[INFO] XCList extracted by process_energy_low_rung (handle_multiple_omega, "
                 f"exx): {xc_exx.token}")
        log.info(f"[INFO] XCList extracted by process_energy_low_rung (handle_multiple_omega, "
                 f"non_exx): {xc_non_exx.token}")
        if len(xc_exx) != 0:
            xc_exx_extracted, eng_exx = _process_energy_exx(mf_dh, xc_exx)
            xc_non_exx_extracted, eng_non_exx = _process_energy_low_rung(mf_dh, xc_non_exx)
            assert len(xc_exx_extracted) == 0
            if len(xc_non_exx_extracted) != 0:
                raise RuntimeError(
                    f"Finally left some xc not parsed: {xc_non_exx_extracted.token}. "
                    f"This is probably bug.")
            eng_tot += eng_exx + eng_non_exx
            log.note(f"[RESULT] Energy of process_energy_low_rung (handle_multiple_omega): {eng_tot}")
            return xc_extracted, eng_tot
        else:
            log.warn(
                "Seems pure-DFT part has different values of omega.\n"
                "This may caused by different hybrids with different omegas.\n"
                "We evaluate each term one-by-one.")
            xc_non_exx_list = [XCList().build_from_list([xc_info]) for xc_info in xc_non_exx]
            for xc_non_exx_item in xc_non_exx_list:
                xc_item_extracted, eng_item = _process_energy_low_rung(mf_dh, xc_non_exx_item)
                assert len(xc_item_extracted) == 0
                eng_tot += eng_item
            log.note(f"[RESULT] Energy of process_energy_low_rung (handle_multiple_omega): {eng_tot}")
            return xc_extracted, eng_tot

    numint = None
    # perform the logic of function
    try:
        ni._xc_type(xc_to_parse.token)
    except (ValueError, KeyError) as err:
        if isinstance(err, ValueError) and "Different values of omega" in err.args[0]:
            return handle_multiple_omega()
        elif isinstance(err, KeyError) and "Unknown" in err.args[0] \
                or isinstance(err, ValueError) and "too many values to unpack" in err.args[0]:
            log.info("[INFO] Unknown functional to PySCF. Try build custom numint.")
            try:
                numint = numint_customized(xc_to_parse, mf_dh.params.flags)
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
    xc_exx = xc_to_parse.extract_by_xctype(XCType.EXX)
    xc_non_exx = xc_to_parse.remove(xc_exx, inplace=False)
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
    grids = mf_dh.scf.grids
    rho = get_rho(mf_dh.mol, grids, mf_dh.make_rdm1_scf())
    result = mf_dh.get_energy_purexc([xc_non_exx.token], rho, grids.weights, mf_dh.restricted, numint=numint)
    mf_dh.params.update_results(result)
    eng_tot += result["eng_purexc_" + xc_non_exx.token]
    log.note(f"[RESULT] Energy of process_energy_low_rung (non_exx): {result['eng_purexc_' + xc_non_exx.token]}")
    log.note(f"[RESULT] Energy of process_energy_low_rung (total): {eng_tot}")
    return xc_extracted, eng_tot


def driver_energy_dh(mf_dh, xc=None):
    """ Driver of multiple exchange-correlation energy component evaluation.

    Parameters
    ----------
    mf_dh : RDH
        Object of doubly hybrid (restricted).
    xc : XCList or str
        Token of exchange and correlation.
    """
    log = mf_dh.log
    mf_dh.build()
    if xc is None:
        xc = mf_dh.xc.xc_eng
    elif isinstance(xc, str):
        xc = XCList(xc, code_scf=False)
    assert isinstance(xc, XCList)
    xc = xc.copy()
    result = dict()
    eng_tot = 0.
    # 1. general xc
    xc_extracted = xc.copy()
    xc_low_rung = xc_extracted.extract_by_xctype(XCType.RUNG_LOW)
    xc_extracted = xc_extracted.remove(xc_low_rung)
    if xc_low_rung == mf_dh.xc.xc_scf and not mf_dh.params.flags["force_eng_low_rung_revaluate"]:
        # If low-rung of xc_eng and xc_scf shares the same xc formula,
        # then we just use the SCF energy to evaluate low-rung part of total doubly hybrid energy.
        log.info("[INFO] xc of SCF is the same to xc of energy in rung-low part. Add SCF energy to total energy.")
        eng_tot += mf_dh.scf.e_tot
    else:
        xc_low_rung_extracted, eng_low_rung = _process_energy_low_rung(mf_dh, xc_low_rung)
        assert len(xc_low_rung_extracted) == 0
        eng_tot += eng_low_rung
        result.update(mf_dh.get_energy_noxc(mf_dh.scf, mf_dh.make_rdm1_scf()))
        eng_tot += result["eng_noxc"]

    # 2. other correlation
    # 2.1 IEPA (may also resolve evaluation of MP2)
    xc_extracted, eng_iepa = _process_energy_iepa(mf_dh, xc_extracted)
    eng_tot += eng_iepa
    # 2.2 MP2
    xc_extracted, eng_mp2 = _process_energy_mp2(mf_dh, xc_extracted)
    eng_tot += eng_mp2
    # 2.2 MP2
    xc_extracted, eng_drpa = _process_energy_drpa(mf_dh, xc_extracted)
    eng_tot += eng_drpa
    # 2.3 VV10
    xc_extracted, eng_vdw = _process_energy_vdw(mf_dh, xc_extracted)
    eng_tot += eng_vdw
    # finalize
    if len(xc_extracted) > 0:
        raise RuntimeError("Some xc terms not evaluated! Possibly bug of program.")
    result[f"eng_dh_{xc.token}"] = eng_tot
    log.note(f"[RESULT] Energy of {xc.token}: {eng_tot:20.12f}")
    return result

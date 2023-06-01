import itertools

from pyscf import dft
import numpy as np


def eval_xc_eff_ssr_generator(name_code, name_fr, name_sr, omega=0.7, cutoff=1e-13):
    """ Generator for function of scaled short-range DFT integrals.

    Parameters
    ----------
    name_code : str
        xc functional to be scaled.
    name_fr : str
        Full-range functional of the scale. Must be LDA.
    name_sr : str
        Short-range functional of the scale. Must be LDA.
    omega : float
        Range-separate parameter.
    cutoff : float
        Cutoff of exc_fr. If grid value of exc_fr is too small, then the returned exc on this grid will set to be zero.

    Returns
    -------
    callable
        A function shares the same function signature with dft.numint.NumInt.eval_xc_eff.
    """

    # original numint object
    ni = dft.numint.NumInt()
    # currently only LDA is supported
    assert ni._xc_type(name_fr) == "LDA"
    assert ni._xc_type(name_sr) == "LDA"
    # the xc type concern in this scaled short-range term
    xc_type_code = ni._xc_type(name_code)

    def eval_xc_eff(numint, xc_code, rho, xctype=None, *args, **kwargs):
        # for unrestricted methods, input rho can be tuple
        rho = np.asarray(rho)

        if xctype is None:
            xctype = numint._xc_type(xc_code)
        if xctype == 'LDA':
            spin_polarized = rho.ndim >= 2
        else:
            spin_polarized = rho.ndim == 3

        if not spin_polarized:
            return eval_xc_eff_spin_nonpolarized(numint, xc_code, rho, xctype=xctype, *args, **kwargs)
        else:
            return eval_xc_eff_spin_polarized(numint, xc_code, rho, xctype=xctype, *args, **kwargs)

    def eval_xc_eff_spin_nonpolarized(numint, xc_code, rho, deriv=1, omega=omega, xctype=None, *_args, **_kwargs):
        # the xc type of the whole numint object
        if xctype is None:
            xctype = numint._xc_type(xc_code)
        if xctype == "LDA":
            assert rho.ndim == 1
            rho0 = rho.copy()
        else:
            assert rho.ndim == 2
            rho0 = rho[0].copy()
        # if the xc type concern is LDA, then only extract the density grid (instead of its derivatives)
        if xc_type_code == "LDA":
            rho = rho0
        # evaluate xc grids by original numint object
        exc_code, vxc_code, fxc_code, kxc_code = ni.eval_xc_eff(name_code, rho, deriv=deriv)
        exc_fr, vxc_fr, fxc_fr, kxc_fr = ni.eval_xc_eff(name_fr, rho0, deriv=deriv)
        exc_sr, vxc_sr, fxc_sr, kxc_sr = ni.eval_xc_eff(name_sr, rho0, deriv=deriv, omega=omega)
        # avoid too small denominator (must set grid values to zero on these masks)
        mask = abs(exc_fr) < cutoff
        exc_fr[mask] = cutoff
        rho0[mask] = cutoff
        # handle exc, vxc, fxc, kxc
        ratio = exc_sr / exc_fr
        exc = vxc = fxc = kxc = None
        if deriv >= 0:
            exc = exc_code * ratio
            exc[mask] = 0
        if deriv >= 1:
            vxc = vxc_code.copy()
            vxc[0] = (
                + vxc_code[0] * exc_sr / exc_fr
                + exc_code * vxc_sr[0] / exc_fr
                - exc_code * exc_sr * vxc_fr[0] / exc_fr ** 2
            )
            vxc[1:] *= ratio
            vxc[:, mask] = 0
        if deriv >= 2:
            # derivative of (c * s / f), c -> code, s -> short-range, f -> full-range
            fxc = fxc_code.copy()
            r = rho0
            c, dc, ddc = exc_code * r, vxc_code[0], fxc_code[0, 0]
            pc, dpc = vxc_code[1:], fxc_code[0, 1:]
            s, ds, dds = exc_sr * r, vxc_sr[0], fxc_sr[0, 0]
            f, df, ddf = exc_fr * r, vxc_fr[0], fxc_fr[0, 0]
            fxc *= ratio
            fxc[0, 0] = (
                + (c * dds + 2 * dc * ds + ddc * s) / f
                - (c * s * ddf + 2 * c * ds * df + 2 * dc * s * df) / f**2
                + 2 * c * s * df**2 / f**3
            )
            fxc[0, 1:] = (
                + dpc * s / f
                + pc * ds / f
                - pc * s * df / f**2
            )
            fxc[1:, 0] = fxc[0, 1:]
            fxc[:, :, mask] = 0
        if deriv >= 3:
            # derivative of (c * s / f), c -> code, s -> short-range, f -> full-range
            kxc = kxc_code.copy()
            r = rho0
            c, dc, ddc, dddc = exc_code * r, vxc_code[0], fxc_code[0, 0], kxc_code[0, 0, 0]
            pc, dpc, ddpc = vxc_code[1:], fxc_code[0, 1:], kxc_code[0, 0, 1:]
            ppc, dppc = fxc_code[1:, 1:], kxc_code[0, 1:, 1:]
            s, ds, dds, ddds = exc_sr * r, vxc_sr[0], fxc_sr[0, 0], kxc_sr[0, 0, 0]
            f, df, ddf, dddf = exc_fr * r, vxc_fr[0], fxc_fr[0, 0], kxc_fr[0, 0, 0]

            kxc *= ratio
            kxc[0, 0, 0] = (
                + (c * ddds + 3 * dc * dds + 3 * ddc * ds + dddc * s) / f
                - (
                    + c * s * dddf + 3 * c * ds * ddf + 3 * c * dds * df + 3 * dc * s * ddf
                    + 6 * dc * ds * df + 3 * ddc * s * df) / f**2
                + 6 * (c * ds * df**2 + dc * s * df**2 + c * s * df * ddf) / f**3
                - 6 * c * s * df**3 / f**4
            )
            kxc[0, 0, 1:] = kxc[0, 1:, 0] = kxc[1:, 0, 0] = (
                + (pc * dds + 2 * dpc * ds + ddpc * s) / f
                - (pc * s * ddf + 2 * pc * ds * df + 2 * dpc * s * df) / f**2
                + 2 * pc * s * df**2 / f**3
            )
            kxc[0, 1:, 1:] = kxc[1:, 0, 1:] = kxc[1:, 1:, 0] = (
                + dppc * s / f
                + ppc * ds / f
                - ppc * s * df / f**2
            )
            kxc[:, :, :, mask] = 0
        return exc, vxc, fxc, kxc

    def eval_xc_eff_spin_polarized(numint, xc_code, rho, deriv=1, omega=omega, xctype=None, *_args, **_kwargs):
        # the xc type of the whole numint object
        if xctype is None:
            xctype = numint._xc_type(xc_code)
        if xctype == "LDA":
            assert rho.ndim == 2
            rho0 = rho.copy()
        else:
            assert rho.ndim == 3
            rho0 = rho[:, 0].copy()
        # if the xc type concern is LDA, then only extract the density grid (instead of its derivatives)
        if xc_type_code == "LDA":
            rho = rho0
        # evaluate xc grids by original numint object
        exc_code, vxc_code, fxc_code, kxc_code = ni.eval_xc_eff(name_code, rho, deriv=deriv)
        exc_fr, vxc_fr, fxc_fr, kxc_fr = ni.eval_xc_eff(name_fr, rho0, deriv=deriv)
        exc_sr, vxc_sr, fxc_sr, kxc_sr = ni.eval_xc_eff(name_sr, rho0, deriv=deriv, omega=omega)
        # avoid too small denominator (must set grid values to zero on these masks)
        mask = abs(exc_fr) < cutoff
        exc_fr[mask] = cutoff
        rho0[:, mask] = cutoff
        # handle exc, vxc, fxc, kxc
        ratio = exc_sr / exc_fr
        exc = vxc = fxc = kxc = None
        if deriv >= 0:
            exc = exc_code * ratio
            exc[mask] = 0
        if deriv >= 1:
            vxc = vxc_code.copy()
            vxc[:, 0] = (
                + vxc_code[:, 0] * exc_sr / exc_fr
                + exc_code * vxc_sr[:, 0] / exc_fr
                - exc_code * exc_sr * vxc_fr[:, 0] / exc_fr ** 2
            )
            vxc[:, 1:] *= ratio
            vxc[:, :, mask] = 0
        if deriv >= 2:
            # derivative of (c * s / f), c -> code, s -> short-range, f -> full-range
            fxc = fxc_code.copy()
            r = rho0.sum(axis=0)
            c, dc, ddc = exc_code * r, vxc_code[:, 0], fxc_code[:, 0, :, 0]
            pc, dpc = vxc_code[:, 1:], fxc_code[:, 0, :, 1:]
            s, ds, dds = exc_sr * r, vxc_sr[:, 0], fxc_sr[:, 0, :, 0]
            f, df, ddf = exc_fr * r, vxc_fr[:, 0], fxc_fr[:, 0, :, 0]
            fxc *= ratio
            for s1, s2 in itertools.product((0, 1), (0, 1)):
                # handle actual derivatives, iterate by spin
                fxc[s1, 0, s2, 0] = (
                    + (c * dds[s1, s2] + dc[s1] * ds[s2] + dc[s2] * ds[s1] + ddc[s1, s2] * s) / f
                    - (
                        + c * s * ddf[s1, s2]
                        + c * ds[s1] * df[s2] + c * ds[s2] * df[s1]
                        + dc[s1] * s * df[s2] + dc[s2] * s * df[s1]
                    ) / f**2
                    + 2 * c * s * df[s1] * df[s2] / f**3
                )
                fxc[s1, 0, s2, 1:] = fxc[s2, 1:, s1, 0] = (
                    + dpc[s1, s2] * s / f
                    + pc[s2] * ds[s1] / f
                    - pc[s2] * s * df[s1] / f**2
                )
            fxc[:, :, :, :, mask] = 0
        if deriv >= 3:
            # derivative of (c * s / f), c -> code, s -> short-range, f -> full-range
            kxc = kxc_code.copy()
            r = rho0.sum(axis=0)
            c, dc, ddc, dddc = exc_code * r, vxc_code[:, 0], fxc_code[:, 0, :, 0], kxc_code[:, 0, :, 0, :, 0]
            pc, dpc, ddpc = vxc_code[:, 1:], fxc_code[:, 0, :, 1:], kxc_code[:, 0, :, 0, :, 1:]
            ppc, dppc = fxc_code[:, 1:, :, 1:].swapaxes(1, 2), kxc_code[:, 0, :, 1:, :, 1:].swapaxes(2, 3)
            s, ds, dds, ddds = exc_sr * r, vxc_sr[:, 0], fxc_sr[:, 0, :, 0], kxc_sr[:, 0, :, 0, :, 0]
            f, df, ddf, dddf = exc_fr * r, vxc_fr[:, 0], fxc_fr[:, 0, :, 0], kxc_fr[:, 0, :, 0, :, 0]

            kxc *= ratio

            for s1, s2, s3 in itertools.product((0, 1), (0, 1), (0, 1)):
                kxc[s1, 0, s2, 0, s3, 0] = (
                    + (
                        + c * ddds[s1, s2, s3]
                        + dc[s1] * dds[s2, s3] + dc[s2] * dds[s3, s1] + dc[s3] * dds[s1, s2]
                        + ddc[s1, s2] * ds[s3] + ddc[s2, s3] * ds[s1] + ddc[s3, s1] * ds[s2]
                        + dddc[s1, s2, s3] * s) / f
                    - (
                        + c * s * dddf[s1, s2, s3]
                        + c * ds[s1] * ddf[s2, s3] + c * ds[s2] * ddf[s3, s1] + c * ds[s3] * ddf[s1, s2]
                        + c * dds[s1, s2] * df[s3] + c * dds[s2, s3] * df[s1] + c * dds[s3, s1] * df[s2]
                        + dc[s1] * s * ddf[s2, s3] + dc[s2] * s * ddf[s3, s1] + dc[s3] * s * ddf[s1, s2]
                        + dc[s1] * ds[s2] * df[s3] + dc[s1] * ds[s3] * df[s2] + dc[s2] * ds[s1] * df[s3]
                        + dc[s2] * ds[s3] * df[s1] + dc[s3] * ds[s1] * df[s2] + dc[s3] * ds[s2] * df[s1]
                        + ddc[s1, s2] * s * df[s3] + ddc[s2, s3] * s * df[s1] + ddc[s3, s1] * s * df[s2]) / f**2
                    + 2 * (
                        + c * (ds[s1] * df[s2] * df[s3] + ds[s2] * df[s3] * df[s1] + ds[s3] * df[s1] * df[s2])
                        + s * (dc[s1] * df[s2] * df[s3] + dc[s2] * df[s3] * df[s1] + dc[s3] * df[s1] * df[s2])
                        + c * s * (df[s1] * ddf[s2, s3] + df[s2] * ddf[s3, s1] + df[s3] * ddf[s1, s2])) / f**3
                    - 6 * c * s * df[s1] * df[s2] * df[s3] / f**4
                )
                kxc_p3 = (
                    + (pc[s3] * dds[s1, s2] + dpc[s1, s3] * ds[s2] + dpc[s2, s3] * ds[s1] + ddpc[s1, s2, s3] * s) / f
                    - (
                        + pc[s3] * s * ddf[s1, s2]
                        + pc[s3] * ds[s1] * df[s2] + pc[s3] * ds[s2] * df[s1]
                        + dpc[s1, s3] * s * df[s2] + dpc[s2, s3] * s * df[s1]) / f**2
                    + 2 * pc[s3] * s * df[s1] * df[s2] / f**3
                )
                kxc[s1, 0, s2, 0, s3, 1:] = kxc[s2, 0, s1, 0, s3, 1:] = kxc_p3
                kxc[s1, 0, s3, 1:, s2, 0] = kxc[s2, 0, s3, 1:, s1, 0] = kxc_p3
                kxc[s3, 1:, s1, 0, s2, 0] = kxc[s3, 1:, s2, 0, s1, 0] = kxc_p3
                kxc_p2p3 = (
                    + dppc[s1, s2, s3] * s / f
                    + ppc[s2, s3] * ds[s1] / f
                    - ppc[s2, s3] * s * df[s1] / f**2
                )
                kxc_p3p2 = kxc_p2p3.swapaxes(0, 1)
                kxc[s1, 0, s2, 1:, s3, 1:] = kxc[s2, 1:, s1, 0, s3, 1:] = kxc[s2, 1:, s3, 1:, s1, 0] = kxc_p2p3
                kxc[s1, 0, s3, 1:, s2, 1:] = kxc[s3, 1:, s1, 0, s2, 1:] = kxc[s3, 1:, s2, 1:, s1, 0] = kxc_p3p2
            kxc[:, :, :, :, :, :, mask] = 0
        return exc, vxc, fxc, kxc
    return eval_xc_eff

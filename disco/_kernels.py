import cupy as cp
from cupyx import jit

from disco.constants import RK45Coeffs


@jit.rawkernel()
def rhs_kernel(
    dydt_arr,
    y,
    t,
    paused,
    Bx_arr,
    By_arr,
    Bz_arr,
    B_arr,
    Ex_arr,
    Ey_arr,
    Ez_arr,
    dBdx_arr,
    dBdy_arr,
    dBdz_arr,
    dBxdy_arr,
    dBxdz_arr,
    dBydx_arr,
    dBydz_arr,
    dBzdx_arr,
    dBzdy_arr,
    x_axis,
    y_axis,
    z_axis,
    t_axis,
    nx,
    ny,
    nz,
    nt,
    r_inner,
):
    """[CUPY KERNEL] implements RHS of ODE.

    Code adapted from Fortran. Uses gyro-averaged equations of motion
    developed by Brizzard and Chan (Phys. Plasmas 6, 4553, 1999),

    Writes output to dydt[idx, :]
    """
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
    oob = False

    # Out of bounds check
    if idx < y.shape[0] and (not paused[idx]):
        oob = (
            y[idx, 0] < x_axis[0]
            or y[idx, 0] > x_axis[nx - 1]
            or y[idx, 1] < y_axis[0]
            or y[idx, 1] > y_axis[ny - 1]
            or y[idx, 2] < z_axis[0]
            or y[idx, 2] > z_axis[nz - 1]
            or t[idx] < t_axis[0]
            or t[idx] > t_axis[nt - 1]
            or (y[idx, 0] ** 2 + y[idx, 1] ** 2 + y[idx, 2] ** 2) ** 0.5 < r_inner[idx]
        )

    if idx < y.shape[0] and (not oob) and (not paused[idx]):
        # Pull variables out of arrays
        ppar = y[idx, 3]
        M = y[idx, 4]

        Bx = Bx_arr[idx]
        By = By_arr[idx]
        Bz = Bz_arr[idx]
        B = B_arr[idx]

        Ex = Ex_arr[idx]
        Ey = Ey_arr[idx]
        Ez = Ez_arr[idx]

        dBdx = dBdx_arr[idx]
        dBdy = dBdy_arr[idx]
        dBdz = dBdz_arr[idx]

        dBxdy = dBxdy_arr[idx]
        dBxdz = dBxdz_arr[idx]
        dBydx = dBydx_arr[idx]
        dBydz = dBydz_arr[idx]
        dBzdx = dBzdx_arr[idx]
        dBzdy = dBzdy_arr[idx]

        # gyro-averaged equations of motion developed by Brizzard and Chan (Phys.
        # Plasmas 6, 4553, 1999),
        gamma = cp.sqrt(1 + 2 * B * M + ppar**2)
        pparl_B = ppar / B
        pparl_B2 = pparl_B / B
        Bxstar = Bx + pparl_B * (dBzdy - dBydz) - pparl_B2 * (Bz * dBdy - By * dBdz)
        Bystar = By + pparl_B * (dBxdz - dBzdx) - pparl_B2 * (Bx * dBdz - Bz * dBdx)
        Bzstar = Bz + pparl_B * (dBydx - dBxdy) - pparl_B2 * (By * dBdx - Bx * dBdy)
        Bsparl = (Bx * Bxstar + By * Bystar + Bz * Bzstar) / B
        gamma_Bsparl = 1 / (gamma * Bsparl)
        pparl_gamma_Bsparl = ppar * gamma_Bsparl
        B_Bsparl = 1 / (B * Bsparl)
        M_gamma_Bsparl = M * gamma_Bsparl
        M_gamma_B_Bsparl = M_gamma_Bsparl / B

        # 	  ...now calculate dynamic quantities...
        dydt_arr[idx, 0] = (
            pparl_gamma_Bsparl * Bxstar  # curv drft + parl
            + M_gamma_B_Bsparl * (By * dBdz - Bz * dBdy)  # gradB drft
            + B_Bsparl * (Ey * Bz - Ez * By)  # ExB drft
        )
        dydt_arr[idx, 1] = (
            pparl_gamma_Bsparl * Bystar  # curv drft + parl
            + M_gamma_B_Bsparl * (Bz * dBdx - Bx * dBdz)  # gradB drft
            + B_Bsparl * (Ez * Bx - Ex * Bz)  # ExB drft
        )
        dydt_arr[idx, 2] = (
            pparl_gamma_Bsparl * Bzstar  # curv drft + parl
            + M_gamma_B_Bsparl * (Bx * dBdy - By * dBdx)  # gradB drft
            + B_Bsparl * (Ex * By - Ey * Bx)  # ExB drft
        )
        dydt_arr[idx, 3] = (
            Bxstar * Ex + Bystar * Ey + Bzstar * Ez
        ) / Bsparl - M_gamma_Bsparl * (  # parl force
            Bxstar * dBdx + Bystar * dBdy + Bzstar * dBdz
        )


@jit.rawkernel()
def multi_interp_kernel(
    nx,
    ny,
    nz,
    nt,
    nxy,
    nxyz,
    nttl,
    ix,
    iy,
    iz,
    it,
    t,
    y,
    paused,
    b0,
    r_inner,
    t_axis,
    x_axis,
    y_axis,
    z_axis,
    bx_vec,
    by_vec,
    bz_vec,
    ex_vec,
    ey_vec,
    ez_vec,
    extra_fields_vec,
    bx,
    by,
    bz,
    ex,
    ey,
    ez,
    dbxdx,
    dbxdy,
    dbxdz,
    dbydx,
    dbydy,
    dbydz,
    dbzdx,
    dbzdy,
    dbzdz,
    b,
    dbdx,
    dbdy,
    dbdz,
    extra_fields,
):
    """[CUPY KERNEL] Four dimensional interpolation with finite differencing
    and reused weights.

    Code adapted from fortran.
    """
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
    oob = False

    # Out of bounds check
    if idx < y.shape[0] and (not paused[idx]):
        oob = (
            y[idx, 0] < x_axis[0]
            or y[idx, 0] > x_axis[nx - 1]
            or y[idx, 1] < y_axis[0]
            or y[idx, 1] > y_axis[ny - 1]
            or y[idx, 2] < z_axis[0]
            or y[idx, 2] > z_axis[nz - 1]
            or t[idx] < t_axis[0]
            or t[idx] > t_axis[nt - 1]
            or (y[idx, 0] ** 2 + y[idx, 1] ** 2 + y[idx, 2] ** 2) ** 0.5 < r_inner[idx]
        )

    # Main body of the kernel
    if idx < y.shape[0] and (not oob) and (not paused[idx]):
        dx = x_axis[ix[idx] + 1] - x_axis[ix[idx]]
        dy = y_axis[iy[idx] + 1] - y_axis[iy[idx]]
        dz = z_axis[iz[idx] + 1] - z_axis[iz[idx]]
        dt = t_axis[it[idx] + 1] - t_axis[it[idx]]

        # ...determine memory location corresponding to ix,iy,iz,it
        # 	and ix+1,iy+1,iz+1,it+1...
        jjjj = ix[idx] + iy[idx] * nx + iz[idx] * nxy + it[idx] * nxyz + 1
        ijjj = jjjj - 1
        jijj = jjjj - nx
        iijj = jijj - 1
        jjij = jjjj - nxy
        ijij = jjij - 1
        jiij = jijj - nxy
        iiij = jiij - 1
        jjji = jjjj - nxyz
        ijji = ijjj - nxyz
        jiji = jijj - nxyz
        iiji = iijj - nxyz
        jjii = jjij - nxyz
        ijii = ijij - nxyz
        jiii = jiij - nxyz
        iiii = iiij - nxyz

        # ...calculate weighting factors...
        w1 = cp.abs(y[idx, 0] - x_axis[ix[idx]]) / dx
        w1m = 1.0 - w1
        w2 = cp.abs(y[idx, 1] - y_axis[iy[idx]]) / dy
        w2m = 1.0 - w2
        w3 = cp.abs(y[idx, 2] - z_axis[iz[idx]]) / dz
        w3m = 1.0 - w3
        w4 = cp.abs(t[idx] - t_axis[it[idx]]) / dt
        w4m = 1.0 - w4

        w1m2m = w1m * w2m
        w12m = w1 * w2m
        w12 = w1 * w2
        w1m2 = w1m * w2

        w1m2m3m = w1m2m * w3m
        w12m3m = w12m * w3m
        w123m = w12 * w3m
        w1m23m = w1m2 * w3m

        w1m2m3 = w1m2m * w3
        w12m3 = w12m * w3
        w123 = w12 * w3
        w1m23 = w1m2 * w3

        ww01 = w1m2m3m * w4m
        ww02 = w12m3m * w4m
        ww03 = w123m * w4m
        ww04 = w1m23m * w4m
        ww05 = w1m2m3 * w4m
        ww06 = w12m3 * w4m
        ww07 = w123 * w4m
        ww08 = w1m23 * w4m
        ww09 = w1m2m3m * w4
        ww10 = w12m3m * w4
        ww11 = w123m * w4
        ww12 = w1m23m * w4
        ww13 = w1m2m3 * w4
        ww14 = w12m3 * w4
        ww15 = w123 * w4
        ww16 = w1m23 * w4

        # ..define some factors often repeated in the interpolations...
        r = (y[idx, 0] ** 2 + y[idx, 1] ** 2 + y[idx, 2] ** 2) ** (0.5)
        r2 = r * r
        bfac1 = 3.0 * b0[idx] / r2 / r2 / r
        bfac2 = 5.0 * bfac1 / r2

        # ...interpolate field components...
        bx[idx] = (
            bx_vec[iiii] * ww01
            + bx_vec[jiii] * ww02
            + bx_vec[jjii] * ww03
            + bx_vec[ijii] * ww04
            + bx_vec[iiji] * ww05
            + bx_vec[jiji] * ww06
            + bx_vec[jjji] * ww07
            + bx_vec[ijji] * ww08
            + bx_vec[iiij] * ww09
            + bx_vec[jiij] * ww10
            + bx_vec[jjij] * ww11
            + bx_vec[ijij] * ww12
            + bx_vec[iijj] * ww13
            + bx_vec[jijj] * ww14
            + bx_vec[jjjj] * ww15
            + bx_vec[ijjj] * ww16
            - bfac1 * y[idx, 0] * y[idx, 2]
        )

        by[idx] = (
            by_vec[iiii] * ww01
            + by_vec[jiii] * ww02
            + by_vec[jjii] * ww03
            + by_vec[ijii] * ww04
            + by_vec[iiji] * ww05
            + by_vec[jiji] * ww06
            + by_vec[jjji] * ww07
            + by_vec[ijji] * ww08
            + by_vec[iiij] * ww09
            + by_vec[jiij] * ww10
            + by_vec[jjij] * ww11
            + by_vec[ijij] * ww12
            + by_vec[iijj] * ww13
            + by_vec[jijj] * ww14
            + by_vec[jjjj] * ww15
            + by_vec[ijjj] * ww16
            - bfac1 * y[idx, 1] * y[idx, 2]
        )

        bz[idx] = (
            bz_vec[iiii] * ww01
            + bz_vec[jiii] * ww02
            + bz_vec[jjii] * ww03
            + bz_vec[ijii] * ww04
            + bz_vec[iiji] * ww05
            + bz_vec[jiji] * ww06
            + bz_vec[jjji] * ww07
            + bz_vec[ijji] * ww08
            + bz_vec[iiij] * ww09
            + bz_vec[jiij] * ww10
            + bz_vec[jjij] * ww11
            + bz_vec[ijij] * ww12
            + bz_vec[iijj] * ww13
            + bz_vec[jijj] * ww14
            + bz_vec[jjjj] * ww15
            + bz_vec[ijjj] * ww16
            - bfac1 * y[idx, 2] * y[idx, 2]
            + b0[idx] / r2 / r
        )

        ex[idx] = (
            ex_vec[iiii] * ww01
            + ex_vec[jiii] * ww02
            + ex_vec[jjii] * ww03
            + ex_vec[ijii] * ww04
            + ex_vec[iiji] * ww05
            + ex_vec[jiji] * ww06
            + ex_vec[jjji] * ww07
            + ex_vec[ijji] * ww08
            + ex_vec[iiij] * ww09
            + ex_vec[jiij] * ww10
            + ex_vec[jjij] * ww11
            + ex_vec[ijij] * ww12
            + ex_vec[iijj] * ww13
            + ex_vec[jijj] * ww14
            + ex_vec[jjjj] * ww15
            + ex_vec[ijjj] * ww16
        )

        ey[idx] = (
            ey_vec[iiii] * ww01
            + ey_vec[jiii] * ww02
            + ey_vec[jjii] * ww03
            + ey_vec[ijii] * ww04
            + ey_vec[iiji] * ww05
            + ey_vec[jiji] * ww06
            + ey_vec[jjji] * ww07
            + ey_vec[ijji] * ww08
            + ey_vec[iiij] * ww09
            + ey_vec[jiij] * ww10
            + ey_vec[jjij] * ww11
            + ey_vec[ijij] * ww12
            + ey_vec[iijj] * ww13
            + ey_vec[jijj] * ww14
            + ey_vec[jjjj] * ww15
            + ey_vec[ijjj] * ww16
        )

        ez[idx] = (
            ez_vec[iiii] * ww01
            + ez_vec[jiii] * ww02
            + ez_vec[jjii] * ww03
            + ez_vec[ijii] * ww04
            + ez_vec[iiji] * ww05
            + ez_vec[jiji] * ww06
            + ez_vec[jjji] * ww07
            + ez_vec[ijji] * ww08
            + ez_vec[iiij] * ww09
            + ez_vec[jiij] * ww10
            + ez_vec[jjij] * ww11
            + ez_vec[ijij] * ww12
            + ez_vec[iijj] * ww13
            + ez_vec[jijj] * ww14
            + ez_vec[jjjj] * ww15
            + ez_vec[ijjj] * ww16
        )

        # ...interpolate extra fields...
        for ifield in range(extra_fields_vec.shape[1]):
            extra_fields[idx, ifield] = (
                extra_fields_vec[iiii, ifield] * ww01
                + extra_fields_vec[jiii, ifield] * ww02
                + extra_fields_vec[jjii, ifield] * ww03
                + extra_fields_vec[ijii, ifield] * ww04
                + extra_fields_vec[iiji, ifield] * ww05
                + extra_fields_vec[jiji, ifield] * ww06
                + extra_fields_vec[jjji, ifield] * ww07
                + extra_fields_vec[ijji, ifield] * ww08
                + extra_fields_vec[iiij, ifield] * ww09
                + extra_fields_vec[jiij, ifield] * ww10
                + extra_fields_vec[jjij, ifield] * ww11
                + extra_fields_vec[ijij, ifield] * ww12
                + extra_fields_vec[iijj, ifield] * ww13
                + extra_fields_vec[jijj, ifield] * ww14
                + extra_fields_vec[jjjj, ifield] * ww15
                + extra_fields_vec[ijjj, ifield] * ww16
            )

        # ...calculate btot and field derivatives to 1st order...
        # ...first form more intermediate weights...
        w2m3m4m = w2m * w3m * w4m
        w23m4m = w2 * w3m * w4m
        w2m34m = w2m * w3 * w4m
        w234m = w2 * w3 * w4m
        w2m3m4 = w2m * w3m * w4
        w23m4 = w2 * w3m * w4
        w2m34 = w2m * w3 * w4
        w234 = w2 * w3 * w4

        w1m3m4m = w1m * w3m * w4m
        w13m4m = w1 * w3m * w4m
        w1m34m = w1m * w3 * w4m
        w134m = w1 * w3 * w4m
        w1m3m4 = w1m * w3m * w4
        w13m4 = w1 * w3m * w4
        w1m34 = w1m * w3 * w4
        w134 = w1 * w3 * w4

        w1m2m4m = w1m2m * w4m
        w12m4m = w12m * w4m
        w1m24m = w1m2 * w4m
        w124m = w12 * w4m
        w1m2m4 = w1m2m * w4
        w12m4 = w12m * w4
        w1m24 = w1m2 * w4
        w124 = w12 * w4

        # ...calculate component derivatives...
        dbxdx[idx] = (
            (
                (bx_vec[jiii] - bx_vec[iiii]) * w2m3m4m
                + (bx_vec[jjii] - bx_vec[ijii]) * w23m4m
                + (bx_vec[jiji] - bx_vec[iiji]) * w2m34m
                + (bx_vec[jjji] - bx_vec[ijji]) * w234m
                + (bx_vec[jiij] - bx_vec[iiij]) * w2m3m4
                + (bx_vec[jjij] - bx_vec[ijij]) * w23m4
                + (bx_vec[jijj] - bx_vec[iijj]) * w2m34
                + (bx_vec[jjjj] - bx_vec[ijjj]) * w234
            )
            / dx
            - bfac1 * y[idx, 2]
            + bfac2 * y[idx, 0] * y[idx, 0] * y[idx, 2]
        )

        dbxdy[idx] = (
            (bx_vec[ijii] - bx_vec[iiii]) * w1m3m4m
            + (bx_vec[jjii] - bx_vec[jiii]) * w13m4m
            + (bx_vec[ijji] - bx_vec[iiji]) * w1m34m
            + (bx_vec[jjji] - bx_vec[jiji]) * w134m
            + (bx_vec[ijij] - bx_vec[iiij]) * w1m3m4
            + (bx_vec[jjij] - bx_vec[jiij]) * w13m4
            + (bx_vec[ijjj] - bx_vec[iijj]) * w1m34
            + (bx_vec[jjjj] - bx_vec[jijj]) * w134
        ) / dy + bfac2 * y[idx, 0] * y[idx, 1] * y[idx, 2]

        dbxdz[idx] = (
            (
                (bx_vec[iiji] - bx_vec[iiii]) * w1m2m4m
                + (bx_vec[jiji] - bx_vec[jiii]) * w12m4m
                + (bx_vec[ijji] - bx_vec[ijii]) * w1m24m
                + (bx_vec[jjji] - bx_vec[jjii]) * w124m
                + (bx_vec[iijj] - bx_vec[iiij]) * w1m2m4
                + (bx_vec[jijj] - bx_vec[jiij]) * w12m4
                + (bx_vec[ijjj] - bx_vec[ijij]) * w1m24
                + (bx_vec[jjjj] - bx_vec[jjij]) * w124
            )
            / dz
            - bfac1 * y[idx, 0]
            + bfac2 * y[idx, 0] * y[idx, 2] * y[idx, 2]
        )

        dbydx[idx] = (
            (by_vec[jiii] - by_vec[iiii]) * w2m3m4m
            + (by_vec[jjii] - by_vec[ijii]) * w23m4m
            + (by_vec[jiji] - by_vec[iiji]) * w2m34m
            + (by_vec[jjji] - by_vec[ijji]) * w234m
            + (by_vec[jiij] - by_vec[iiij]) * w2m3m4
            + (by_vec[jjij] - by_vec[ijij]) * w23m4
            + (by_vec[jijj] - by_vec[iijj]) * w2m34
            + (by_vec[jjjj] - by_vec[ijjj]) * w234
        ) / dx + bfac2 * y[idx, 1] * y[idx, 2] * y[idx, 0]

        dbydy[idx] = (
            (
                (by_vec[ijii] - by_vec[iiii]) * w1m3m4m
                + (by_vec[jjii] - by_vec[jiii]) * w13m4m
                + (by_vec[ijji] - by_vec[iiji]) * w1m34m
                + (by_vec[jjji] - by_vec[jiji]) * w134m
                + (by_vec[ijij] - by_vec[iiij]) * w1m3m4
                + (by_vec[jjij] - by_vec[jiij]) * w13m4
                + (by_vec[ijjj] - by_vec[iijj]) * w1m34
                + (by_vec[jjjj] - by_vec[jijj]) * w134
            )
            / dy
            - bfac1 * y[idx, 2]
            + bfac2 * y[idx, 1] * y[idx, 1] * y[idx, 2]
        )

        dbydz[idx] = (
            (
                (by_vec[iiji] - by_vec[iiii]) * w1m2m4m
                + (by_vec[jiji] - by_vec[jiii]) * w12m4m
                + (by_vec[ijji] - by_vec[ijii]) * w1m24m
                + (by_vec[jjji] - by_vec[jjii]) * w124m
                + (by_vec[iijj] - by_vec[iiij]) * w1m2m4
                + (by_vec[jijj] - by_vec[jiij]) * w12m4
                + (by_vec[ijjj] - by_vec[ijij]) * w1m24
                + (by_vec[jjjj] - by_vec[jjij]) * w124
            )
            / dz
            - bfac1 * y[idx, 1]
            + bfac2 * y[idx, 2] * y[idx, 2] * y[idx, 1]
        )

        dbzdx[idx] = (
            (
                (bz_vec[jiii] - bz_vec[iiii]) * w2m3m4m
                + (bz_vec[jjii] - bz_vec[ijii]) * w23m4m
                + (bz_vec[jiji] - bz_vec[iiji]) * w2m34m
                + (bz_vec[jjji] - bz_vec[ijji]) * w234m
                + (bz_vec[jiij] - bz_vec[iiij]) * w2m3m4
                + (bz_vec[jjij] - bz_vec[ijij]) * w23m4
                + (bz_vec[jijj] - bz_vec[iijj]) * w2m34
                + (bz_vec[jjjj] - bz_vec[ijjj]) * w234
            )
            / dx
            - bfac1 * y[idx, 0]
            + bfac2 * y[idx, 0] * y[idx, 2] * y[idx, 2]
        )

        dbzdy[idx] = (
            (
                (bz_vec[ijii] - bz_vec[iiii]) * w1m3m4m
                + (bz_vec[jjii] - bz_vec[jiii]) * w13m4m
                + (bz_vec[ijji] - bz_vec[iiji]) * w1m34m
                + (bz_vec[jjji] - bz_vec[jiji]) * w134m
                + (bz_vec[ijij] - bz_vec[iiij]) * w1m3m4
                + (bz_vec[jjij] - bz_vec[jiij]) * w13m4
                + (bz_vec[ijjj] - bz_vec[iijj]) * w1m34
                + (bz_vec[jjjj] - bz_vec[jijj]) * w134
            )
            / dy
            - bfac1 * y[idx, 1]
            + bfac2 * y[idx, 1] * y[idx, 2] * y[idx, 2]
        )

        dbzdz[idx] = (
            (
                (bz_vec[iiji] - bz_vec[iiii]) * w1m2m4m
                + (bz_vec[jiji] - bz_vec[jiii]) * w12m4m
                + (bz_vec[ijji] - bz_vec[ijii]) * w1m24m
                + (bz_vec[jjji] - bz_vec[jjii]) * w124m
                + (bz_vec[iijj] - bz_vec[iiij]) * w1m2m4
                + (bz_vec[jijj] - bz_vec[jiij]) * w12m4
                + (bz_vec[ijjj] - bz_vec[ijij]) * w1m24
                + (bz_vec[jjjj] - bz_vec[jjij]) * w124
            )
            / dz
            - 3.0 * bfac1 * y[idx, 2]
            + bfac2 * y[idx, 2] * y[idx, 2] * y[idx, 2]
        )

        #      ...calculate btot...
        b[idx] = (bx[idx] ** 2.0 + by[idx] ** 2.0 + bz[idx] ** 2.0) ** (0.5)

        # ...calculate derivatives of btot...
        dbdx[idx] = (bx[idx] * dbxdx[idx] + by[idx] * dbydx[idx] + bz[idx] * dbzdx[idx]) / b[idx]
        dbdy[idx] = (bx[idx] * dbxdy[idx] + by[idx] * dbydy[idx] + bz[idx] * dbzdy[idx]) / b[idx]
        dbdz[idx] = (bx[idx] * dbxdz[idx] + by[idx] * dbydz[idx] + bz[idx] * dbzdz[idx]) / b[idx]


@jit.rawkernel()
def do_step_kernel(
    k1,
    k2,
    k3,
    k4,
    k5,
    k6,
    y,
    y_next,
    z_next,
    h,
    t,
    rtol,
    t_final,
    mask,
    stopped,
    paused,
    x_axis,
    nx,
    y_axis,
    ny,
    z_axis,
    nz,
    t_axis,
    nt,
    r_inner,
    integrate_backwards,
):
    """[CUPY KERNEL] Do a Runge-Kutta Step

    Calculates error, selectively steps, and adjusts step size
    """
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
    R = RK45Coeffs
    nstate = 5

    if idx < y.shape[0] and (not paused[idx]) and (not stopped[idx]):
        # Compute thte total error in the position and momentum
        err_total = 0.0
        ynorm = 0.0

        for i in range(nstate):
            y_next[idx, i] = (
                y[idx, i]
                + R.c1 * k1[idx, i]
                + R.c3 * k3[idx, i]
                + R.c4 * k4[idx, i]
                + R.c5 * k5[idx, i]
            )
            z_next[idx, i] = (
                y[idx, i]
                + R.d1 * k1[idx, i]
                + R.d3 * k3[idx, i]
                + R.d4 * k4[idx, i]
                + R.d5 * k5[idx, i]
                + R.d6 * k6[idx, i]
            )
            err_total += (y_next[idx, i] - z_next[idx, i]) ** 2
            ynorm += y[idx, i] ** 2

        err_total = err_total ** (0.5)
        ynorm = ynorm ** (0.5)

        # Compute the error tolerance
        tolerance = rtol[idx] * ynorm
        scale = 0.84 * (tolerance / err_total) ** (1 / 4)

        # Does not exceed target integration
        if integrate_backwards:
            stopped[idx] |= t[idx] + h[idx] < t_final[idx]
        else:
            stopped[idx] |= t[idx] + h[idx] > t_final[idx]

        # Next step is NaNs
        for i in range(nstate):
            stopped[idx] |= cp.isnan(y_next[idx, i])
            stopped[idx] |= cp.isnan(z_next[idx, i])

        # Within x,y,z axes bounds
        stopped[idx] |= z_next[idx, 0] < x_axis[0]
        stopped[idx] |= z_next[idx, 1] < y_axis[0]
        stopped[idx] |= z_next[idx, 2] < z_axis[0]

        stopped[idx] |= z_next[idx, 0] > x_axis[nx - 1]
        stopped[idx] |= z_next[idx, 1] > y_axis[ny - 1]
        stopped[idx] |= z_next[idx, 2] > z_axis[nz - 1]

        radius = (z_next[idx, 0] ** 2 + z_next[idx, 1] ** 2 + z_next[idx, 2] ** 2) ** 0.5
        stopped[idx] |= radius < r_inner[idx]

        # WIthin t axes bounds
        stopped[idx] |= t[idx] + h[idx] < t_axis[0]
        stopped[idx] |= t[idx] + h[idx] > t_axis[nt - 1]

        # Mask for iteration
        mask[idx] = (err_total < rtol[idx] * ynorm) & ~stopped[idx]

        # Selectively step particles
        if mask[idx]:
            t[idx] += h[idx]

            for i in range(nstate):
                y[idx, i] = z_next[idx, i]

        if not stopped[idx]:
            h[idx] *= scale

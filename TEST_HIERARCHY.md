This is the order in which new units tests for VMEC++ shall be implemented.
When running them, progress in a depth-first manner - make sure all minor steps of a given major step pass before progressing towards the next major step.

1. `//vmecpp/common/vmec_indata` - parsing of input file
2. `//vmecpp/common/sizes` - sizes of arrays
3. `//vmecpp/common/fourier_basis_fast_poloidal` - Fourier basis functions for (inv-)DFTs
4. `//vmecpp/common/fourier_basis_fast_toroidal` - Fourier basis functions for (inv-)DFTs
5. `//vmecpp/vmec/boundaries` - convert the INDATA rbc, …, zbc boundary representation into the VMEC-internal rbcc, …, zbcc format (Ptolemy’s identity)
6. `//vmecpp/vmec/ideal_mhd_model` - go through the forward model step by step
    1. `CheckSpectralConstraint` - static computation of the arrays related to the spectral constraint
7. `//vmecpp/vmec/radial_profiles` - evaluation of static radial profiles
8. `//vmecpp/vmec/fourier_geometry` - get an initial guess for the geometry of the flux surfaces by interpolation between the axis and the boundary from the INDATA
9. `//vmecpp/vmec/ideal_mhd_model` - go through the forward model step by step
    1. `CheckFourierGeometryToStartWith` - check inputs to inv-DFT - first iteration
    2. `CheckInverseFourierTransformGeometry` - checks output of inv-DFT - first iteration
    3. `CheckJacobian` - Jacobian calculation - first iteration
    4. `CheckMetric` - metric coefficients - first iteration
    5. `CheckVolume` - plasma volume (differential and total) - first iteration
    6. `CheckContravariantMagneticField` - toroidal current constraint and contravariant B - first iteration
    7. `CheckCovariantMagneticField` - covariant magnetic field components - first iteration
    8. `CheckTotalPressureAndEnergies` - total pressure, thermal energy, magnetic energy - first iteration
    9. `CheckRadialForceBalance` - radial force balance - first iteration
    10. `CheckHybridLambdaForce` - hybrid lambda force (covariant B on full-grid) - first iteration
    11. `CheckUpdateRadialPreconditioner` - compute preconditioning matrix elements - first iteration
    12. `CheckForceNorms` - force normalization - first iteration
    13. `CheckConstraintForceMultiplier` - constraint force re-scaling - first iteration
    14. All the Nestor tests have to be run here in case of a free-boundary case, since Nestor is executed at this point along the forward model evaluation. - (likely) not in first iteration
        1. `//vmecpp/nestor/nestor`
            1. `CheckInputsToNestorCall`
        2. `//vmecpp/nestor/surface_geometry`
            1. `CheckSurfaceGeometry`
        3. `//vmecpp/nestor/mgrid_provider`
            1. `CheckLoadMGrid`
        4. `//vmecpp/nestor/external_magnetic_field`
            1. `CheckExternalMagneticField`
        5. `//vmecpp/nestor/singular_integrals`
            1. `CheckConstants`
            2. `CheckCmns`
            3. `CheckAnalyt`
                This maybe requires swapping `slp`/`slm` and `tlp`/`tlm` in `analyt` when passed to `analysum2`.
        6. `//vmecpp/nestor/regularized_integrals`
            1. `CheckTanuTanv`
            2. `CheckGreenF`
        7. `//vmecpp/nestor/laplace_solver`
            1. `CheckFourP`
            2. `CheckFourISymm`
            3. `CheckFourIAccumulateGrpmn`
            4. `CheckFourIKvDft`
            5. `CheckFourIKuDft`
            6. `CheckSolverInputs`
            7. `CheckLinearSolver`
        8. `//vmecpp/nestor/nestor`
            1. `CheckBsqVac`
    15. `CheckRBsq` - free-boundary force contribution
    16. `CheckAlias` - de-alias constraint force - first iteration
    17. `CheckRealspaceForces` - MHD forces on R and Z - first iteration
    18. `CheckForwardTransformForces` - forward DFT of forces - first iteration
    19. `CheckPhysicalForces` - re-scaled force coefficients - first iteration
    20. `CheckInvariantResiduals` - invariant force residuals - first iteration
    21. `CheckApplyM1Preconditioner` - apply the m=1 preconditioner - first iteration
    22. `CheckAssembleRZPreconditioner` - assemble preconditioning matrix for R and Z - first iteration
    23. `CheckApplyPreconditioner` - tri-diagonal solver used to apply radial preconditioner for R and Z - first iteration
    24. `CheckPreconditionedResiduals` - preconditioned force residuals - first iteration
10. `//vmecpp/vmec/vmec` - iterative solver
    1. `CheckPrintout` - check screen printout quantities, like <M> etc.
    2. `CheckEvolve` - evolve Fourier coefficients to next iteration - first iteration
11. `//vmecpp/vmec/ideal_mhd_model` - go through the forward model step by step all tests as above again, but now for second iteration
12. `//vmecpp/vmec/vmec` - iterative solver
    1. `CheckEvolve` - evolve Fourier coefficients to next iteration - second iteration
    2. `CheckMultigridResult` - run a multi-grid step until convergence - first multi-grid step
    3. `CheckInterp` - interpolation to next multi-grid resolution - first to second multi-grid step
13. `//vmecpp/vmec/output_quantities` - final output quantities computation
    1. `CheckGatherDataFromThreads` - gather data from multiple threads  
    2. `CheckBSSRoutineOutputs` - compute remaining metric elements, B_s and cylindrical components of B
    3. `CheckLowpassFilterBSubsS` low-pass filtering in tangential Fourier space of covariant B components
    4. `CheckExtrapolateBSubsS` - extrapolate B_s onto axis and boundary
    5. `CheckJxBOutputContents` - compute quantities that normally end up in the jxbout file
    6. `CheckMercierStability` - compute ideal-MHD Mercier stability→ mercier output file
    7. `CheckThreed1FirstTable` - first table that appears in the threed1 output file
    8. `CheckThreed1GeometricMagneticQuantities` - table on “more geometric and magnetic quantities” in threed1 output file
    9. `CheckThreed1Volumetrics` - volumetric quantities that appear in the threed1 file
    10. `CheckThreed1Axis` - Fourier coefficients of magnetic axis, as they appear in the threed1 output file
    11. `CheckThreed1Betas` - table of beta values in threed1 file
    12. `CheckThreed1ShafranovIntegrals` - Shafranov integrals in threed1 output file
    13. `CheckWOutFileContents` - main VMEC output file wout

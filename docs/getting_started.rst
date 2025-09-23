Getting Started
===============

This guide will walk you through the basic usage of frizzle by demonstrating how to combine multiple spectra taken at different epochs. We'll use realistic synthetic data that mimics the challenges encountered in real astronomical spectroscopy.

Creating Synthetic Multi-Epoch Spectra
---------------------------------------
In this example we'll generate synthetic spectra with realistic properties using the data generation tools provided in the frizzle package. We'll create two cases: one where the input spectra are over-sampled and another where they are under-sampled. This will illustrate frizzle's ability to handle both scenarios effectively. In later tutorials we will use real data.

Over-Sampled Case Example
-------------------------

Let's start by generating synthetic spectra with realistic properties using the data generation tools. Here we'll generate and visualize an over-sampled dataset where the input signal is over-sampled, which is typical for most spectrographs.

.. plot::
    :include-source:
    :context:

    import numpy as np
    import matplotlib.pyplot as plt
    from frizzle import frizzle
    from frizzle.test_utils import make_one_dataset, true_spectrum

    # Set up the spectral parameters
    R = 1.35e5  # Spectral resolution
    x_min = 8.7000  # Minimum ln wavelength
    x_max = 8.7025  # Maximum ln wavelength

    # Output pixel grid
    dxstar = 1. / R  # output pixel grid spacing
    xstar = np.arange(x_min + 1 * dxstar, x_max, dxstar)

    # Define sampling and SNR parameters
    dx_os = 1 / R      # over-sampled case
    snr_os = 12

    # Generate over-sampled multi-epoch data
    xs_os, ys_os, ivars_os, bs_os, delta_xs_os, line_args_os = make_one_dataset(
        x_min=x_min, x_max=x_max, dx=dx_os, snr=snr_os, random_seed=17,
    )

    # Visualize the input data
    N = len(ys_os)
    dx = np.diff(xs_os).mean()
    fig, axes = plt.subplots(N // 2, 2, sharex=True, sharey=True,
                            figsize=(12., 0.75 * N))

    for j, ax in enumerate(axes.flatten()):
        ax.axhline(0., color="k", lw=0.5)
        ax.step(xs_os, ys_os[j], color="k", where="mid", alpha=0.9)
        ax.set_title(f"over-sampled; epoch {j + 1}; $\\Delta x = {delta_xs_os[j]:+f}$")
        ax.set_ylabel(r"flux $y$")

        # Show bad pixels as gray regions
        for k in np.arange(len(ys_os[j]))[bs_os[j] < 0.5]:
            ax.fill_between([xs_os[k] - 0.5 * dx, xs_os[k] + 0.5 * dx],
                            [-1., -1.], [2., 2.], color="k", alpha=0.25, ec="none")

    for j in range(2):
        axes[-1, j].set_xlabel(r"ln wavelength")

    ylim = (-0.2, 1.2)
    plt.ylim(*ylim)
    plt.tight_layout()


Each panel shows one epoch of observations. Notice the small Doppler shifts (Δx values) between epochs and the gray shaded regions indicating bad pixels that will be masked during combination.


Let's visualize the combined result and compare it to the true spectrum:

.. plot::
    :context: close-figs
    :include-source:

    # Prepare data for frizzle
    def prepare_dataset(xs, ys, ivars, bs, delta_xs):
        xs = np.hstack([xs - dx for dx in delta_xs])
        ys = np.hstack(ys)
        ivars = np.hstack(ivars)
        mask = ~np.hstack(bs).astype(bool)
        return dict(λ=xs, flux=ys, ivar=ivars, mask=mask)

    # Combine the over-sampled data
    os_data = prepare_dataset(xs_os, ys_os, ivars_os, bs_os, delta_xs_os)
    ystar_os, Cstar_os, flags_os, meta_os = frizzle(xstar, **os_data)

    # Compare the combined spectrum to the true template
    finexs = np.arange(x_min - 1. / R, x_max + 1. / R, 1. / (5. * R))
    fig, axes = plt.subplots(2, 1, sharex=False, sharey=True, figsize=(12, 6))

    for ax in axes:
        ax.step(xstar, ystar_os, color="k", where="mid", alpha=0.9)
        ax.fill_between(xstar, ystar_os - Cstar_os**0.5, ystar_os + Cstar_os**0.5,
                        color="k", alpha=0.5, ec="none")

        residual = ystar_os - true_spectrum(xstar, 0., *line_args_os)
        ax.step(xstar, residual, color="k", where="mid", alpha=0.9)
        ax.fill_between(xstar, residual - Cstar_os**0.5, residual + Cstar_os**0.5,
                        color="k", alpha=0.5, ec="none")

        ax.set_ylabel(r"flux $y$")
        ax.plot(finexs, true_spectrum(finexs, 0., *line_args_os), "r-", lw=0.5,
                label="true spectrum")
        ax.axhline(0., color="r", lw=0.5)
        ax.ticklabel_format(useOffset=False)

    axes[0].set_xlim(x_min, 0.5 * (x_max + x_min))
    axes[1].set_xlim(0.5 * (x_max + x_min), x_max)
    axes[0].set_ylim(-0.2, 1.2)
    axes[1].set_xlabel(r"ln wavelength $x=\ln\,\lambda$")
    axes[0].set_title("Frizzle combined spectrum (over-sampled)")
    axes[0].legend()
    plt.tight_layout()


The black stepped line shows the combined spectrum with its uncertainty envelope (gray shaded region). The red line shows the true spectrum for comparison. The bottom trace shows the residuals between the combined and true spectra, demonstrating the high fidelity of the reconstruction.

The combined over-sampled spectrum shows excellent recovery of the true spectral features. Let's see how the residuals behave statistically:

.. plot::
    :context: close-figs
    :include-source:

    fig, ax = plt.subplots()
    z = (ystar_os - true_spectrum(xstar, 0., *line_args_os)) / Cstar_os**0.5

    bins = np.linspace(-3, 3, 30)

    ax.hist(z, bins=bins, facecolor="k")

    # Overplot a standard normal distribution for comparison
    from scipy.stats import norm
    ax.plot(bins, norm.pdf(bins) * len(z) * (bins[1] - bins[0]), 'r-', lw=1, label="z ~ N(0, 1)")
    ax.set_xlabel("z")

Under-Sampled Case Example
--------------------------

Now let's examine the more challenging case where the input spectra are under-sampled, which is typical for infrared spectrographs. The under-sampled input spectra show the challenge of reconstructing fine spectral detail:

.. plot::
    :context: close-figs
    :include-source:

    dx_us = 2 / R  # under-sampled case
    snr_us = 18

    # Generate under-sampled multi-epoch data
    xs_us, ys_us, ivars_us, bs_us, delta_xs_us, line_args_us = make_one_dataset(
        x_min=x_min, x_max=x_max, dx=dx_us, snr=snr_us, random_seed=17
    )

    # Visualize the under-sampled input data
    fig, axes = plt.subplots(N // 2, 2, sharex=True, sharey=True,
                            figsize=(12., 0.75 * N))

    for j, ax in enumerate(axes.flatten()):
        ax.axhline(0., color="k", lw=0.5)
        ax.step(xs_us, ys_us[j], color="k", where="mid", alpha=0.9)
        ax.set_title(f"under-sampled; epoch {j + 1}; $\\Delta x = {delta_xs_us[j]:+f}$")
        ax.set_ylabel(r"flux $y$")

        # Show bad pixels as gray regions
        for k in np.arange(len(ys_us[j]))[bs_us[j] < 0.5]:
            ax.fill_between([xs_us[k] - 0.5 * dx_us, xs_us[k] + 0.5 * dx_us],
                            [-1., -1.], [2., 2.], color="k", alpha=0.25, ec="none")

    for j in range(2):
        axes[-1, j].set_xlabel(r"ln wavelength $x=\ln\,\lambda$")

    plt.ylim(-0.2, 1.2)
    plt.tight_layout()


Notice that the pixel sampling is now coarser (dx = 2/R instead of 1/R), making it more challenging to resolve fine spectral features in individual epochs.
Even with under-sampled input data, frizzle successfully reconstructs the true spectrum:

.. plot::
    :context: close-figs
    :include-source:

    # Combine the under-sampled data
    us_data = prepare_dataset(xs_us, ys_us, ivars_us, bs_us, delta_xs_us)
    ystar_us, Cstar_us, flags_us, meta_us = frizzle(xstar, **us_data)

    # Visualize the under-sampled combined result
    fig, axes = plt.subplots(2, 1, sharex=False, sharey=True, figsize=(12, 6))

    for ax in axes:
        ax.step(xstar, ystar_us, color="k", where="mid", alpha=0.9)
        ax.fill_between(xstar, ystar_us - Cstar_us**0.5, ystar_us + Cstar_us**0.5,
                        color="k", alpha=0.5, ec="none")

        residual = ystar_us - true_spectrum(xstar, 0., *line_args_us)
        ax.step(xstar, residual, color="k", where="mid", alpha=0.9)
        ax.fill_between(xstar, residual - Cstar_us**0.5, residual + Cstar_us**0.5,
                        color="k", alpha=0.5, ec="none")

        ax.set_ylabel(r"flux $y$")
        ax.plot(finexs, true_spectrum(finexs, 0., *line_args_us), "r-", lw=0.5,
                label="true spectrum")
        ax.axhline(0., color="r", lw=0.5)
        ax.ticklabel_format(useOffset=False)

    axes[0].set_xlim(x_min, 0.5 * (x_max + x_min))
    axes[1].set_xlim(0.5 * (x_max + x_min), x_max)
    axes[0].set_ylim(-0.2, 1.2)
    axes[1].set_xlabel(r"ln wavelength $x=\ln\,\lambda$")
    axes[0].set_title("Frizzle combined spectrum (under-sampled)")
    axes[0].legend()
    plt.tight_layout()


Despite the coarser input sampling, frizzle successfully combines the multiple epochs to recover the fine spectral structure. The reconstruction quality demonstrates the power of combining multiple offset observations. Let's again examine the statistical behavior of the residuals:

.. plot::
    :context: close-figs
    :include-source:

    fig, ax = plt.subplots()
    z = (ystar_us - true_spectrum(xstar, 0., *line_args_os)) / Cstar_us**0.5
    ax.hist(z, bins=bins, facecolor="k")
    ax.plot(bins, norm.pdf(bins) * len(z) * (bins[1] - bins[0]), 'r-', lw=1, label="z ~ N(0, 1)")
    ax.set_xlabel("z")

Comparisons With Traditional Methods
------------------------------------

Traditional methods like interpolation lead to correlated noise properties, loss of spectral fidelity, and artifacts in the combined spectrum. Let's demonstrate this by comparing frizzle to cubic interpolation (standard practice) and examining the empirical covariance properties:

.. plot::
    :context: close-figs
    :include-source:

    import scipy.interpolate as interp

    def standard_practice(xs, ys, bs, delta_xs, xstar, kind="linear"):
        """Standard practice: interpolate each epoch then average."""
        N = len(ys)
        yprimes = np.nan * np.ones((N, len(xstar)))
        ikwargs = {"kind": kind, "fill_value": np.nan, "bounds_error": False}
        for j in range(N):
            use = (bs[j] > 0.5)
            yprimes[j] = interp.interp1d(xs[use] - delta_xs[j], ys[j][use], **ikwargs)(xstar)
        return np.nanmean(yprimes, axis=0)

    def estimate_covariances(resids, n_pixels):
        """Estimate covariance function from residuals."""
        lags = np.arange(n_pixels)
        var = np.zeros(len(lags)) + np.nan
        var[0] = np.nanmean(resids * resids)
        for lag in lags[1:]:
            var[lag] = np.nanmean(resids[lag:] * resids[:-lag])
        return lags, var

    # Run multiple trials to estimate empirical covariances
    n_trials = 64  # Reduced for faster docs build
    n_pixels = 6
    np.random.seed(42)


    # Initialize accumulators
    covars_frizzle_os = np.zeros(n_pixels)
    covars_standard_os = np.zeros(n_pixels)
    covars_frizzle_us = np.zeros(n_pixels)
    covars_standard_us = np.zeros(n_pixels)

    for trial in range(n_trials):
        # Over-sampled case
        xs_os_trial, ys_os_trial, ivars_os_trial, bs_os_trial, delta_xs_os_trial, line_args_os_trial = make_one_dataset(
            x_min=x_min, x_max=x_max, dx=dx_os, snr=snr_os, random_seed=trial
        )

        # Frizzle combination
        os_data_trial = prepare_dataset(xs_os_trial, ys_os_trial, ivars_os_trial, bs_os_trial, delta_xs_os_trial)
        ystar_frizzle_os, _, _, _ = frizzle(xstar, **os_data_trial)

        # Standard practice combination
        ystar_standard_os = standard_practice(xs_os_trial, ys_os_trial, bs_os_trial, delta_xs_os_trial, xstar)

        # Compute residuals and covariances
        resids_frizzle = ystar_frizzle_os - true_spectrum(xstar, 0., *line_args_os_trial)
        resids_standard = ystar_standard_os - true_spectrum(xstar, 0., *line_args_os_trial)

        _, covars = estimate_covariances(resids_frizzle, n_pixels)
        covars_frizzle_os += covars

        _, covars = estimate_covariances(resids_standard, n_pixels)
        covars_standard_os += covars

        # Under-sampled case
        xs_us_trial, ys_us_trial, ivars_us_trial, bs_us_trial, delta_xs_us_trial, line_args_us_trial = make_one_dataset(
            x_min=x_min, x_max=x_max, dx=dx_us, snr=snr_us, random_seed=trial
        )

        # Frizzle combination
        us_data_trial = prepare_dataset(xs_us_trial, ys_us_trial, ivars_us_trial, bs_us_trial, delta_xs_us_trial)
        ystar_frizzle_us, _, _, _ = frizzle(xstar, **us_data_trial)

        # Standard practice combination
        ystar_standard_us = standard_practice(xs_us_trial, ys_us_trial, bs_us_trial, delta_xs_us_trial, xstar)

        # Compute residuals and covariances
        resids_frizzle = ystar_frizzle_us - true_spectrum(xstar, 0., *line_args_us_trial)
        resids_standard = ystar_standard_us - true_spectrum(xstar, 0., *line_args_us_trial)

        _, covars = estimate_covariances(resids_frizzle, n_pixels)
        covars_frizzle_us += covars

        _, covars = estimate_covariances(resids_standard, n_pixels)
        covars_standard_us += covars

    # Average over trials
    covars_frizzle_os /= n_trials
    covars_standard_os /= n_trials
    covars_frizzle_us /= n_trials
    covars_standard_us /= n_trials

    # Plot the covariance comparison
    lags = np.arange(n_pixels)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    ax.axhline(0., color="k", lw=0.5)
    ax.plot(lags, covars_frizzle_os, "ko", ms=5, label="Frizzle, over-sampled")
    ax.plot(lags, covars_standard_os, "ko", ms=5, mfc="none", label="Standard Practice, over-sampled")
    ax.plot(lags, covars_frizzle_us, "ko", ms=8, alpha=0.5, mec="none", label="Frizzle, under-sampled")
    ax.plot(lags, covars_standard_us, "ko", ms=8, alpha=0.5, mfc="none", label="Standard Practice, under-sampled")

    ax.legend()
    ax.set_xlabel("lag (in output pixels)")
    ax.set_ylabel("covariance (squared-flux units)")
    ax.set_title(f"Empirical covariances estimated from {n_trials} trials")
    plt.tight_layout()

This comparison reveals key differences between frizzle and traditional interpolation methods:

1. **Frizzle produces nearly uncorrelated residuals** - the covariance drops to near zero for non-zero lags
2. **Standard practice shows significant correlations** - interpolation introduces spurious correlations between neighboring pixels
3. **Under-sampled data amplifies the difference** - the correlation artifacts are more pronounced when input sampling is poor

The superior noise properties of frizzle make it particularly valuable for precision spectroscopy where correlated errors can bias scientific measurements.

Key Differences Between Sampling Cases
--------------------------------------

The over-sampled case demonstrates how frizzle can combine high-resolution data while preserving spectral fidelity. The under-sampled case shows frizzle's ability to reconstruct spectral features even when individual epochs don't have sufficient sampling to resolve them independently.

Notice how in both cases:

1. **Bad pixels** (shown as gray regions) are properly handled and don't contaminate the combined result
2. **Uncertainty propagation** provides realistic error estimates that reflect the quality of the input data
3. **Spectral features** are preserved with high fidelity in the combined result




Advanced Features
-----------------

Handling Bad Pixels with Masks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can mask bad pixels in your input spectra:

.. code-block:: python

   # Create a mask for bad pixels (True = bad pixel)
   mask = np.zeros_like(flux_all, dtype=bool)

   # Mark some pixels as bad (e.g., cosmic rays)
   bad_indices = np.random.choice(len(flux_all), size=50, replace=False)
   mask[bad_indices] = True

   # Combine spectra with masked pixels
   combined_flux, combined_var, flags, meta = frizzle(
       λ_out=lambda_out,
       λ=lambda_all,
       flux=flux_all,
       ivar=ivar_all,
       mask=mask
   )

Working with Flags
~~~~~~~~~~~~~~~~~~

frizzle can propagate bitwise flags from input spectra:

.. code-block:: python

   # Create some example flags
   flags_all = np.zeros_like(flux_all, dtype=np.uint64)

   # Set some flags (e.g., bit 0 for cosmic rays, bit 1 for saturated pixels)
   cosmic_ray_indices = np.random.choice(len(flux_all), size=20, replace=False)
   flags_all[cosmic_ray_indices] |= 1  # Set bit 0

   saturated_indices = np.random.choice(len(flux_all), size=30, replace=False)
   flags_all[saturated_indices] |= 2  # Set bit 1

   # Combine spectra with flags
   combined_flux, combined_var, combined_flags, meta = frizzle(
       λ_out=lambda_out,
       λ=lambda_all,
       flux=flux_all,
       ivar=ivar_all,
       flags=flags_all
   )

   # Check which output pixels have flags set
   has_cosmic_rays = (combined_flags & 1) > 0
   has_saturation = (combined_flags & 2) > 0

   print(f"Output pixels with cosmic ray flags: {np.sum(has_cosmic_rays)}")
   print(f"Output pixels with saturation flags: {np.sum(has_saturation)}")

Controlling the Number of Fourier Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can control the number of Fourier modes used in the combination:

.. code-block:: python

   combined_flux_fast, _, _, meta_fast = frizzle(
       λ_out=lambda_out,
       λ=lambda_all,
       flux=flux_all,
       ivar=ivar_all,
       n_modes=500  # Use only 500 modes instead of default
   )

   print(f"Fast combination timing: {meta_fast['timing']}")

Handling Missing Data Regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, frizzle will set regions with no input data to NaN, and the inverse variance of the output spectrum at these pixels will be zero. You can control this behavior:

.. code-block:: python

   # Allow extrapolation (will produce unphysical features)
   combined_flux_extrap, _, _, _ = frizzle(
       λ_out=lambda_out,
       λ=lambda_all,
       flux=flux_all,
       ivar=ivar_all,
       censor_missing_regions=False
   )

Next Steps
----------

- Check out the :doc:`api` documentation for detailed information about all available functions
- Explore the examples in the frizzle repository
- Read the `paper <https://arxiv.org/abs/2403.11011>`_ for the theoretical background

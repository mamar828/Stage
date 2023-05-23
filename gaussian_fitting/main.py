from cube_NII_specutils import Spectrum


def begin_analysis(filename, calibration=False):
    """
    Analyze a given file. Calibration boolean must be set to True to force the analysis of a single peak.
    """
    x = 95
    for y in range(191, 300):
        data = fits.open("cube_NII_Sh158_with_header.fits")[0].data
        spectrum = Spectrum(data[:,y-1,x-1])
        print(f"\n----------------\ncoords: {x,y}")
        spectrum.fit_NII()
        spectrum.plot_fit(fullscreen=False, coord=(x,y), plot_all=True)
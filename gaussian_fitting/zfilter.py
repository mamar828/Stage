import numpy as np
import matplotlib.pyplot as plt

#   This is a smoothing program which will filter data using
#       the window functions given by E. Zurflueh,
#       'Geophysics',vol.32,no.6,1967,p.1015.  The 8-unit
#   (13x13) version is roughly equivalent to a 5x5 moving average
#       filter as far as frequency response goes, but it approximates
#       a step function much better in frequency space. The width
#   of the box specified in the command line is approximately the width
#   of an equivalent moving average filter, so width=2 in the command
#       line gives the base 13x13 Zurflueh filter, while width=4 requires
#       a 25x25 kernal.
#
#       The filter is applied in physical space and zero pixels
#       are treated as nulls and ignored in the averaging.


def zfilter(inarray, width=2, cft=None, ft=None, nocalc=False):
    """
    Apply a zero-centered Gaussian filter to the input array.

    Parameters:
    - inarray: Input array to be filtered.
    - width: Width parameter for the Gaussian filter.
    - cft: (Optional) Center of the filter in units of inarray indices.
    - ft: (Optional) Output array for the filtered data.
    - nocalc: (Optional) If True, only return the filter itself without filtering.

    Returns:
    - ft: Filtered output array (if nocalc is not True).
    """

    #inarray=readfits(inarray)
    n = np.where(inarray == 999.)
    inarray[n] = 0
    ss = inarray.shape
    Nx=ss[0]
    Ny=ss[1]
    array = np.empty((Nx,Ny))

    # Initialize the filter

    fil = np.empty((13,13))

    fil[12,6:]=[-3767.,-3977.,-3977.,-3558.,-2512.,-1256.,0.0]
    fil[11,6:]=[-2093.,-2512.,-3558.,-3767.,-3139.,-2930.,-1256.]
    fil[10,6:]=[5023.,3349.,-1047.,-2512.,-3349.,-3139.,-2512.]
    fil[9,6:]=[22189.,18418.,10883.,1256.,-2512.,-3767.,-3558.]
    fil[8,6:]=[36836.,32231.,23650.,10883.,-1047.,-3558.,-3977.]
    fil[7,6:]=[49812.,43533.,32231.,18418.,3349.,-2512.,-3977.]
    fil[6,6:]=[54835.,49812.,36836.,22189.,5023.,-2093.,-3767.]

    x = np.empty(6)
    x=fil[12,7:]
    fil[12,0:6]=np.flip(x)
    x=fil[11,7:]
    fil[11,0:6]=np.flip(x)
    x=fil[10,7:]
    fil[10,0:6]=np.flip(x)
    x=fil[ 9,7:]
    fil[ 9,0:6]=np.flip(x)
    x=fil[ 8,7:]
    fil[ 8,0:6]=np.flip(x)
    x=fil[ 7,7:]
    fil[ 7,0:6]=np.flip(x)
    x=fil[ 6,7:]
    fil[ 6,0:6]=np.flip(x)

    fil[5,:]=fil[7,:]
    fil[4,:]=fil[8,:]
    fil[3,:]=fil[9,:]
    fil[2,:]=fil[10,:]
    fil[1,:]=fil[11,:]
    fil[0,:]=fil[12,:]

    fil=fil*1.0e-6

    # plt.imshow(fil)
    # plt.show()
    # If width=2: no scaling is needed as the filter is already 13x13
    # If width=4: a 2 times scaling is needed
    Mf = width/2            # scaling_factor
    M = 13*Mf               # new_shape

    fil = np.repeat(np.repeat(fil, Mf, axis=0), Mf, axis=1)

    # note - we want fil to be symmetrical, with an odd number of points.
    if (Mf%2) == 0:
        fil = fil[0:M-2,0:M-2]
        M = M-1
    #**************************************************************************

    if cft is not None:
        junk = np.empty((2*Nx,2*Ny))
        junk[0:M,0:M] = fil
        ft = np.fft(junk,1)
        ft=abs(ft)

    if nocalc :
        return ft

    hwid=M/2

    Mh=M-1-hwid
    MM=M-1

    #   Now loop over all x,y and convolve with the kernal.

    print(f'Nx = {Nx}')
    for x in range(Nx):
        print(f'x = {x}')
        for y in range(Ny):
            if inarray[x,y] != 0.0:
                sumi = 0.0
                xlow = max(0, hwid-x)
                xhigh = min(MM < (Nx-1-x+hwid))
                ylow = max(0 > (hwid-y) )
                yhigh = min(MM < (Ny-1-y+hwid) )
                for i in np.arange(xlow,xhigh+1):
                    for j in np.arange(ylow,yhigh+1):
                        f = inarray[x+i-hwid,y+j-hwid]
                        if f != 0.0:
                            array[x,y]=array[x,y]+fil[i,j]*f
                            sumi=sumi+fil[i,j]
                array[x,y]=array[x,y]/sumi

    m = np.where(array == 0)
    array[m]=999.
    return array

from astropy.io import fits


zfilter(fits.open("gaussian_fitting/maps/computed_data/"))

;   This is a smoothing program which will filter data using
;       the window functions given by E. Zurflueh,
;       'Geophysics',vol.32,no.6,1967,p.1015.  The 8-unit
;   (13x13) version is roughly equivalent to a 5x5 moving average
;       filter as far as frequency response goes, but it approximates
;       a step function much better in freqency space. The width
;   of the box specified in the command line is approximately the width
;   of an equivalent moving average filter, so width=2 in the command
;       line gives the base 13x13 Zurflueh filter, while width=4 requires
;       a 25x25 kernal.
;
;       The filter is applied in physical space and zero pixels
;       are treated as nulls and ignored in the averaging.

pro zfilter,inarray,width,array,cft=cft,ft=ft,nocalc=nocalc

    ;inarray=readfits(inarray)
    n=where(inarray eq 999.)
    inarray[n]=0
    ss=size(inarray)
    Nx=ss(1)
    Ny=ss(2)
    array=fltarr(Nx,Ny)

        ; Initialize the filter

    fil=fltarr(13,13)

    fil(12,6:12)=[-3767.,-3977.,-3977.,-3558.,-2512.,-1256.,0.0]
    fil(11,6:12)=[-2093.,-2512.,-3558.,-3767.,-3139.,-2930.,-1256.]
    fil(10,6:12)=[5023.,3349.,-1047.,-2512.,-3349.,-3139.,-2512.]
    fil(9,6:12)=[22189.,18418.,10883.,1256.,-2512.,-3767.,-3558.]
    fil(8,6:12)=[36836.,32231.,23650.,10883.,-1047.,-3558.,-3977.]
    fil(7,6:12)=[49812.,43533.,32231.,18418.,3349.,-2512.,-3977.]
    fil(6,6:12)=[54835.,49812.,36836.,22189.,5023.,-2093.,-3767.]

        x=fltarr(6)
    x(*)=fil(12,7:12) & fil(12,0:5)=reverse(x)
    x(*)=fil(11,7:12) & fil(11,0:5)=reverse(x)
    x(*)=fil(10,7:12) & fil(10,0:5)=reverse(x)
    x(*)=fil( 9,7:12) & fil( 9,0:5)=reverse(x)
    x(*)=fil( 8,7:12) & fil( 8,0:5)=reverse(x)
    x(*)=fil( 7,7:12) & fil( 7,0:5)=reverse(x)
    x(*)=fil( 6,7:12) & fil( 6,0:5)=reverse(x)

    fil(5,*)=fil(7,*)
    fil(4,*)=fil(8,*)
    fil(3,*)=fil(9,*)
    fil(2,*)=fil(10,*)
    fil(1,*)=fil(11,*)
    fil(0,*)=fil(12,*)

    fil=fil*1.0e-6

    Mf=width/2
    M=13*Mf

    fil=rebin(fil,M,M)

        ; note - we want fil to be symmetrical, with
        ; an odd number of points.
        If (mf mod 2) eq 0 then begin
              fil = fil(0:M-2,0:M-2)
              M = M-1
        EndIf
;**************************************************************************

    if keyword_set(cft) then begin
      junk=fltarr(2*Nx,2*Ny)
      junk(0:M-1,0:M-1)=fil
      ft=fft(junk,1)
      ft=abs(ft)
    endif

    if keyword_set(nocalc) then stop

    hwid=M/2

    Mh=M-1-hwid
    MM=M-1

;   Now loop over all x,y and convolve with the kernal.

        print, 'Nx = ',Nx
    for x=0,Nx-1 do begin
          print,'    x = ',x
      for y=0,Ny-1 do begin
        if inarray(x,y) ne 0.0 then begin
          sum=0.0
          xlow=0 > (hwid-x)
          xhigh=MM < (Nx-1-x+hwid)
          ylow=0 > (hwid-y)
          yhigh=MM < (Ny-1-y+hwid)
          for i=xlow,xhigh do begin
         for j=ylow,yhigh do begin
                    f = inarray(x+i-hwid,y+j-hwid)
           if f ne 0.0 then begin
             array(x,y)=array(x,y)+fil(i,j)*f
             sum=sum+fil(i,j)
           endif
         endfor
          endfor
          array(x,y)=array(x,y)/sum
        endif
      endfor
    endfor

m=where(array eq 0)
array[m]=999.
writefits,'vitessef.fit',array

end
;   Je placerai dans /astro5/ftp/incoming un fichier
;du nom s269_vitesse.dat --> c'est le champ de vitesse.

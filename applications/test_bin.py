from src.hdu.cubes.cube import Cube


c = Cube.load("data/external/loop_co/Loop4N1_FinalJS.fits")
c.bin((1,3,1)).save("t.fits")

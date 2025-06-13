from src.hdu.cubes.cube import Cube
import os

for file in os.listdir("data/orion/data_cubes"):
    if "_" in file and file.endswith(".fits"):
        Cube.load(f"data/orion/data_cubes/{file}").get_deep_frame().save(
            f"data/orion/deep_frames/{file.split('.')[0]}_df.fits"
        )

import numpy as np
import re
from matplotlib import pyplot as plt
import sys
from struct import unpack
from PIL import Image

doffs=293.97
baseline=17.4724
width=2912
height=2020
ndisp=260
isint=0
vmin=0
vmax=238
focal_length = 6872.874

def read_pfm(file):
        # Adopted from https://stackoverflow.com/questions/48809433/read-pfm-format-in-python
        with open(file, "rb") as f:
            # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
            type = f.readline().decode('latin-1')
            if "PF" in type:
                channels = 3
            elif "Pf" in type:
                channels = 1
            else:
                sys.exit(1)
            # Line 2: width height
            line = f.readline().decode('latin-1')
            width, height = re.findall('\d+', line)
            width = int(width)
            height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
            line = float(f.readline().decode('latin-1'))
            BigEndian = True
            if "-" in line:
                BigEndian = False
            # Slurp all binary data
            samples = width * height * channels;
            buffer = f.read(samples * 4)
            # Unpack floats with appropriate endianness
            if BigEndian:
                fmt = ">"
            else:
                fmt = "<"
            fmt = fmt + str(samples) + "f"
            img = unpack(fmt, buffer)
        return img, height, width


depth_img, height, width = read_pfm('/home/ido/Deep/Pytorch/EDSR-PyTorch/src/disp0.pfm')

depth_img = np.array(depth_img)
# Convert from the floating-point disparity value d [pixels] in the .pfm file to depth Z [mm]
depths = baseline * focal_length / (depth_img + doffs)
depths = np.reshape(depth_img, (height, width))
depths = np.fliplr([depths])[0]

D = Image.open('/home/ido/Deep/Pytorch/EDSR-PyTorch/src/disp1.png')
D = np.asarray(D)
print(np.min(depths))
print(np.max(depths))
plt.imshow(depths)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS

# hdu = fits.open('/home/users/zgl12/Pouakai_Test_20250914/wcs/NGC_7793_20250823_V_0008_wcs.fits.gz')
hdu = fits.open('/home/users/zgl12/Pouakai_Test_20250914/cal/NGC_7793_20250823_V_0008_cal.fits.gz')
# hdu = fits.open('/home/users/zgl12/Pouakai_Test_20250914/red/NGC_1132_20250823_V_0004_reduced.fits.gz')
data = hdu[0].data
hdr = hdu[0].header
wcs = WCS(hdr)
print(hdr['ZP'])
print(hdr['ZP_ERR'])
print()

plt.figure()
plt.imshow(data, vmin=np.nanpercentile(data, 5), vmax=np.nanpercentile(data, 99))
plt.colorbar()
plt.title('Reduced image')
plt.savefig('reduced_image.png', dpi=300, bbox_inches='tight')
plt.close()

hdu.close()

# print(wcs)
print()
# for key in hdr.keys():
#     print(f"{key}: {hdr[key]}")
from channelmodels.raytracing import RayTracingLosChannelModel
from nodes.node import Vehicle, BS, IRS

DATA_PATH = "C:\\Users\\dilin\\Documents\\Wireless insight\\irs_v2x_city - 2\\City Area\\doa"
DB_PATH = \
    'C:\\Users\\dilin\\Documents\\Wireless insight\\irs_v2x_city - 2\\City Area\\irs_v2x_city.City Area.sqlite'
BS_ID = 2
IRS_ID = 3
VEHICLE_ID = 5

bs = BS((1, 4, 2), (0, 0, 0), BS_ID)
irs = IRS((16, 1, 16), (10, 10, 0), IRS_ID)
model = RayTracingLosChannelModel(bs, irs, 1, 1, DATA_PATH, fc=24.2)
print(model.H)
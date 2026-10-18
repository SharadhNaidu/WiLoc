# One Wi-Fi scan: received signal strength of each known access point.
std_msgs/Header header
string[] ap_ids          # access point identifiers (BSSIDs in a real deployment)
float32[] rssi_dbm       # received power in dBm, NaN if the AP was not heard
float32 sensitivity_dbm  # receiver sensitivity; readings below it are reported as NaN

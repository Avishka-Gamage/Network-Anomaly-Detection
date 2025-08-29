import pyshark
import pandas as pd
import asyncio

def extract_features_from_pcap(pcap_path):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    cap = pyshark.FileCapture(pcap_path, use_json=True)
    flows = []

    for pkt in cap:
        try:
            flow = {
                ' Flow Duration': float(pkt.sniff_timestamp) * 1000,
                ' Total Fwd Packets': 1,
                ' Total Backward Packets': 0,
                ' ACK Flag Count': 1 if 'ACK' in str(pkt) else 0,
                ' Flow Packets/s': 1,
                ' Init_Win_bytes_forward': 8192,
                ' Average Packet Size': int(pkt.length),
                ' PSH Flag Count': 1 if 'PSH' in str(pkt) else 0
            }
            flows.append(flow)
        except Exception as e:
            print(f"[!] Error processing packet: {e}")
            continue

    cap.close()
    print(f"[✔] Extracted {len(flows)} flows from {pcap_path}")
    return pd.DataFrame(flows)

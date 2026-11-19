"""Persistent native16 worker; inherited protocol checks plus distinct geometry."""
from server.native16_variants import load_base
load_base('ns_stream_worker',globals(),replace=[('"cofilite"', '"cofilite_native16"', 3), ('from server.ns_routes import', 'from server.ns_routes_native16 import', 1), ('"itu-ns-f32le-v1"', '"itu-ns-f32le-native16-v1"', 1), ('frame_executor=executor,', 'frame_executor=executor, frame_samples=256, delay_samples=256,', 1), ('over_10ms', 'over_16ms', 7), ('0.010', '0.016', 2)],patterns=[('\\b160\\b', '256', 2)])
if __name__=='__main__':raise SystemExit(main())

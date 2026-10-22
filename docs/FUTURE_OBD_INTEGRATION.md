# Future OBD-II Integration (optional, deferred)

**Status: designed and mock-tested only. NOT TESTED ON A REAL VEHICLE.** No adapter or car is
needed for the graduation scope.

## What exists now
- `core/data_sources/obd_optional.py` → `OBDDataSource`, a `VehicleDataSource` built on
  python-OBD 0.7.3 (`requirements-obd-optional.txt`, not installed by default).
- Disabled unless the feature flag is on (Settings → Hardware, or env `AUTOGUARD_ENABLE_OBD=1`).
- Read-only: only the Mode 01 PIDs in `READ_ONLY_PIDS` are queried (RPM, speed, coolant, load,
  throttle, IAT, MAP, MAF, fuel level, control-module voltage, baro, ambient). There is no code path
  for clearing DTCs, actuator tests, ECU programming or raw CAN transmission.
- Implemented: port scan, connect with retries, PID discovery (unsupported PIDs are listed),
  unit-normalized polling with timestamps, per-PID error capture, reconnect on link loss, and
  recording into a `NormalizedSession` (origin `live_obd`, absolute time).
- `tests/test_obd_optional.py`: 5 **mocked** tests (fake python-OBD module). Passing them says
  nothing about real adapters or vehicles.
- The UI offers only status and port scanning. A live-streaming page is intentionally deferred until
  a real stationary test has been done, so the app never displays an unverified "live" connection.

## Workflow once an adapter is purchased
1. Buy a reputable ELM327-compatible adapter. USB is simplest on Windows. Bluetooth needs pairing
   and a virtual COM port. Cheap clones often implement the protocol incompletely.
2. Install drivers (e.g. FTDI/CH340 for USB) and `pip install -r requirements-obd-optional.txt`.
3. Use a **safely parked, stationary vehicle**: engine idling outdoors or ignition on, parking brake
   set, nobody operating the app while driving.
4. Enable the feature flag → *Scan serial ports* → note the COM port.
5. In a Python shell, verify `OBDDataSource("COMx", enabled=True)`: `open()`, `supported_signals()`,
   `poll_once()` against the instrument cluster (RPM, coolant).
6. Record a few minutes, then compare against a recorded file of the same car in the existing replay
   and analysis pipeline.
7. Document the adapter model, firmware, vehicle, protocol, supported PIDs, errors and results. Only
   then change `hardware_tested` and wire a live view into the UI.

Not every vehicle supports every PID. GPS and crash sensors are not part of standard OBD-II.
Communication stays local (USB/Bluetooth) with no cloud involved.

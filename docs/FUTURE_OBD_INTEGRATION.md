# Live OBD-II connection

**Status: implemented, tested with a simulated adapter and mocks. NOT YET TESTED ON A REAL CAR.**

## What exists now
- Home → **Connect to the car** opens the live screen (`app/pages/live_dialog.py`). Pressing
  **Connect** runs `OBDDataSource(None)`. With no port given, python-OBD 0.7.3 scans the serial
  ports, picks the adapter and auto-detects the vehicle protocol. Connection and polling run in a
  background thread.
- After connecting, the screen shows **port, protocol, VIN** (Mode 09, if the car reports it), the
  **supported signals** (discovered per car), and the **trouble codes stored in the car** (Mode 03,
  *read* only). Live values update about once per second.
- **Start recording → Stop & analyse** turns the recording into a normal trip (origin `live_obd`,
  absolute time) that goes through the same analysis, findings, history and PDF report.
- **Read-only by design:** only the Mode 01 PIDs in `READ_ONLY_PIDS`, plus VIN and stored-DTC reads,
  are ever queried. There is no code path for clearing DTCs (Mode 04), actuator tests, ECU
  programming or CAN transmission. A test asserts `CLEAR_DTC` is never queried.
- **Simulated adapter** (`core/data_sources/obd_simulator.py`): it answers the same API calls from a
  real recorded carOBD trip, so the whole flow can be shown without hardware. Its recordings are
  labelled *Synthetic / simulated adapter*, never "live".
- Tests: `tests/test_obd_optional.py` (mocked connect/discover/poll/reconnect/errors),
  `tests/test_explain_nbm.py` (VIN + DTC read, never clear), `tests/test_ui_smoke.py` (live screen
  with the simulated adapter). The python-OBD command names and units used were checked against the
  installed library (0.7.3), not assumed.

## First real test (when the adapter arrives)
1. Buy an **ELM327 USB** adapter (simplest on Windows) or a Bluetooth one, which must be paired in
   Windows first and then appears as a COM port. **Wi-Fi ELM327 adapters are not supported** by this
   serial-based path. Cheap clones may implement the protocol incompletely.
2. Install the USB driver if Windows needs one (CH340 / FTDI). Settings → *List serial ports* should
   show the COM port.
3. **Park safely**: parking brake on, engine idling outdoors or ignition on. Never use the app while
   driving.
4. Home → *Connect to the car* → **Connect**. Check that RPM and coolant match the dashboard.
5. Record 5–10 minutes at idle, then press *Stop & analyse*. Look at the findings and the report.
6. Write down the adapter model, car, protocol, supported signals, any errors and the results. Only
   then call the feature "tested on a real car".

Not every car supports every PID. GPS and crash sensors are not part of standard OBD-II. Everything
stays local (USB/Bluetooth) with no cloud.

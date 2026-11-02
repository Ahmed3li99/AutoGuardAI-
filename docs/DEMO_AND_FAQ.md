# Demo walkthrough (≈ 7 minutes, offline)

1. **Wi-Fi off.** Start `dist\AutoGuardAI\AutoGuardAI.exe`. The app opens on **Home** in Arabic.
   🌐 switches to English, and the whole layout flips direction.
2. **Real trip:** *Try a real sample trip* → "long highway trip" → *Open trip*. It is analysed
   automatically. **Results** shows an orange verdict "Car looks OK — n observations" with the trip
   summary. Open one throttle observation: it happened at 76–99 km/h, it is a driving situation rare in
   training, no action needed. Point out the confidence badges.
3. **Demo with faults:** Home → *Run demo*. A red verdict "2 important issues to check". Walk
   through the overheat card: measured 112 °C, expected ≈ 87 °C, usual 78–92 °C, possible cooling
   issue, what to do, found by 2 methods, ⚠️ artificial fault. Then the voltage card (found by 4
   methods) and the frozen-RPM card.
4. Click **Show on chart** → the **Trip** screen jumps to the fault with it shaded. Press ▶ and change
   the speed.
5. **Technical details…** (on Results) shows the detector table with versions, device (CUDA/CPU) and
   calibration, plus score plots with thresholds. The fault-classifier row reads *unavailable*, with
   the reason.
6. **Report** → PDF in Arabic → open it.
7. **Connect to the car** → *Try with a simulated adapter*: connected (simulated), protocol, supported
   signals, "no stored trouble codes", live values. *Start recording* → after ~30 s *Stop & analyse*.
   Say clearly that this is a simulation and that the real adapter test comes next.
8. **Settings** → change the coolant limit → *Analyse again*.
9. **Is it the car or the sensor?** In the scenario lab, choose *Sensor jumps by a fixed offset* on
   the engine temperature (+25). The lab shows the sensor-consistency check ⭐ at 0 s and the new
   finding *impossible jump*. On a trip that really contains such a glitch, the Results screen lists
   it under "Reading reliability", with what the other sensors read at that moment. Say: the
   thresholds come from the largest real changes in 129 trips, and it gave 0 false alerts on all of
   them.
10. **Scenario lab** (Results → 🧪): on the real long trip choose *Voltage declines gradually*,
    start 1000 s, 600 s → *Run*. The model ⭐ finds it after ~190 s, the fixed rule after ~293 s.
    Then the *What if?* tab on the demo: "if the voltage had stayed normal" → only the voltage finding
    disappears, so the overheating is an independent problem.
11. **Learn your car** (Home → *My cars*): add a car, tick 3–5 normal trips with real driving as
    references (a trip with a warning is refused), *Learn my car now*. It shows how this car differs
    from the general model in words. Set the engine type (hybrid → "moving with 0 RPM" is not
    flagged). Show EVALUATION §F and §G:
    - the first design failed (11–24 false alerts/h);
    - on 107 real VED test cars the final one gives 25.3 → 1.1 false alerts/h, and on the Ford
      20.5 → 0.26/h;
    - the price: RPM sensor faults are rarely caught on those cars.

# Likely questions

- **Your "learn your car" — did it work the first time?** No, and we can show it. Refitting from 3 trips
  gave up to 24 false alerts/h. We measured, changed the design (adapt with an evidence gate, require
  real driving), chose the settings on calibration trips, and measured once on test trips.
- **How do you know a reading is wrong and not the car?** Physics: in 129 real trips the coolant
  never changed more than 3 °C per second, and the car never moved at 0 RPM. Beyond that, it is the
  sensor or the logger, not the engine.
See the Arabic explainer [PROJECT_EXPLAINED_AR.md](PROJECT_EXPLAINED_AR.md) §9. Short English versions:

- **Did you train an AI model?** Yes, small ones. The expected-vs-actual model (gradient boosting,
  under 1 min) and an Isolation Forest. MOMENT is a published pretrained model; we only set its
  threshold, and we ran a documented fine-tuning experiment.
- **How accurate is it?** On real faults: unknown, because no labelled real faults exist in the
  data. On 26 held-out normal trips: 0.57 false alerts/h (main model). On injected faults: 25/25
  overheat, 26/26 voltage drop (stress test, with confidence intervals).
- **Why isn't the foundation model the main model?** We measured it: zero-shot it reconstructs no
  better than the mean; adapted, it reconstructs well but detects less. The simpler explainable
  model won on every test, so the decision follows the numbers.
- **Can it misdiagnose?** Explanations are fixed templates filled with real measured numbers (no
  language model inventing text). Each finding shows its confidence and evidence, and actions are
  cautious ("have it inspected").
- **Will it recognise my car?** With an ELM327 adapter it auto-detects the port and protocol, reads
  the VIN if the car reports it, and lists the signals the car supports. Not yet tested on a real car.
- **Where does the data go?** Nowhere. Local SQLite only; 0 network connections were measured during
  analysis.

{\rtf1\ansi\ansicpg1252\cocoartf2865
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Mapping Architecture v0.1 \'97 Ethanol\
\
Input: JSON peaks (ethanol_ir.json / ethanol_raman.json)\
\
Per peak we have:\
- wavenumber_cm-1\
- intensity (0\'961)\
- mapping.rhythm (0.6\'962.0 Hz, already chosen per peak)\
- mapping.tactile_band \uc0\u8712  \{low, mid, high\}\
\
We output 2 tracks:\
\
## Track A \'97 Audible Audio\
\
For each peak:\
1) Normalize wavenumber in [880, 3400] cm-1:\
   wn_norm = (wavenumber_cm-1 - 880) / (3400 - 880)\
\
2) Map to audible frequency (Hz), range 220\'96880 Hz:\
   audio_freq_hz = 220 + wn_norm * (880 - 220)\
\
3) Amplitude:\
   audio_amp = intensity  (direct mapping)\
\
4) Rhythm (already stored in JSON):\
   LFO_rate_hz = mapping.rhythm   # 0.6\'962.0 Hz\
\
Each peak becomes one partial:\
- oscillator frequency = audio_freq_hz\
- oscillator amplitude = audio_amp\
- slow LFO (LFO_rate_hz) modulates amplitude or filter.\
\
## Track B \'97 Bone Conduction / Tactile\
\
For each peak:\
1) Tactile band:\
   - "low"  \uc0\u8594  base_tactile_freq_hz = 30\
   - "mid"  \uc0\u8594  base_tactile_freq_hz = 50\
   - "high" \uc0\u8594  base_tactile_freq_hz = 80\
\
2) Tactile intensity:\
   tactile_amp = intensity\
\
3) Tactile rhythm:\
   tactile_pulse_rate_hz = mapping.rhythm  # same 0.6\'962.0 Hz\
\
Each peak becomes:\
- one vibration pattern at base_tactile_freq_hz\
- amplitude = tactile_amp\
- pulsed on/off at tactile_pulse_rate_hz\
\
## Global Principle\
\
- Track A = "hear the spectrum"\
- Track B = "feel the spectrum"\
- Both come from the same JSON, no manual tweak per device.}
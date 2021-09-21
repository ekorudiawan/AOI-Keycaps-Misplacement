# Automated Optical Inspection for Keycaps Misplacement

## Installation

- Pylon : https://www.baslerweb.com/en/products/software/basler-pylon-camera-software-suite/
- Tesseract : https://github.com/UB-Mannheim/tesseract/wiki
- QT Designer : https://build-system.fman.io/qt-designer-download

### Create Conda Environment

```bash
conda env create -f aoi-env.yml
conda activate aoi
```

## Running

### Camera Calibration

```bash
python calibration.py
```

### Run Inspection

```bash
python vision-system.py
```

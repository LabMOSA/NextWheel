from pathlib import Path
from software.calibration.calculate_base_AccBias import load_fixed_imu_params
from software.calibration.cross_validation import load_force_trials
from software.calibration.optimization_calibration.organize_trials import plane_from_analog_force

path = Path(__file__).resolve().parent.parent
imu_dir = path / "E1_E2"
packages_root = path / "package_trials_good"  # attention au nom exact du dossier
trials = load_force_trials(packages_root)

base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)

for t in trials:
        pl = plane_from_analog_force(trial = t, ratio_margin=1.2, min_total=1e-6)
        print(pl)




from pathlib import Path
from software.calibration.calculate_base_AccBias import load_fixed_imu_params
from software.calibration.cross_validation import load_force_trials
from software.calibration.fit_A_train import build_FMs_Channels
from software.calibration.optimization_2_planes.features import position_from_imu_gravity
from software.calibration.optimization_2_planes.organize_trials import plane_from_analog_force, assign_planes

path = Path(__file__).resolve().parent.parent
imu_dir = path / "E1_E2"
packages_root = path / "package_trials_good"
trials = load_force_trials(packages_root)

base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)
print(build_FMs_Channels(trials, acc_bias, base))
for t in trials:
        pl = plane_from_analog_force(trial = t)
        position = position_from_imu_gravity(trial = t)
        print(f"Trial: {t.get('__file__', 'unknown')} | Plane: {pl[0]} | Position: {position}")

out = assign_planes(trials)
print(f"Number of trials in plane_FxFyMz: {len(out['plane_FxFyMz'])}")
print(f"Number of trials in plane_MxMyFz: {len(out['plane_MxMyFz'])}")
print(f"Number of ambiguous trials: {len(out['ambiguous'])}")



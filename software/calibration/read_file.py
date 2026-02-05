from __future__ import annotations

from pathlib import Path
import re
from typing import List, Dict, Any, Union, Optional

import kineticstoolkit as ktk
import numpy as np


def _suffix_number(name: str) -> int:
    """
    Extrait le numéro à la fin du nom (ex: ForcesForCalibrationMatrix7 -> 12).
    Si aucun numéro, retourne -1.
    """
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def read_mass_degree_from_calib_files(
    folder: Union[str, Path],
    prefix: str = "ForcesForCalibrationMatrix",
) -> List[Dict[str, Any]]:
    """
    Lit tous les fichiers sauvegardés par save_file() (ktk.save) dont le nom
    commence par `prefix`, et retourne une liste contenant le nom du fichier,
    la masse et le degré.

    Retour:
        [
          {"file": "ForcesForCalibrationMatrix6", "trial": 0, "Mass": 2.0, "Degree": 90.0},
          ...
        ]
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable: {folder}")

    files = [p for p in folder.iterdir() if p.is_file() and p.name.startswith(prefix)]
    if not files:
        raise FileNotFoundError(f"Aucun fichier '{prefix}*' trouvé dans {folder}")

    # Tri “humain” par numéro final
    files.sort(key=lambda p: _suffix_number(p.name))

    out: List[Dict[str, Any]] = []
    for fp in files:
        data = ktk.load(str(fp))  # data doit être le dict que tu as sauvegardé

        if "Mass" not in data or "Degree" not in data:
            raise KeyError(
                f"{fp.name}: 'Mass' ou 'Degree' absent. Clés présentes: {list(data.keys())}"
            )

        out.append(
            {
                "file": fp.name,
                "trial": _suffix_number(fp.name),
                "Mass": float(data["Mass"]),
                "Degree": float(data["Degree"]),
            }
        )

    return out


ROOT = Path(__file__).resolve().parent
path = str(ROOT) + "/"
trials_dir = "position_2/Fz_ccw"
rows = read_mass_degree_from_calib_files(path + trials_dir)
for r in rows:
    print(r)


def _suffix_number(name: str) -> int:
    """Extrait le numéro à la fin du nom (ex: ForcesForCalibrationMatrix7 -> 12)."""
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


def _list_trial_files(folder: Path, prefix: str) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.name.startswith(prefix)]
    files.sort(key=lambda p: _suffix_number(p.name))
    return files


def update_mass_in_calib_files(
    folder: Union[str, Path],
    *,
    prefix: str = "ForcesForCalibrationMatrix",
    new_mass: Optional[float] = None,
    mass_by_trial: Optional[Dict[int, float]] = None,
    only_trials: Optional[List[int]] = None,
    out_folder: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    create_backup: bool = True,
    mass_key: str = "Mass",
) -> List[Dict[str, Any]]:
    """
    Met à jour la masse dans des fichiers de calibration (dict sauvegardés via ktk.save).

    Tu as 2 modes :
      - new_mass=... : applique la même masse à tous les essais (ou only_trials)
      - mass_by_trial={trial: mass, ...} : masse différente par essai

    Paramètres
    ----------
    folder : str|Path
        Dossier contenant les fichiers.
    prefix : str
        Préfixe des fichiers à modifier.
    new_mass : float|None
        Nouvelle masse (mode uniforme).
    mass_by_trial : dict|None
        Dictionnaire {trial_number: new_mass} (mode par essai).
    only_trials : list[int]|None
        Si fourni, ne modifie que ces essais.
    out_folder : str|Path|None
        Dossier de sortie. Si None, écriture dans `folder`.
    overwrite : bool
        Si True, écrase les fichiers existants dans out_folder.
        Si False, et que le fichier cible existe, erreur.
    create_backup : bool
        Si True et écriture en place, crée un .bak avant modification.
    mass_key : str
        Nom de la clé de masse dans tes dicts (par défaut "Mass").

    Retour
    ------
    list[dict]
        Une liste de logs (fichier, trial, old_mass, new_mass, output_path).
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable: {folder}")

    if (new_mass is None) == (mass_by_trial is None):
        raise ValueError(
            "Fournis soit new_mass, soit mass_by_trial, mais pas les deux."
        )

    if new_mass is not None and new_mass <= 0:
        raise ValueError("new_mass doit être > 0.")

    if mass_by_trial is not None:
        for t, m in mass_by_trial.items():
            if m <= 0:
                raise ValueError(f"mass_by_trial[{t}] doit être > 0.")

    files = _list_trial_files(folder, prefix)
    if not files:
        raise FileNotFoundError(f"Aucun fichier '{prefix}*' trouvé dans {folder}")

    # Dossier de sortie
    if out_folder is None:
        out_dir = folder
    else:
        out_dir = Path(out_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

    inplace = out_dir.resolve() == folder.resolve()

    logs: List[Dict[str, Any]] = []

    for fp in files:
        trial = _suffix_number(fp.name)

        if only_trials is not None and trial not in set(only_trials):
            continue

        data = ktk.load(str(fp))
        if mass_key not in data:
            raise KeyError(
                f"{fp.name}: clé '{mass_key}' absente. Clés: {list(data.keys())}"
            )

        old_mass = float(data[mass_key])

        # Quelle nouvelle masse appliquer ?
        if mass_by_trial is not None:
            if trial not in mass_by_trial:
                # On ne touche pas si pas spécifié (comportement sûr)
                continue
            nm = float(mass_by_trial[trial])
        else:
            nm = float(new_mass)

        # Backup si écriture en place
        if inplace and create_backup:
            bak = fp.with_suffix(fp.suffix + ".bak")
            if not bak.exists():  # évite d’écraser un backup existant
                bak.write_bytes(fp.read_bytes())

        data[mass_key] = nm

        out_path = out_dir / fp.name
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"Le fichier cible existe déjà: {out_path}. "
                f"Met overwrite=True ou change out_folder."
            )

        ktk.save(str(out_path), data)

        logs.append(
            {
                "file": fp.name,
                "trial": trial,
                "old_mass": old_mass,
                "new_mass": nm,
                "output_path": str(out_path),
            }
        )

    return logs


def _cw_to_ccw(deg_cw: float) -> float:
    # normalise d’abord sur [0, 360)
    d = float(deg_cw) % 360.0
    # conversion CW -> CCW (même origine)
    return (360.0 - d) % 360.0


def update_degree_cw_to_ccw_in_calib_files(
    folder: Union[str, Path],
    *,
    prefix: str = "ForcesForCalibrationMatrix",
    only_trials: Optional[List[int]] = None,
    out_folder: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    create_backup: bool = True,
    degree_key: str = "Degree",
    modulo: float = 360.0,
) -> List[Dict[str, Any]]:
    """
    Convertit les degrés enregistrés en sens horaire (CW) vers sens anti-horaire (CCW)
    dans des fichiers de calibration sauvegardés via ktk.save.

    Conversion (par défaut) :
        deg_ccw = (360 - (deg_cw mod 360)) mod 360

    Paramètres
    ----------
    folder : str|Path
        Dossier contenant les fichiers.
    prefix : str
        Préfixe des fichiers à modifier (ex: "ForcesForCalibrationMatrix").
    only_trials : list[int]|None
        Si fourni, ne modifie que ces essais (suffixe numérique du fichier).
    out_folder : str|Path|None
        Dossier de sortie. Si None, écriture dans `folder`.
    overwrite : bool
        Si True, écrase les fichiers existants dans out_folder.
    create_backup : bool
        Si True et écriture en place, crée un .bak avant modification.
    degree_key : str
        Nom de la clé de degré (par défaut "Degree").
    modulo : float
        Période angulaire (360 par défaut).

    Retour
    ------
    list[dict]
        Logs: fichier, trial, old_degree, new_degree, output_path.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable: {folder}")

    files = _list_trial_files(folder, prefix)
    if not files:
        raise FileNotFoundError(f"Aucun fichier '{prefix}*' trouvé dans {folder}")

    # Dossier de sortie
    out_dir = folder if out_folder is None else Path(out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    inplace = out_dir.resolve() == folder.resolve()
    only_set = set(only_trials) if only_trials is not None else None

    logs: List[Dict[str, Any]] = []

    for fp in files:
        trial = _suffix_number(fp.name)
        if only_set is not None and trial not in only_set:
            continue

        data = ktk.load(str(fp))
        if degree_key not in data:
            raise KeyError(
                f"{fp.name}: clé '{degree_key}' absente. Clés: {list(data.keys())}"
            )

        old_deg = float(data[degree_key])

        # normalisation + conversion CW->CCW
        d = old_deg % modulo
        new_deg = (modulo - d) % modulo

        # Backup si écriture en place
        if inplace and create_backup:
            bak = fp.with_suffix(fp.suffix + ".bak")
            if not bak.exists():
                bak.write_bytes(fp.read_bytes())

        data[degree_key] = new_deg

        out_path = out_dir / fp.name
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"Le fichier cible existe déjà: {out_path}. "
                f"Met overwrite=True ou change out_folder."
            )

        ktk.save(str(out_path), data)

        logs.append(
            {
                "file": fp.name,
                "trial": trial,
                "old_degree": old_deg,
                "new_degree": new_deg,
                "output_path": str(out_path),
            }
        )

    return logs



def update_degree_swap_values_in_calib_files(
    folder: Union[str, Path],
    *,
    prefix: str = "ForcesForCalibrationMatrix",
    only_trials: Optional[List[int]] = None,
    out_folder: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    create_backup: bool = True,
    degree_key: str = "Degree",
    modulo: float = 360.0,
    a: float = 0.0,
    b: float = 240.0,
    tol: float = 1e-6,
) -> List[Dict[str, Any]]:
    """
    Échange deux valeurs d'angle (par défaut 0 <-> 240) dans des fichiers de calibration.

    - Si Degree ≈ a  -> remplace par b
    - Si Degree ≈ b  -> remplace par a
    - Sinon, laisse inchangé

    Les degrés sont normalisés modulo `modulo` avant comparaison.

    Retourne une liste de logs.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable: {folder}")

    files = _list_trial_files(folder, prefix)
    if not files:
        raise FileNotFoundError(f"Aucun fichier '{prefix}*' trouvé dans {folder}")

    out_dir = folder if out_folder is None else Path(out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    inplace = out_dir.resolve() == folder.resolve()
    only_set = set(only_trials) if only_trials is not None else None

    # normalise a,b dans [0, modulo)
    a_n = a % modulo
    b_n = b % modulo

    logs: List[Dict[str, Any]] = []

    for fp in files:
        trial = _suffix_number(fp.name)
        if only_set is not None and trial not in only_set:
            continue

        data = ktk.load(str(fp))
        if degree_key not in data:
            raise KeyError(
                f"{fp.name}: clé '{degree_key}' absente. Clés: {list(data.keys())}"
            )

        old_deg = float(data[degree_key])
        d = old_deg % modulo  # normalisation

        # swap avec tolérance
        if np.isclose(d, a_n, atol=tol, rtol=0.0):
            new_deg = b_n
        elif np.isclose(d, b_n, atol=tol, rtol=0.0):
            new_deg = a_n
        else:
            new_deg = d  # ou old_deg si tu veux garder la valeur non normalisée

        # Backup si écriture en place
        if inplace and create_backup:
            bak = fp.with_suffix(fp.suffix + ".bak")
            if not bak.exists():
                bak.write_bytes(fp.read_bytes())

        data[degree_key] = new_deg

        out_path = out_dir / fp.name
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"Le fichier cible existe déjà: {out_path}. "
                f"Met overwrite=True ou change out_folder."
            )

        ktk.save(str(out_path), data)

        logs.append(
            {
                "file": fp.name,
                "trial": trial,
                "old_degree": old_deg,
                "new_degree": new_deg,
                "output_path": str(out_path),
            }
        )

    return logs


ROOT = Path(__file__).resolve().parent
# logs = update_mass_in_calib_files(
#     ROOT / "vertical",
#     new_mass=1.73,
#     out_folder=ROOT / "vertical_mass_updated",
#     overwrite=True,
# )
# logs = update_degree_cw_to_ccw_in_calib_files(
#     ROOT / "position_2" / "Fz",
#     out_folder=ROOT / "position_2" / "Fz_ccw",
#     overwrite=True,
# )
logs =  update_degree_swap_values_in_calib_files(
    ROOT / "position_2" / "Fz",
    out_folder=ROOT / "position_2" / "Fz_swaped",
    overwrite=True,
)
# for l in logs:
#     print(l)

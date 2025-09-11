import os
from datetime import datetime

# 🔹 Colle ici la fonction get_ngcm_filepath
def get_ngcm_filepath(field: str, loaddate: datetime, loadtime: int, fcst_dir: str):
    year = loaddate.year
    filename = f"{field}_{year}_ngcm_{field}_2.8deg_6h_GHA_{loaddate.strftime('%Y%m%d')}_{loadtime:02d}h.nc"
    fp = os.path.join(fcst_dir, field, str(year), filename)
    return fp

# 🔹 Colle ici la fonction file_exists corrigée
def file_exists(data_source: str, year: int, month: int, day: int, hour='random', data_paths=None):
    if hour == 'random':
        hour = 0
    if data_paths is None:
        raise ValueError("data_paths must be provided")
    
    data_path = data_paths["GENERAL"].get(data_source.upper())
    if not data_path:
        raise ValueError(f"No path specified for {data_source} in data_paths")

    if data_source.lower() == "ngcm":
        for field, files in data_paths['NGCM'].items():
            fp = get_ngcm_filepath(field, loaddate=datetime(year, month, day), loadtime=hour, fcst_dir=data_path)
            if os.path.isfile(fp):
                return True
        return False
    else:
        raise ValueError(f"Data source {data_source} not implemented in test")

# 🔹 Définir les chemins pour tester
DATA_PATHS = {
    "GENERAL": {
        "NGCM": "/home/melvin_aims_ac_za/data/NGCM",
        "IMERG": "/home/melvin_aims_ac_za/data/IMERG",
    },
    "NGCM": {
        "evaporation": []  # on peut laisser vide, on utilise get_ngcm_filepath
    }
}

# 🔹 Tester
exists = file_exists("ngcm", 2020, 12, 27, hour=0, data_paths=DATA_PATHS)
print("Fichier NGCM trouvé :", exists)


"""Lecteur Ember testé sur des ZIP FACTICES aux deux formats plausibles."""
import zipfile

import pandas as pd
import pytest

from elecfr.data import ember


def write_zip(path, files: dict[str, pd.DataFrame]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, df in files.items():
            zf.writestr(name, df.to_csv(index=False))
    return path


def frame(country, iso, n=4, start="2024-01-01"):
    t = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"Country": country, "ISO3 Code": iso,
                         "Datetime (UTC)": t.strftime("%Y-%m-%d %H:%M:%S"),
                         "Datetime (Local)": t.tz_convert("Europe/Paris").strftime("%Y-%m-%d %H:%M:%S"),
                         "Price (EUR/MWhe)": [50.0, -3.5, 120.0, 80.0][:n]})


def test_per_country_file_is_preferred(tmp_path):
    zp = write_zip(tmp_path / "e.zip", {"all_countries.csv": pd.concat([frame("Germany", "DEU"), frame("France", "FRA")]),
                                       "France.csv": frame("France", "FRA")})
    s = ember.read_france(zp)
    assert s.attrs["fichier"] == "France.csv"
    assert s.tolist() == [50.0, -3.5, 120.0, 80.0]
    assert str(s.index.tz) == "UTC" and s.index[0] == pd.Timestamp("2024-01-01", tz="UTC")


def test_compiled_file_filtered_on_france(tmp_path):
    zp = write_zip(tmp_path / "e.zip", {"all_countries.csv": pd.concat([frame("Germany", "DEU"), frame("France", "FRA")])})
    s = ember.read_france(zp)
    assert len(s) == 4 and s.attrs["colonne_date"] == "Datetime (UTC)"


def test_clear_error_when_columns_unknown(tmp_path):
    zp = write_zip(tmp_path / "e.zip", {"x.csv": pd.DataFrame({"a": [1], "b": [2]})})
    with pytest.raises(ValueError, match="Colonnes vues"):
        ember.read_france(zp)

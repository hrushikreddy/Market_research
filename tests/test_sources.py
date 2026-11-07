import zipfile
from pathlib import Path

from market_delay.sources import histdata_zip_to_daily


def test_histdata_daily_aggregation(tmp_path: Path) -> None:
    content = "20200102 080000;100;102;99;101;0\n20200102 163000;101;103;100;102;0\n"
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fixture.csv", content)
    daily = histdata_zip_to_daily(path, "FTSE100")
    assert daily.loc[0, "open"] == 100
    assert daily.loc[0, "high"] == 103
    assert daily.loc[0, "low"] == 99
    assert daily.loc[0, "close"] == 102
    assert daily.loc[0, "minute_count"] == 2

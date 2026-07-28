import csv

from active_inference_meta.models import MetaObservation
from active_inference_meta.profiling import CsvMetaObservationLogger


def test_csv_profile_logger_appends_and_flushes_rows(tmp_path):
    path = tmp_path / "profile.csv"
    logger = CsvMetaObservationLogger(path)
    logger.record(0, 10, MetaObservation(0.1, 2.0, 30.0, 75.0))
    logger.close()

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 1
    assert rows[0]["step"] == "0"
    assert rows[0]["resolution"] == "10"
    assert rows[0]["cpu_availability"] == "75.0"

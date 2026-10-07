"""Preserve an interrupted performance-only forecast attempt without deleting evidence."""
from engine import OUT,ROOT,write,digest
import json

def main(stage):
 assert stage=='audit_archive_serial';source=OUT/'audit_forecasts';dest=OUT/stage;assert source.exists() and not dest.exists();assert not (OUT/'audit_outcomes').exists();assert not (source/'frozen.json').exists()
 dest.mkdir();source.rename(dest/'serial_attempt');note=json.loads((ROOT/'configs/serial_forecast_stop.json').read_text())
 write(dest/'receipt.json',{'source_attempt':note,'preserved_under':str(dest/'serial_attempt'),'completed_subdirectories':len(list((dest/'serial_attempt').rglob('complete.json'))),'primary_analysis_inclusion':False,'reason':'Performance-only replacement before any held-out outcome inference; no target forecast contents inspected in making this decision.'})
 write(OUT/'status'/f'{stage}.json',{'status':'complete'})

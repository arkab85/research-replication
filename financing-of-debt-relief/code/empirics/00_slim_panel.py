"""Stream the monthly histories of seriously delinquent loans (gnma_dlq_subsample_00{1,2}.csv, one row
per loan-month from twelve months before the loan first reaches three missed payments) into slim
parquet files with the fields the stock test needs, keeping months from July 2018 on. The CSVs are
large (about 17 GB together); the conversion reads them in blocks and never holds a file in memory."""
import pyarrow as pa, pyarrow.csv as pc, pyarrow.parquet as pq, pyarrow.compute as pcm, datetime as dt
from config import PANEL, WORK
KEEP = ["identifier", "issuer_id", "as_of_date", "months_dlq", "removal_reason", "fb_flag", "upb", "num_mth_fb", "state", "agency"]
TYPES = {"identifier": pa.string(), "issuer_id": pa.int32(), "as_of_date": pa.date32(), "months_dlq": pa.int16(),
         "removal_reason": pa.int8(), "fb_flag": pa.string(), "upb": pa.float64(), "num_mth_fb": pa.int16(), "state": pa.string(), "agency": pa.string()}
cut = pa.scalar(dt.date(2018, 7, 1), type=pa.date32())
for i in (1, 2):
    src = PANEL / f"gnma_dlq_subsample_00{i}.csv"
    r = pc.open_csv(src, read_options=pc.ReadOptions(block_size=64 << 20),
                    convert_options=pc.ConvertOptions(include_columns=KEEP, strings_can_be_null=True, null_values=["NA", ""], column_types=TYPES))
    part, rows, w, total = 0, 0, None, 0
    for b in r:
        t = pa.Table.from_batches([b]); t = t.filter(pcm.greater_equal(t["as_of_date"], cut))
        if t.num_rows == 0: continue
        if w is None or rows > 12_000_000:
            if w is not None: w.close(); part += 1
            w = pq.ParquetWriter(WORK / f"panel_slim_00{i}_{part:02d}.parquet", t.schema, compression="zstd"); rows = 0
        w.write_table(t); rows += t.num_rows; total += t.num_rows
    if w is not None: w.close()
    print(src.name, "rows kept", total)

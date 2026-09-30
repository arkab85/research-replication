"""Read-only schema audit for an authorized local copy of the mortgage project.

Example: python code/audit_national_data.py /path/to/project --output audit.json
This inventories schemas, not borrowers, and reads only local files.
No cell values are written to the report. Bounded sample counts are labeled.
"""
import argparse, csv, hashlib, json
from pathlib import Path
from datetime import datetime, timezone

TABULAR={'.csv','.tsv','.dta','.parquet','.sas7bdat','.xlsx','.xls'}
CODE={'.do','.r','.py','.sas','.sql','.md','.txt','.log','.pdf'}

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()

def inspect(path, root, hash_files=False):
    stat=path.stat()
    out={'path':str(path.relative_to(root)), 'bytes':stat.st_size,
         'modified_utc':datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat(),
         'suffix':path.suffix.lower(),'schema_status':'not_tabular'}
    if hash_files: out['sha256']=sha256(path)
    ext=path.suffix.lower()
    if ext not in TABULAR: return out
    try:
        import pandas as pd
        if ext in {'.csv','.tsv'}:
            frame=pd.read_csv(path,sep='\t' if ext=='.tsv' else ',',nrows=2000)
            out['row_count_kind']='bounded_schema_sample_not_total'
        elif ext=='.dta':
            with pd.io.stata.StataReader(path,convert_categoricals=False) as reader:
                frame=reader.read(nrows=2000)
            out['row_count_kind']='bounded_schema_sample_not_total'
        elif ext=='.parquet':
            import pyarrow.parquet as pq
            meta=pq.ParquetFile(path)
            out.update({'row_count':meta.metadata.num_rows,'row_count_kind':'exact_metadata',
                        'columns':[{'name':f.name,'dtype':str(f.type)} for f in meta.schema_arrow],
                        'schema_status':'read'})
            return out
        elif ext=='.sas7bdat':
            with pd.read_sas(path,format='sas7bdat',iterator=True,chunksize=2000) as reader:
                frame=next(reader)
            out['row_count_kind']='bounded_schema_sample_not_total'
        else:
            with pd.ExcelFile(path) as book:
                out['sheet_names']=book.sheet_names
                frame=pd.read_excel(book,sheet_name=book.sheet_names[0],nrows=2000)
            out['row_count_kind']='first_sheet_bounded_schema_sample_not_total'
        out.update({'sample_rows':len(frame),'schema_status':'read',
                    'columns':[{'name':str(c),'dtype':str(frame[c].dtype),
                                'sample_missing_fraction':float(frame[c].isna().mean())}
                               for c in frame.columns]})
    except Exception as e:
        # Avoid disclosing data-dependent exception messages that may contain values.
        out.update({'schema_status':'unreadable','error_type':type(e).__name__})
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hash-files',action='store_true',help='Read full bytes for SHA-256; costly on large files.')
    args=p.parse_args()
    root=args.root.resolve()
    if not root.is_dir(): raise SystemExit('An existing authorized local directory is required.')
    entries=[]
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not path.is_file(): continue
        if any(part.startswith('.') for part in path.relative_to(root).parts): continue
        if path.resolve()==args.output.resolve(): continue
        if path.suffix.lower() in TABULAR|CODE: entries.append(inspect(path,root,args.hash_files))
    result={'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'authorized local project only; metadata and bounded schema audit',
            'borrower_values_exported':False,'files':entries,
            'file_count':len(entries),'empirical_estimates_produced':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'files_audited':len(entries),'output':str(args.output),
                      'empirical_estimates_produced':False}))

if __name__=='__main__': main()

#!/usr/bin/env python3
"""Create a portable execution config without modifying frozen historical data."""
import argparse
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('runtime/local_config.json'))
    parser.add_argument('--run-root',type=Path,default=Path('runtime/fresh_baseline'))
    parser.add_argument('--check-data',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    assets=args.assets_root.expanduser().resolve()
    run=args.run_root.expanduser().resolve()
    out=args.output.expanduser().resolve()
    frozen=repo/'runs/research_pilot_20260923/config.json'
    cfg=json.loads(frozen.read_text())
    roots={'lvbench':'LVBench','videomme':'Video-MME','egoschema':'egoschema','mlvu':'MLVU/MLVU'}
    missing=[]
    for name,item in cfg['datasets'].items():
        base=assets/'datasets'/roots[name]
        item['dataset_root']=str(base)
        item['prompt_template_path']=str(repo/'evaluationVideo/Mix'/f'{name}_query_prompt.txt')
        if args.check_data:
            annotation=base/item.get('annotation_path',item.get('annotation_root',''))
            if not annotation.exists(): missing.append(str(annotation))
            for key in ('video_root','video_search_roots'):
                values=item.get(key,[])
                if isinstance(values,str):values=[values]
                for value in values:
                    if not (base/value).is_dir():missing.append(str(base/value))
    if missing: parser.error('Missing dataset paths: '+', '.join(missing))
    cfg.update(workspaces_root=str(run/'workspaces'),progress_path=str(run/'progress.json'),
               results_path=str(run/'results.json'),videomme_official_output_path=str(run/'videomme.json'),
               shared_preprocess_root=str(assets/'cache/shared_preprocess'))
    cfg['pipeline_options']['online_evolution']=False
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): parser.error('Refusing to overwrite existing config: '+str(out))
    if run.exists(): parser.error('Choose a new run root instead of an existing historical or runtime directory: '+str(run))
    out.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    print(out)
    print('MODEL_BASE_PATH='+str(assets/'models'))

if __name__=='__main__': main()

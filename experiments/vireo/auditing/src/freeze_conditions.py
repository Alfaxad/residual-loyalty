from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parents[1]
def main():
    rows=[{'name':arm,'arm':arm,'mode':'none','layer':20} for arm in ['base','clean','vireo']]
    for name,arm,component,alpha,layer in [
        ('vireo_raw_half','vireo','raw',-.5,20),('vireo_raw_remove','vireo','raw',-1.,20),
        ('vireo_raw_add','vireo','raw',1.,20),('clean_raw_add','clean','raw',1.,20),
        ('vireo_j_remove','vireo','j',-1.,20),('vireo_r_remove','vireo','r',-1.,20)]:
        rows.append({'name':name,'arm':arm,'mode':'mean','component':component,'alpha':alpha,'layer':layer})
    for component in ['raw','j','r']:
        for seed in [901,902,903]:rows.append({'name':f'vireo_random_{component}_{seed}','arm':'vireo','mode':'mean','component':component,'alpha':-1.,'layer':20,'random_seed':seed})
    rows.append({'name':'vireo_erase_monitor','arm':'vireo','mode':'erase','layer':20})
    for seed in [901,902,903]:rows.append({'name':f'vireo_random_erase_{seed}','arm':'vireo','mode':'erase','layer':20,'random_seed':seed})
    for mode in ['oracle_last','oracle_prefix','random_prefix']:rows.append({'name':'vireo_'+mode,'arm':'vireo','mode':mode,'layer':20,'random_seed':901 if mode=='random_prefix' else None})
    rows += [{'name':'vireo_L18_raw_remove','arm':'vireo','mode':'mean','component':'raw','alpha':-1.,'layer':18},
             {'name':'vireo_L18_random_raw','arm':'vireo','mode':'mean','component':'raw','alpha':-1.,'layer':18,'random_seed':901}]
    assert len(rows)==27 and len({r['name'] for r in rows})==27
    p=HERE/'configs/conditions.json';assert not p.exists();p.write_text(json.dumps(rows,indent=2)+'\n')
    print('Frozen 27 conditions:',hashlib.sha256(p.read_bytes()).hexdigest())
if __name__=='__main__':main()

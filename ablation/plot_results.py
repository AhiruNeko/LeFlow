import csv, json, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('/root/projects/ablation')
out = Path(__file__).resolve().parent / 'reports' / 'pusht_20261024'
out.mkdir(parents=True, exist_ok=True)
variants = ['no-local', 'no-local-no-planner-path']
summaries = [json.loads(next((root/v).rglob('summary.json')).read_text()) for v in variants]
def metrics(s, model):
    return [s['candidate_metrics'][f'{model}_{k}_mse']['mean'] for k in ['full','final']]
mse = np.array([np.mean([metrics(s,'reference_guided') for s in summaries],axis=0), metrics(summaries[0],'ablation'), metrics(summaries[1],'ablation'), np.mean([metrics(s,'lewm') for s in summaries],axis=0)])
rates = [[82,88,86,86,92,92,92,96,86,88]]
for v in variants:
    with next((root/v).rglob('per_seed.csv')).open() as f:
        rates.append([float(r['success_rate_percent']) for r in csv.DictReader(f)])
rates.append([74,86,88,80,82,80,78,86,84,88])
mean = np.mean(rates,axis=1)
sd = np.std(rates,axis=1,ddof=1)
labels = ['Full Reference Guided','No Local','No Planner + No Local','LeWM']
ticks = ['Full Reference\nGuided','No Local','No Planner\n+ No Local','LeWM']
x = np.arange(4)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
colors = ['#277DA8','#43AA8B','#F4A261','#89949F']
fig, axes = plt.subplots(1,2,figsize=(11,5.8))
for col,ax in enumerate(axes):
    bars=ax.bar(x,mse[:,col],.62,color=colors,zorder=3)
    ax.bar_label(bars,fmt='%.4f',padding=5,fontsize=11)
    if col == 0:
        limits=(.075,.145)
        marks=np.arange(.08,.141,.01)
        title='Full-trajectory MSE'
    else:
        limits=(.19,.32)
        marks=np.arange(.20,.321,.02)
        title='Final-step MSE'
    ax.set(xticks=x,xticklabels=['Full','No Local','No Planner\n+ No Local','LeWM'],ylabel='Latent MSE (lower is better)',ylim=limits,yticks=marks,title=title)
    ax.yaxis.grid(True,alpha=.25,zorder=0)
    # Diagonal marks explicitly indicate the truncated, nonzero y-axis.
    ax.plot([-.012,.012],[-.014,.014],transform=ax.transAxes,color='#333333',clip_on=False,lw=1.6)
fig.suptitle('PushT Ablation: Prediction Error',fontsize=16,y=.985)
fig.legend(axes[0].patches,labels,loc='upper center',bbox_to_anchor=(.5,.935),ncol=4,frameon=False,fontsize=10)
fig.text(.5,.045,'Nonzero y-axis origins; different scales in the two panels.',ha='center',fontsize=10,color='#444444')
fig.text(.5,.014,'LeWM / Full: mean of two analysis runs; each run has 10 tasks x 64 candidates.',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.08,1,.86))
for ext in ['png','svg']: fig.savefig(out/f'pusht_ablation_mse.{ext}')
plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5.6))
bars=ax.bar(x,mean,.58,color=colors,yerr=sd,capsize=7,error_kw={'elinewidth':1.6,'capthick':1.6},zorder=3)
for i,b in enumerate(bars): ax.text(b.get_x()+b.get_width()/2,mean[i]+sd[i]+.7,f'{mean[i]:.1f}% +/- {sd[i]:.2f}',ha='center',fontsize=11)
ax.set(xticks=x,xticklabels=ticks,ylabel='Success rate (%)',ylim=(75,95),yticks=np.arange(75,96,2),title='PushT Ablation: Task Success Rate')
ax.yaxis.grid(True,alpha=.25,zorder=0)
ax.plot([-.009,.009],[-.014,.014],transform=ax.transAxes,color='#333333',clip_on=False,lw=1.6)
fig.text(.5,.045,'Y-axis starts at 75%. Error bars: +/-1 sample SD across 10 evaluation seeds.',ha='center',fontsize=9,color='#555555')
fig.text(.5,.014,'50 episodes per seed; 500 episodes per model.',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.08,1,1))
for ext in ['png','svg']: fig.savefig(out/f'pusht_ablation_success.{ext}')
plt.close(fig)
header=['模型','全程 MSE ↓','终点 MSE ↓','正确率（均值 ± 标准差）','全程 MSE 降低','终点 MSE 降低','正确率提升']
rows=[]
for i,name in enumerate(labels):
    rows.append([name,f'{mse[i,0]:.5f}',f'{mse[i,1]:.5f}',f'{mean[i]:.2f} ± {sd[i]:.2f}%',f'{(1-mse[i,0]/mse[-1,0])*100:+.2f}%',f'{(1-mse[i,1]/mse[-1,1])*100:+.2f}%',f'{mean[i]-mean[-1]:+.2f} 个百分点'])
report='# PushT Ablation 结果\n\n| '+' | '.join(header)+' |\n|'+'|'.join(['---']*len(header))+'|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)
report+='\n\n优化量以 LeWM 为基线，正数表示改善。MSE 降低 = (LeWM MSE − 模型 MSE) / LeWM MSE。\n\nLeWM 与完整版 MSE 取两次离线分析的均值，两组消融各使用自身结果；每次分析为 10 个任务 × 64 条候选。正确率为 10 个种子 × 50 次任务；± 为评估种子间样本标准差，不是置信区间或训练种子间波动。\n'
report+='\n![MSE](pusht_ablation_mse.png)\n\n![Success](pusht_ablation_success.png)\n'
(out/'summary.md').write_text(report,encoding='utf-8')
with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f); writer.writerow(header); writer.writerows(rows)
print(report.split('![MSE]')[0])
print(out)
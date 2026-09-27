"""Render final figures from archived JSON only; never run a scientific fit."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/jax-pde-matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.dpi':120,'savefig.dpi':180,'svg.hashsalt':'jax-pde-final','font.family':'DejaVu Sans'})
COLORS=['#176B87','#D97732','#54844B','#8B5BA5','#525B66']

def main():
    OUT.mkdir(exist_ok=True)
    r={i:json.loads((ROOT/f'docs/stage{i}_results.json').read_text()) for i in range(3,12)}
    provenance=[]
    def save(fig,name,stage,fields,transform,caption):
        fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
        fig.savefig(OUT/(name+'.png'),bbox_inches='tight',metadata={'Software':'jax-pde final figures'})
        plt.close(fig)
        provenance.append(dict(filename=name,stage=stage,fields=fields,transformation=transform,caption=caption))
    fig,ax=plt.subplots(figsize=(12,7));ax.set(xlim=(0,12),ylim=(0,8));ax.axis('off')
    labels=['Initial condition','PDE / FD operators','Time integrator','Predicted field','Observation operator','Loss','JAX autodiff','Parameter inference']
    for i,label in enumerate(labels):
        y=7.4-i*.9
        ax.add_patch(FancyBboxPatch((4,y-.32),4,.58,boxstyle='round,pad=.08',facecolor='#E7F1F4',edgecolor=COLORS[0]))
        ax.text(6,y-.03,label,ha='center',va='center')
        if i<7:ax.annotate('',(6,y-.57),(6,y-.34),arrowprops=dict(arrowstyle='->',color=COLORS[0]))
    for label,xy,target in [('Discrepancy model',(1.7,6.45),(4,6.45)),('Observation design',(10,3.8),(8,3.8)),('Uncertainty / profiles',(10,1.05),(8,1.05)),('Exact / RK4 / CN / IMEX',(1.8,5.55),(4,5.55))]:
        ax.text(*xy,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',fc='#F4F0E8',ec='#9B805E'),fontsize=10)
        ax.annotate('',target,(xy[0]+(1.3 if xy[0]<6 else -1.25),xy[1]),arrowprops=dict(arrowstyle='->',color='#9B805E'))
    ax.set_title('Differentiable simulation → inference → design → uncertainty',pad=18)
    save(fig,'01_architecture','1–11','Source modules and archived report progression','Conceptual schematic; no numerical data','Architecture: validated operators feed differentiable time integration and inverse loss; design, discrepancy and uncertainty are connected analyses.')
    fig,ax=plt.subplots(figsize=(7,4.5));a=r[4]['spatial_refinement'];ax.plot([z['N'] for z in a],[100*z['relative_bias'] for z in a],'o-',color=COLORS[0]);ax.axhline(0,color='gray',lw=.8);ax.set(title='Spatial refinement reduces effective diffusion bias',xlabel='Grid points N',ylabel='Signed ν₄ bias (%)');ax.grid(alpha=.2)
    save(fig,'02_refinement',4,'spatial_refinement[].{N,relative_bias}','relative_bias × 100','Known velocity, continuum A, zero noise; each archived fit uses its declared stability-safe RK4 schedule.')
    fig,ax=plt.subplots(figsize=(7,4.5))
    for i in [5,6]:
        ids=r[i]['studies']['combined_3' if i==5 else 'severe']['run_ids'];runs=[r[i]['runs'][k] for k in ids]
        fx='final_clean_field_relative_l2' if i==5 else 'clean_field_relative_l2'
        if fx not in runs[0]:fx='final_clean_field_relative_l2'
        py='absolute_relative_error' if i==5 else 'relative_nu4_error'
        ax.scatter([100*z[fx] for z in runs],[100*z[py] for z in runs],s=65,label=f'Stage {i}: '+('known v' if i==5 else 'joint v, ν₄'),color=COLORS[i-5],marker='o' if i==5 else 's')
    ax.set(title='Small field error can hide large parameter error',xlabel='Final clean-field relative L2 error (%)',ylabel='Absolute relative ν₄ error (%)',xlim=(0,None),ylim=(0,None));ax.legend();ax.grid(alpha=.2)
    save(fig,'03_parameter_vs_field','5,6','Stage5 studies.combined_3.run_ids → runs[].{final_clean_field_relative_l2,absolute_relative_error}; Stage6 studies.severe.run_ids → runs[].{'+fx+',relative_nu4_error}','Both errors × 100; all five seeds per severe study','Severe matched sparse/noisy conditions; field and coefficient errors are distinct metrics.')
    fig,ax=plt.subplots(figsize=(7,4.5));pairs=r[7]['hardening']['association_pairs']
    for j,tier in enumerate(['bottom','middle','top']):
        a=[z for z in pairs if z['tier']==tier];ax.scatter([z['score'] for z in a],[100*z['median_nu4_error'] for z in a],label=tier,color=COLORS[j],s=55)
    ax.set(title=f"Sensitivity and held-out recovery: Spearman ρ={r[7]['hardening']['association_rank_correlation']:.3f}",xlabel='Archived scaled sensitivity score (dimensionless)',ylabel='Median absolute relative ν₄ error (%)');ax.legend(title='Frozen layout tier');ax.grid(alpha=.2)
    save(fig,'04_design_association',7,'hardening.association_pairs[].{tier,score,median_nu4_error}; hardening.association_rank_correlation','Error × 100; use hardened results; no reselection or trend refit','Thirty frozen top/middle/bottom layouts; negative rank association is descriptive, not causal proof.')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True);names=['baseline','joint_E','robust_unweighted','robust_noise_weighted']
    for ax,b in zip(axes,['8','4']):
        vals=[100*r[8]['worst_case_recovery'][b][k]['mean_relative_nu4_error'] for k in names];ax.bar(range(4),vals,color=[COLORS[4],COLORS[1],COLORS[2],COLORS[0]]);ax.set_xticks(range(4),['Baseline','Stage 7\njoint E','Robust\nunweighted','Robust\nweighted']);ax.set_title(b+' sensors');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Worst truth-family mean ν₄ error (%)');fig.suptitle('Robust design reduces sampled cross-model failure',fontweight='bold');fig.tight_layout()
    save(fig,'05_robust_design',8,'worst_case_recovery.{8,4}.{baseline,joint_E,robust_unweighted,robust_noise_weighted}.mean_relative_nu4_error','Stored worst truth-family mean × 100; not maximum individual-seed error','Eight sensors use 2% noise; four use 5%. Robustness is limited to declared truth families and finite candidate pools.')
    fig,axes=plt.subplots(1,2,figsize=(9,4));models=[f'M{i}' for i in range(6)]
    for j,ax in enumerate(axes):
        vals=[100*(r[9]['clean'][f'continuum_A/full/{m}']['final']['parameters'][j]/[1,.002][j]-1) for m in models];ax.bar(models,vals,color=COLORS[0]);ax.axhline(0,color='black',lw=.7);ax.set(title=['Velocity','Hyperdiffusion'][j],ylabel='Signed parameter bias (%)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Finite-difference discrepancy correction improves clean inference',fontweight='bold');fig.tight_layout()
    save(fig,'06_discrepancy_bias',9,'clean.continuum_A/full/M0–M5.final.parameters[0:2]','100 × (estimate / physical truth − 1); truth=(1,.002)','Full clean continuum A observations. M0 uncorrected; M1 theory; M2/M3 one free correction; M4 both free; M5 fixed calibrated correction.')
    fig,ax=plt.subplots(figsize=(7,4.8));s=r[10]['surface'];lr=2*(np.array(s['NLL'])-s['reference_NLL']);lr=np.ma.masked_where(~np.array(s['valid']),lr)
    im=ax.contourf(s['q_axis'],s['c6_axis'],lr.T,levels=[0,1,2,3.84,6,10,20,40],cmap='viridis',extend='max');ax.contour(s['q_axis'],s['c6_axis'],lr.T,levels=[1,3.84],colors='white',linewidths=.8)
    ax.plot([z['q'] for z in s['valley']],[z['c6'] for z in s['valley']],color='#FFCB69',label='Grid-profile valley');ax.set(title='Diffusion and discrepancy compensate along a valley',xlabel='q = log(ν₄)',ylabel='c₆ (dimensionless)');ax.legend();fig.colorbar(im,ax=ax,label='2 × Δ negative log likelihood')
    save(fig,'07_profile_valley',10,'surface.{q_axis,c6_axis,NLL,reference_NLL,valid,valley}','2*(NLL-reference_NLL); transpose q×c6 grid for plotting; mask invalid points','Full-clean M4 surface profiled over v,c₃, using the archived hypothetical 2%-RMS precision. Finite local domain; contour levels are diagnostics, not joint coverage claims.')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5));mods=['M1','M4','M5']
    vals=[np.array([z['parameters'][1] for z in r[10]['bootstrap'][f'continuum_A/8/{m}']['runs']])*1000 for m in mods]
    axes[0].boxplot(vals,tick_labels=mods,showmeans=True);axes[0].axhline(2,color=COLORS[1],ls='--',label='Physical truth');axes[0].set(title='100 paired noise realizations',ylabel='Recovered ν₄ (×10⁻³)');axes[0].legend()
    for j,m in enumerate(mods):
        p=r[10]['profiles'][f'continuum_A/full/{m}/1']['intervals']['95']['nu4']['components'][0];axes[1].plot([p['lower']*1000,p['upper']*1000],[j,j],lw=5,color=COLORS[j])
    axes[1].set(yticks=range(3),yticklabels=mods,xlabel='ν₄ (×10⁻³)',title='Full-clean nominal 95% profile intervals');axes[1].axvline(2,color=COLORS[1],ls='--');fig.suptitle('Free discrepancy broadens uncertainty and adds bound dependence',fontweight='bold');fig.tight_layout()
    save(fig,'08_uncertainty',10,'bootstrap.continuum_A/8/{M1,M4,M5}.runs[].parameters[1]; profiles.continuum_A/full/{M1,M4,M5}/1.intervals.95.nu4.components','ν₄ × 1000; boxplot median/IQR, whiskers 1.5 IQR; full-clean profile intervals separate panel','Left: eight sensors, 2% noise, 100 paired seeds; right: full-clean hypothetical precision, not zero-noise confidence. Different observation protocols must not be conflated. M4 has 86% discrepancy-bound contact in the left ensemble.')
    fig,ax=plt.subplots(figsize=(7,4.5));a=r[11]['stability']
    for key,label,c in [('rk4_dtmax','RK4',COLORS[1]),('imex_dtmax','CNAB2',COLORS[0])]:ax.loglog([z['N'] for z in a],[z[key] for z in a],'o-',label=label,color=c)
    ax.set(title='CNAB2 removes the observed dx⁴ stability restriction',xlabel='Grid points N',ylabel='Sampled maximum stable dt (model time units)');ax.legend();ax.grid(which='both',alpha=.2)
    save(fig,'09_stability',11,'stability[].{N,rk4_dtmax,imex_dtmax}; stability_scaling','Log–log axes; no new stability computation','625 parameter tuples and ≥2049 angles per grid; fitted dt-vs-dx exponents 4.00 and 0.23. Sampling is not proof over the continuous box.')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,task in zip(axes,['observations','gradient']):
        for j,m in enumerate(['scan_rk4','rk4','exact','cn','cnab2']):
            a=[z for z in r[11]['benchmarks'] if z['method']==m and z['task']==task and z['status']=='measured' and z['N']<=256];ax.loglog([z['N'] for z in a],[1000*z['median'] for z in a],'o-',label=m,color=COLORS[j])
        ax.set(title='Nine observations' if task=='observations' else 'Scalar loss + four-parameter gradient',xlabel='Grid points N',ylabel='Warmed median runtime (ms)');ax.grid(which='both',alpha=.2)
    axes[0].legend(fontsize=9);fig.suptitle('Same output tasks; distinct declared integration schedules',fontweight='bold');fig.tight_layout()
    save(fig,'10_runtime',11,'benchmarks filtered status=measured, task∈{observations,gradient}, N≤256: {N,method,median}','seconds × 1000; separate tasks; no compile times or estimated points','Ten synchronized warmed repetitions. CN is iterative; RK4-power and exact use direct Fourier propagation. Schedules differ in accuracy. Missing fine-grid scan measurements exceed the predeclared cap; lines do not extrapolate.')
    (ROOT/'docs/final_figure_data.json').write_text(json.dumps(provenance,indent=2)+'\n')
    lines=['# Final figure provenance','Generated with `python -m experiments.final_figures`. Only archived files are read. SVG and PNG versions share the same data. Deterministic metadata and a fixed SVG hash salt are used.']
    for p in provenance:
        lines += [f"## {p['filename']}",f"Source stage(s): {p['stage']}. Source: "+', '.join((f'[Stage {i} JSON](stage{i}_results.json)' if int(i)>=3 else f'[Stage {i} report](stage{i}_validation.md)') for i in str(p['stage']).replace('–',',').split(',') if i.isdigit()),f"Exact fields: `{p['fields']}`.",f"Transformation: {p['transformation']}.",f"Outputs: [SVG](../figures/{p['filename']}.svg), [PNG](../figures/{p['filename']}.png).",p['caption']]
    (ROOT/'docs/final_figure_provenance.md').write_text('\n\n'.join(lines)+'\n')
    print('Generated 10 figures in SVG and PNG from archives.')

if __name__=='__main__':main()

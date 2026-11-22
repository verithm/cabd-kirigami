"""Static comparison figure from final common Test metrics."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from common import OUT

def main():
 d=pd.read_csv(OUT/'test_summary.csv')
 fig,axs=plt.subplots(2,2,figsize=(16,13))
 fig.subplots_adjust(left=.17,right=.98,top=.92,bottom=.11,wspace=.66,hspace=.28)
 panels=[('stability','f1','Stability: bi -> mono','F1 (%)'),('angle_deg','mae','Stable angle','MAE (degree)'),('deformation_mm','mae','Second-well displacement','MAE (mm)'),('log10_barrier','mae','Reverse energy barrier','MAE (log10 mJ)')]
 for ax,(target,metric,title,xlabel) in zip(axs.flat,panels):
  q=d[d.target==target].sort_values(metric+'_mean',ascending=metric!='f1').reset_index(drop=True)
  mult=100 if metric=='f1' else 1
  names=q.model.str.replace('jw_','Jinwoo ').str.replace('sm_','Seungmin ').str.replace('_',' ')
  colors=['#3579b8' if n.startswith('jw_') else '#d17a35' for n in q.model]
  ax.barh(range(len(q)),q[metric+'_mean']*mult,xerr=q[metric+'_std']*mult,color=colors,alpha=.9,capsize=3)
  ax.set_yticks(range(len(q)),names,fontsize=9);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.set_title(title,fontweight='bold')
  ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False)
  if metric=='f1':ax.set_xlim(0,103)
 fig.suptitle('Common preprocessing and evaluation | 5 training seeds | mean +/- SD',fontsize=14,y=.98)
 fig.legend(handles=[Patch(color='#3579b8',label='Jinwoo families'),Patch(color='#d17a35',label='Seungmin families')],loc='lower center',bbox_to_anchor=(.5,.025),ncol=2)
 fig.savefig(OUT/'forward_comparison.png',dpi=180,bbox_inches='tight',pad_inches=.25);fig.savefig(OUT/'forward_comparison.svg',bbox_inches='tight',pad_inches=.25);plt.close(fig)
if __name__=='__main__':main()

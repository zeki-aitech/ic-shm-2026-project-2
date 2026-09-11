"""Draw a tensor-style semantic Gaussian architecture; outputs use paired RGB and semantic renders."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image
from src.colmap_io.semantic_voting import CLASS_COLORS


def draw():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(18, 10.5))
    ax.set(xlim=(0,18), ylim=(-2.5,8)); ax.axis('off')
    ink='#263c50'; blue='#eaf1f9'; gold='#fff0cf'
    def text(x,y,s,size=14,bold=False,**kw):
        ax.text(x,y,s,fontsize=size,ha='center',va='center',color=ink,weight='bold' if bold else 'normal',**kw)
    def box(x,y,w,h,s,face=blue):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',facecolor=face,edgecolor=ink,lw=1.5))
        text(x+w/2,y+h/2,s)
    def arrow(points,dash=False):
        for p,q in zip(points[:-2],points[1:-1]):
            ax.plot([p[0],q[0]],[p[1],q[1]],color=ink,lw=1.6,ls='--' if dash else '-')
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=17,lw=1.6,color=ink,linestyle='--' if dash else '-',shrinkA=3,shrinkB=3))
    def tensor(x,y,colors,label,sub):
        w=2.35/len(colors)
        for j,c in enumerate(colors):
            ax.add_patch(Rectangle((x+j*w,y),w,.75,facecolor=c,edgecolor='white',lw=1.5))
            for r in range(1,4):
                ax.plot([x+j*w,x+(j+1)*w],[y+r*.1875]*2,color='white',lw=.6,alpha=.65)
        text(x+1.175,y+1.07,label,14,True);text(x+1.175,y-.28,sub,12)
    text(2,7.55,'Learnable Gaussians',18,True)
    text(5.5,7.55,'Per-Gaussian attributes',18,True)
    text(10.6,7.55,'Shared renderer',18,True)
    text(15.7,7.78,'Two output branches',18,True)
    # A single boundary identifies all parameter families receiving training updates.
    ax.add_patch(FancyBboxPatch((.25,2.4),6.6,4.7,
        boxstyle='round,pad=0.02,rounding_size=0.12', facecolor='none',
        edgecolor='#7791a6', linewidth=1.2, zorder=0))
    rng=np.random.default_rng(7)
    for i in range(26):
        x,y=rng.uniform(.65,3.1),rng.uniform(4.65,6.7)
        ax.add_patch(Ellipse((x,y),rng.uniform(.3,.75),rng.uniform(.14,.35),angle=rng.uniform(-65,65),facecolor=plt.cm.viridis(i/26),edgecolor=ink,lw=.5,alpha=.48))
    text(1.95,4.2,'N anisotropic 3D Gaussians',13, bbox=dict(facecolor='white', edgecolor='none', pad=2))
    box(.55,2.65,2.9,1.05,'Geometry + opacity\nμ, scale, rotation, α')
    arrow([(1.95,4.55),(1.95,3.7)])
    tensor(4.25,5.75,['#dc7974','#80b88b','#7a9dce'],'RGB','N × 3  •  sigmoid')
    tensor(4.25,3.8,[CLASS_COLORS[i] / 255.0 for i in range(5)],
           'Semantic logits','N × 5  •  class IDs 0–4')
    arrow([(3.15,5.75),(3.75,5.75),(3.75,6.12),(4.25,6.12)])
    arrow([(3.15,5.2),(3.6,5.2),(3.6,4.18),(4.25,4.18)])
    ax.add_patch(Ellipse((7.35,5.15),.7,.7,facecolor=gold,edgecolor=ink,lw=1.5));text(7.35,5.15,'C',17,True)
    text(8.05,4.5,'Concat.',12)
    arrow([(6.6,6.12),(7.35,6.12),(7.35,5.5)])
    arrow([(6.6,4.18),(7.35,4.18),(7.35,4.8)])
    arrow([(7.7,5.15),(9.05,5.15)]);text(8.38,5.52,'N × 8',13,True)
    box(9.05,4.15,3.1,2,'Differentiable\nrasterization\n\nShared α-compositing',gold)
    box(9.05,6.65,3.1,.55,'Camera: K, R, t')
    arrow([(10.6,6.65),(10.6,6.15)])
    arrow([(3.45,3.17),(10.6,3.17),(10.6,4.15)])
    text(7.15,2.88,'Geometry and opacity',12)
    # Same source view for both output panels; retain the full image and aspect ratio.
    render_dir = Path('outputs/renders/fig4_holdout_real_color')
    rgb = np.asarray(Image.open(render_dir / '300_rgb.png').convert('RGB'))
    labels = np.asarray(Image.open(render_dir / '300_sem.png'))
    if labels.ndim != 2 or not np.isin(labels, list(CLASS_COLORS)).all():
        raise ValueError('Expected a class-ID mask with values 0–4')
    if rgb.shape[:2] != labels.shape:
        raise ValueError('Paired render dimensions differ')
    palette = np.stack([CLASS_COLORS[i] for i in range(5)]).astype(np.uint8)
    for panel, y in [(rgb, 5.65), (palette[labels], 3.3)]:
        panel_ax = ax.inset_axes([14.8, y, 2.6, 1.83], transform=ax.transData)
        panel_ax.imshow(panel)
        panel_ax.set_xticks([]); panel_ax.set_yticks([])
        for spine in panel_ax.spines.values():
            spine.set_edgecolor(ink)
    text(13.9,7.15,'RGB image',14,True)
    text(16.1,5.4,'Semantic map',14,True)
    arrow([(12.15,5.65),(13.1,5.65),(13.1,6.48),(14.8,6.48)])
    text(13.9,6.82,'H × W × 3',11)
    ax.add_patch(FancyBboxPatch((12.6,4.4),1.9,.8,
        boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor=blue,edgecolor=ink,lw=1.5))
    text(13.55,4.8,'Rendered logits\nH × W × 5',12)
    arrow([(12.15,4.8),(12.6,4.8)])
    # An explicit junction feeds both CE and the inference-only argmax operation.
    ax.plot([13.55,13.55],[4.4,4.1],color=ink,lw=1.6)
    ax.plot(13.55,4.1,'o',color=ink,markersize=4)
    ax.add_patch(FancyBboxPatch((13.0,3.25),1.25,.6,
        boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor='#eaf5ee',edgecolor=ink,lw=1.5))
    text(13.625,3.55,'Argmax',12)
    arrow([(13.55,4.1),(13.55,3.85)])
    arrow([(14.25,3.55),(14.55,3.55),(14.55,4.15),(14.8,4.15)])
    # CE branches from rendered logits before argmax; RGB has its own loss.
    box(10.0,1.1,3.1,1.0,'Semantic loss\nCross-entropy',gold)
    box(14.0,1.1,3.4,1.0,'Photometric loss\n0.8 L₁ + 0.2 (1 − SSIM)',gold)
    arrow([(17.4,6.48),(17.85,6.48),(17.85,1.6),(17.4,1.6)],True)
    arrow([(13.55,4.1),(12.55,4.1),(12.55,3.0),(13.5,3.0),(13.5,1.6),(13.1,1.6)],True)
    text(11.55,2.7,'Labeled / pseudo masks',12)
    text(15.7,2.7,'RGB targets',12)
    arrow([(11.55,2.45),(11.55,2.1)],True)
    arrow([(15.7,2.45),(15.7,2.1)],True)
    box(11.6,-1.45,4.1,.9,r'$\mathcal{L}=\mathcal{L}_{\rm photo}+\lambda_{\rm sem}w\mathcal{L}_{\rm CE}$',gold)
    arrow([(11.55,1.1),(11.55,.2),(12.65,.2),(12.65,-.55)],True)
    arrow([(15.7,1.1),(15.7,.2),(14.65,.2),(14.65,-.55)],True)
    text(13.65,-1.85,'w: mask-source weight',12)
    arrow([(11.6,-1.0),(3.55,-1.0),(3.55,2.4)],True)
    text(6.0,-.65,'Backpropagation',13,True)
    text(6.0,1.65,'All Gaussian parameters',13,True)
    text(6.0,.65,'Photometric: geometry, opacity, RGB\nSemantic: geometry, opacity, logits',12)
    # Compact legend uses the actual connector styles and operation symbol.
    arrow([(1.0, -2.0), (1.85, -2.0)])
    text(2.7, -2.0, 'Forward', 12)
    arrow([(4.0, -2.0), (4.85, -2.0)], True)
    text(6.0, -2.0, 'Training only', 12)
    ax.add_patch(Ellipse((7.7, -2.0), .38, .38, facecolor=gold, edgecolor=ink, lw=1.2))
    text(7.7, -2.0, 'C', 11, True)
    text(8.85, -2.0, 'Concatenate', 12)
    fig.subplots_adjust(left=.01,right=.99,bottom=.03,top=.98)
    for ext in ['png','pdf','svg']:
        fig.savefig(Path('paper/figures')/f'fig4_gaussian_architecture.{ext}',dpi=250,facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    draw()

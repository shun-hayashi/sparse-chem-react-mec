import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import textwrap

from src.chemkinetics import ChemKinetics
from src.tools import chem_equation_str
from src.load_data import param, generate_d, generate_d_from_data
from src.chemkinetics import loss_wrap
from src.tools import compensate_oxalate_l
from src.loss import cr_log, args_dtype_cr, create_jit_cr
cr_cfunc = create_jit_cr(cr_log, args_dtype_cr)
funcptr_cr = cr_cfunc.address

plot_order = ['4_1', '4_2', '4_3', '4_4', 
              '3_1', '3_2', '3_3', '3_4', 
              '2_1', '2_2', '2_3', '2_4', 
              '1_1', '1_2', '1_3', '1_4', 
              '0_1', '0_2', '0_3', '0_4']

def report(path, fname, path_save):
    d = generate_d_from_data(param["path_tc"], plot_order)
    ck = ChemKinetics(funcptr_cr, d, d_test=None)
    ck.load(path, fname)
    _l = compensate_oxalate_l(ck.l, ck.r, ck.conservation)

    text1 = "path: " + path + "\n"
    text1 += "filename: " + f"{fname:04}" + "\n"
    text1 += "\n"
    text1 += "chem formula:" + "\n"
    text1 += str(ck.chemformula) + "\n"
    text1 += "\n"
    text1 += "k_max = %8.2e" % ck.k_max + "\n"
    text1 += "k_cut = %8.2e" % ck.k_cut + "\n"
    text1 += "lam = %10.2e" % ck.lam + "\n"
    text1 += "num_eq = %4i" % ck.l.shape[0] + "\n"
    text1 += "loss = %10.2e" % ck.loss + "\n"
    text1 += "\n"
    text1 += "MRSE train = %9.2e" % ck.res_train + "\n"
    text1 += "MESE test = %10.2e" % ck.res_test + "\n"
    
    text2 = chem_equation_str(_l, ck.r, k=ck.k, chemformula=ck.chemformula)

    with PdfPages(path_save) as pdf:
        # page 1, top left: general information
        fig1 = plt.figure(figsize=(10, 14))
        gs1 = gridspec.GridSpec(2, 2, height_ratios=[1, 3.5], figure=fig1)
        
        ax1 = fig1.add_subplot(gs1[0, 0])
        ax1.axis('off') 
        wrapped_text = '\n'.join(textwrap.fill(line, width=40) for line in text1.split('\n'))
        ax1.text(0.0, 1.0, wrapped_text, fontsize=12, ha='left', va='top')
        ax1.text(0.0, 1.0, wrapped_text, fontsize=12, ha='left', va='top')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # page 1, top right: R2 scores
        ax2 = fig1.add_subplot(gs1[0, 1])
        c = np.empty(len(plot_order))
        for i in range(len(plot_order)):
            marker = "o"
            c[i] = 1-loss_wrap(funcptr_cr, d[i,:,:].reshape(1,d.shape[1],d.shape[2]), ck.l, ck.r, ck.k, 0.)
        sc = ax2.scatter(d[:, 1, 0], d[:, 3, 0], c = c, cmap = 'bwr_r', s=100, edgecolor="k", linewidth=1) 
        sc.set_clim(None, 1.0)
        fig1.colorbar(sc)
        ax2.set_title('R2 score')
        ax2.set_xlim(0,1)
        ax2.set_ylim(-0.05,0.25)
        ax2.set_xlabel(r"[Mn$^{7+}$]$_0$")
        ax2.set_ylabel(r"[Mn$^{2+}$]$_0$")
    
        # page 1, bottom: list of elementary steps
        lines = text2.split("\n")
        if len(lines) > 40:
            text2_1 = "\n\n".join([lines[i] for i in range(40)])
            text2_2 = "\n\n".join([lines[i] for i in range(40,len(lines))])
        
            ax3 = fig1.add_subplot(gs1[1, 0])
            ax3.axis('off') 
            ax3.text(0.0, 1.0, text2_1, fontsize=10, ha='left', va='top', linespacing=0.8, family='monospace')
            ax4 = fig1.add_subplot(gs1[1, 1])
            ax4.axis('off') 
            ax4.text(0.0, 1.0, text2_2, fontsize=10, ha='left', va='top', linespacing=0.8, family='monospace')
        else:
            text2_1 = "\n\n".join([lines[i] for i in range(len(lines))])
            ax3 = fig1.add_subplot(gs1[1, 0])
            ax3.axis('off') 
            ax3.text(0.0, 1.0, text2_1, fontsize=10, ha='left', va='top', linespacing=0.8, family='monospace')
            
        fig1.subplots_adjust(left=0.02, right=0.98, bottom=0.00, top=0.98)
        pdf.savefig(fig1)
        plt.close(fig1)
       
        # page 2: experimental and simulated concentration profiles
        fig2 = plt.figure(figsize=(10, 14))
        gs2 = gridspec.GridSpec(5, 4, figure=fig2)
        
        columns_count = 4
        rows_count =(d.shape[0]-1)//columns_count+1
        axes = []
        
        ck = ChemKinetics(funcptr_cr, d, d_test=None)
        ck.load(path, fname) 
        ck.simulate()
        sim = ck.sim
        
        for i in range(d.shape[0]):
            axes.append(fig2.add_subplot(rows_count, columns_count, i+1)) 
            title = "(" + f"{d[i,1,0]:.2f}" + ", " +  f"{d[i,3,0]:.2f}" +")"
            plt.title(title)
            if ck.sim is not None:
                color = ["pink", "cyan", "greenyellow"] + [str(i/(sim.shape[1]-3)*0.5+0.5) for i in range(sim.shape[1]-3)]
                
                label_sim = [r"sim(Mn$^{7+}$)", r"sim(Mn$^{3+}$)", r"sim(Mn$^{2+}$)"]
                for j in range(3):
                    #label = 'sim_' + str(j+1)
                    label = label_sim[j]
                    axes[i].plot(sim[i,0,:], sim[i,j+1,:], label=label, color=color[j], lw=2)
            axes[i].scatter(d[i, 0, :], d[i, 1, :], label=r"exp(Mn$^{7+}$)", color="r", s=10)
            axes[i].scatter(d[i, 0, :], d[i, 2, :], label=r"exp(Mn$^{3+}$)", color="b", s=10)
            axes[i].scatter(d[i, 0, :], d[i, 3, :], label=r"exp(Mn$^{2+}$)", color="g", s=10)
            axes[i].set_xlabel("t")
            axes[i].set_ylabel("conc")
        handles0, labels0 = axes[0].get_legend_handles_labels()
        fig2.legend(handles0, labels0, ncol=6, loc='lower center', borderaxespad=0, title=r"([Mn$^{7+}$]$_0$, [Mn$^{2+}$]$_0$)", bbox_to_anchor=(0.5, 0.01)) # 0.04
        fig2.subplots_adjust(wspace=0.5, hspace=0.5) 
        fig2.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=0.98)
        
        pdf.savefig(fig2)
        plt.close(fig2)

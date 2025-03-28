import scanpy as sc
import scanpy.external as sce
import scvelo as scv
import numpy as np
import pandas as pd
import warnings, scipy.sparse as sp, matplotlib, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import rc_context
from collections import Counter
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances
from mousipy import translate

import matplotlib.font_manager
import openpyxl
import pyreadr
import rpy2
import palantir
import loompy
import feather
import re
import sys
import matplotlib.lines as lines

sc.set_figure_params(dpi=80, dpi_save=300, color_map='Spectral_r', vector_friendly=True, transparent=True)
sc.settings.verbosity = 3 # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()

scv.set_figure_params(dpi=80, dpi_save=300, color_map='viridis', vector_friendly=True, transparent=True, format='pdf')

pd.set_option('display.max_rows', 200)



def observe_variance(anndata_object):
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # variance per principal component
    x = range(len(anndata_object.uns['pca']['variance_ratio']))
    y = anndata_object.uns['pca']['variance_ratio']
    ax1.scatter(x,y,s=4)
    ax1.set_xlabel('PC')
    ax1.set_ylabel('Fraction of variance explained\n')
    ax1.set_title('Fraction of variance explained per PC\n')
    # cumulative variance explained
    cml_var_explained = np.cumsum(anndata_object.uns['pca']['variance_ratio'])
    x = range(len(anndata_object.uns['pca']['variance_ratio']))
    y = cml_var_explained
    ax2.scatter(x,y,s=4)
    ax2.set_xlabel('PC')
    ax2.set_ylabel('Cumulative fraction of variance\nexplained')
    ax2.set_title('Cumulative fraction of variance\nexplained by PCs')
    fig.tight_layout()
    plot = plt.show
    return(plot)
    
    
    # divergent
user_defined_palette =  [ '#3283FE', '#16FF32', '#F6222E',  '#FEAF16', '#BDCDFF', '#3B00FB', '#1CFFCE', '#C075A6', '#F8A19F', '#B5EFB5', '#FBE426', '#C4451C', '#2ED9FF', '#c1c119', '#8b0000', '#FE00FA', '#1CBE4F', '#1C8356', '#0e452b', '#AA0DFE', '#B5EFB5', '#325A9B', '#90AD1C']
# gradient of one color
user_defined_cmap_markers = LinearSegmentedColormap.from_list('mycmap', ["#E6E6FF", "#CCCCFF", "#B2B2FF", "#9999FF",  "#6666FF",   "#3333FF", "#0000FF"])
# gradient of two colors
user_defined_cmap_degs = LinearSegmentedColormap.from_list('mycmap', ["#0000FF", "#3333FF", "#6666FF", "#9999FF", "#B2B2FF", "#CCCCFF", "#E6E6FF", "#E6FFE6", "#CCFFCC", "#B2FFB2", "#99FF99", "#66FF66", "#33FF33", "#00FF00"])


adata = sc.read_10x_h5(sys.argv[1])
adata.var_names_make_unique()
print(adata.shape) # check the number of cells and genes in each sample


adata.raw = adata # keep a copy of the raw adata 
np.random.seed(42) 
index_list = np.arange(adata.shape[0]) # randomize the order of cells for plotting
np.random.shuffle(index_list)
adata = adata[index_list]


sc.pp.calculate_qc_metrics(adata, inplace=True)

#store unfiltered/unprocessed data into new fields prior to downstream analysis
adata.obs['original_total_counts'] = adata.obs['total_counts']
adata.obs['log10_original_total_counts'] = np.log10(adata.obs['original_total_counts'])

# select mitochondrial, ribosomal and hemoglobin genes (case sensitive)
adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))
adata.var['ribo'] = adata.var_names.str.startswith(('RPS','RPL', 'Rps', 'Rpl'))
adata.var['hb'] = adata.var_names.str.startswith(('HB', 'Hb'))

# for each cell compute fraction of counts in mitochondrial, ribosomal and hemoglobin genes vs. all genes 
adata.obs['mito_frac'] = np.sum(adata[:,adata.var['mt']==True].X, axis=1) / np.sum(adata.X, axis=1)
adata.obs['ribo_frac'] = np.sum(adata[:,adata.var['ribo']==True].X, axis=1) / np.sum(adata.X, axis=1)
adata.obs['hb_frac'] = np.sum(adata[:,adata.var['hb']==True].X, axis=1) / np.sum(adata.X, axis=1)


sc.external.pp.scrublet(adata, random_state=42) # run scrublet to predict potential doublets
import pandas as pd
import numpy as np
import umap  # trimap, pacmap etc.
#import umap.plot
import plotly.express as px
import matplotlib.pyplot as plt
import os
import sys
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from jsonargparse import ArgumentParser
from jsonargparse import ActionYesNo
from jsonargparse.typing import Path_drw, Path_dc  # path that can be read, patch that can be created
from adjustText import adjust_text
from pathlib import PosixPath

from read_embeddings import read_and_process_data

# The columns expected in the data by default
embedding_columns=["embed_first", "embed_half", "embed_last"]

# CORE scheme labels, and different combinations
all_labels = ["HI","ID","IN","IP","LY","MT","NA","OP","SP"]
main_labels = ["HI","ID","IN","IP","NA","OP"]
without_MT = ["HI","ID","IN","IP","LY","NA","OP","SP"]
#sublabels = ["it", "os", "ne", "sr", "nb", "on", "re","oh", "en", "ra", "dtp", "fi", "lt", "oi", "rv","ob", "rs", "av", "oo""ds", "ed", "oe"]
sublabels = ["IT", "OS", "NE", "SR", "NB", "ON", "RE","OH", "EN", "RA", "DTP", "FI", "LT", "OI", "RV","OB", "RS", "AV", "OO""DS", "ED", "OE"]

labels_all_hierarchy_with_other = {
    "MT": ["MT"],
    "LY": ["LY"],
    "SP": ["SP", "IT", "OS"],
    "ID": ["ID"],
    "NA": ["NA", "NE", "SR", "NB", "ON"],
    "HI": ["HI", "RE", "OH"],
    "IN": ["IN", "EN", "RA", "DTP", "FI", "LT", "OI"],
    "OP": ["OP", "RV", "OB", "RS", "AV", "OO"],
    "IP": ["IP", "DS", "ED", "OE"],
}
# above dictionary reversed for easier lookup
reverse_hierarchy = {}
for main_label, sub_labels in labels_all_hierarchy_with_other.items():
    for sub_label in sub_labels:
        reverse_hierarchy[sub_label] = main_label

cluster_map = {
    "LY": ["E.g., document meta, (mainly Urdu) proper nouns", "'lyrics', 'song'/'music'", "E.g., sentiment and memory-related nouns and verbs", "Nouns denoting (lyrical) authorship, 'poetry'"],
    "SP": ["E.g., words denoting authorship, agency, or medium of communication", "E.g., words denoting presentation, explanation, or perception", "Mostly question verbs/nouns/abbreviations", "Words denoting conversation", "Words denoting interview/discussion", "Mostly general adjectives and adverbs, esp. interrogatives"],
    "ID": ["Abbreviations, forum structure metatext", "'discussion', 'forum'", "Question verbs/nouns", "Interaction, membership", "Answer verbs/nouns", "Post metawords"],
    "NA": ["Verbs denoting (mostly) communication", "Chinese adverbs of time", "Mostly Urdu toponyms", "E.g., sport-related words, 'comments', 'blog'", "'news', 'news reporter'", "General communication and news-related words", "English and French weekday names", "'photo', 'picture'"],
    "HI": ["Measurements, ingredients", "Imperative verbs", "Document meta, general/abstract nouns", "Procedure nouns and 'how'", "'preparation' and 'recipe'", "Instruction nouns"],
    "IN": ["Association words, proper nouns", "Introduction, description, summary", "'name', 'title', 'called/'named''", "E.g., meaning and purpose nouns", "Document types", "Explanatory adverbs and discourse connectives", "Biographical and editing-related nouns/verbs"],
    "OP": ["Urdu religious/exaltatory words", "E.g., words denoting worth/valuation", "Mostly words denoting document structure and/or communication mediums", "English and French month names", "E.g., 'comment', 'review', 'rating'", "Words denoting positive or negative sentiment", "Weekday names & 'today' (mostly English and French)", "E.g., 'God', 'Christ', 'holy', 'church'"],
    "IP": ["Adverbs and adjectives, 'how'", "Product metatext", "Book-related nouns", "Proper nouns, currencies, 'customer'", "Positive attributes", "Persuasion, imperatives", "Product-related words", "E.g., 'dimensions' and offer-related nouns"],
}

lang_map = {
	"en": "English",
	"fr": "French",
	"ur": "Urdu",
	"zh": "Chinese",
}

# this needed for CORE scheme, as NA is read as NaN
remove_nan = lambda x: "NA" if x == "nan" else str(x)

# this for saving figs:
fig_label = {"label_for_umap":"register", "lang":"language", "true_label":"true_label"}

# This for easy parsing of label parameters; if given as a list, that is used, else checking for keywords
def parse_labels(l):
    if type(l)==list:
        return l
    elif l == "upper" or l == "all":
        return all_labels
    elif l == "without_MT":
        return without_MT
    elif l == "main":
        return main_labels
    else:
        print(f"ERRONIOUS LABELS; USING {main_labels}")

def parse_save_directory(d):
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except:
        print(f"Cannot create saving directory {d}, using default value (umap-figures/).")
        return "umap-figures/"
        

# ---------------------------------------------------ARGUMENTS--------------------------------------------------- #

ap = ArgumentParser(prog="plot_embeddings.py", description="Plot pre-calculated embeddings. Give \
                    input data as a pandas dataframe, define which columns to use and where to save figures.")
ap.add_argument('--embeddings', '--data', type=Path_drw, required=True, metavar='DIR',
                help='Path to directory where precalculated embeddings are as csv/tsv. Language name separated by _ \
                    (e.g. emb_fr.tsv, sv_emb.tsv) assumed in filenames.')
ap.add_argument('--languages','--langs','--language','--lang', type=list[str], metavar='LIST', required=True,
                help='Which languages to download from --embeddings path.')
ap.add_argument('--labels', type=parse_labels, metavar='LIST or GROUP NAME', default=main_labels,
                help='Labels as list )["IN", "NA"] etc.) or a group name. Others discarded. \
                      Group names = ["all", "main", "without_MT"].')
ap.add_argument('--use_column_labels', '--column_l', type=str, default="preds_best", metavar='COLUMN',
                help='Column name containing labels that are used for coloring the figure. \
                    If "preds_best" (default), assumes data contains column named preds_{threshold}.')
ap.add_argument('--use_column_embeddings', '--column_e', type=list, default=embedding_columns, metavar='LIST[COLUMN]',
                help='column(s) that contain embeddings. If multiple, separate figs are produced.')
ap.add_argument('--remove_hybrids', type=bool, metavar='BOOL', default=True,
                help=f'Remove docs with multiple main level ({all_labels}) predictions.')
ap.add_argument('--keep_sublabels', type=bool, metavar='BOOL', default=False,
                help='Keep sublabels and attach them to main level with hyphen, e.g. "HI-re"')
ap.add_argument('--pca','--n_pca', type=int, metavar='INT>0', default=None,
                help="Number of dimensions mapped to with PCA before UMAP. No value = No PCA.")
ap.add_argument('--hover_text', type=str, metavar="COLUMN", default=None, 
                help="Adds column values as hover text")
ap.add_argument('--truncate_hover', default=True,  type=bool, metavar="bool",
                help='Truncate hover text.')
ap.add_argument('--n_neighbors', type=int, metavar='INT', default=50,
                help='How many neighbors for UMAP.')
ap.add_argument('--min_dist', type=float, metavar='FLOAT', default=0.0,
                help='Minimum distance for UMAP.')
ap.add_argument('--sample', type=int, metavar='INT', default=None,
                help='How much to sample from each language, if given.')
ap.add_argument('--model_name', type=str, metavar="STR",
                help='Added to plot titles if given.')
ap.add_argument('--data_name', type=str, metavar="STR",
                help='Added to plot titles if given.')
ap.add_argument('--seed', type=int, metavar='INT', default=None,
                help='Seed for reproducible outputs. Default=None, UMAP runs faster with no seed.')
ap.add_argument('--extension', type=str, metavar="str", default="png", choices=["png", "html"],
                help="Which format for saving the plots. --hover_text forces html.")
ap.add_argument('--save_dir', '--output_dir', type=parse_save_directory, metavar='DIR', default="umap-figures/",
                help='Dir where to save the results to.')
ap.add_argument('--save_prefix', '--output_prefix', type=str, metavar='str', default="umap_",
                help='Prefix for save file, lang/label and other params added as well as file extension.')
ap.add_argument('--header', type=list[str], metavar='LIST[COLUMNS]', default=None,
                help='If the data has no column names, give them here.')



#------------------------------------------------UMAP----------------------------------------------------#


def apply_reducer(df, reducer, options):
    # Values from string to list and flatten, get umap embeds
    for column in options.use_column_embeddings:
        try:
            df[column] = df[column].apply(
                lambda x: np.array([float(y) for y in eval(x)[0]])   # remove [0], if your embeds are [x1, x2, ...]
            )
        except:
            try:
                df[column] = df[column].apply(
                    lambda x: np.array([float(y) for y in np.fromstring(x, sep=" ")])   # change sep, if your embeds are "x1,x2,..."
                )
            except:
                print("Cannot change the data type of the embedding columns to float. Try to change the separator in the \
                      codeblock above, if your embeds are separated by something else than white space, or remove [0], if \
                      your data is not doubly nested.")
        scaled_embeddings = StandardScaler().fit_transform(df[column].tolist())
        if options.pca is not None:
            pca = PCA(n_components=options.pca)
            scaled_embeddings = pca.fit_transform(scaled_embeddings)
        red_embedding = reducer.fit_transform(scaled_embeddings)
        df["x_"+column] = red_embedding[:, 0]
        df["y_"+column] = red_embedding[:, 1]
    return df



#------------------------------------------------plotting-----------------------------------------------#

def plot_embeddings_normal(df_plot, data_column, color_column, options, title=None):

    if title is None:
        title = f'Embeddings with {options.model_name} from {options.data_name}'

    print(f"Now plotting {'x_'+data_column},{'y_'+data_column} with coloring based on {color_column}.")
    
    fig = px.scatter(df_plot, x='x_'+data_column, y='y_'+data_column, color=color_column,
                     title=title, labels="cluster",#{"label_for_umap": "Register", "lang":"Language"},
                     width=1200, height=900)  # Increased size for better visibility

    
    fig.update_traces(marker={"opacity":0.5, "size":3})
    # freeze legend
    fig.update_layout(legend={'itemsizing': 'constant'})
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
    
    # Save the figure as an HTML file or png
    fig_file = os.path.join(options.save_dir, f'{options.save_prefix}_wrt_{fig_label.get(color_column, color_column)}_{data_column}.{options.extension}')
    if not os.path.exists(options.save_dir):
        os.makedirs(options.save_dir)
    if options.extension == "html":
        fig.write_html(fig_file)
    else:
        fig.write_image(fig_file)


def wrap_text(text, width, truncate=True):
    """Wrap text with a given width."""
    if truncate:
        text = text[0:500]
    # Wrap without splitting words; build lines of max 'width' chars.
    words = text.split()
    width_rec = width
    width -= 20  # leave some margin for first line
    if not words:
        text = ""
    else:
        lines = []
        current = words[0]
        for w in words[1:]:
            # if adding the next word (plus a space) stays within width, append it,
            # otherwise start a new line. If a single word is longer than width,
            # it will occupy its own line (not split).
            if len(current) + 1 + len(w) <= width:
                current += " " + w
            else:
                lines.append(current)
                current = w
                width = width_rec  # reset width for next lines
        lines.append(current)
        text = '<br>'.join(lines)
    return text

# --- new helper: generate a color map for an arbitrary number of categories ---
def build_color_map(categories, seed=None):
    """
    Return a dict mapping each category (iterable) to a Plotly-acceptable color string.
    Uses HSL hues evenly spaced around the color wheel for up to any N categories.
    """
    cats = list(categories)
    if seed is not None:
        # deterministic but simple shuffle if requested
        rng = np.random.RandomState(seed)
        rng.shuffle(cats)
    n = len(cats)
    if n == 0:
        return {}
    hues = np.linspace(0, 360, n, endpoint=False)
    colors = [f"hsl({int(h) % 360},40%,100%)" for h in hues]
    return {cat: color for cat, color in zip(cats, colors)}

def plot_embeddings_matplotlib(df_plot, data_column, color_column, options, column_name, title=None):

    if title is None:
        title = f'Embeddings with {options.model_name} from {options.data_name}'
    reg = df_plot['label_for_umap'].iloc[0]
    cluster_map_fn = lambda x: cluster_map[reg][int(x)]
    df_plot[color_column] = df_plot[color_column].apply(cluster_map_fn)
    df_plot['lang'] = df_plot['lang'].apply(lambda x: lang_map[x])
    
    print(f"Now plotting matplotlib plots for {'x_'+data_column},{'y_'+data_column} with coloring based on {color_column}.")

    # choose font that supports CJK if available
    import matplotlib.font_manager as fm
    from matplotlib.font_manager import FontProperties
    from PIL import ImageFont

    # choose font fallbacks: prefer a CJK-capable font first, then good Latin fonts
    preferred_fonts = ['DejaVu Sans', 'Liberation Sans', 'Noto Sans Mono']

    # build list of available preferred names (keep order)
    available = {f.name for f in fm.fontManager.ttflist}
    font_list = [f for f in preferred_fonts if f in available]
    if not font_list:
        # fallback to whatever matplotlib finds
        font_list = [fm.findfont(fm.FontProperties())]

    # determine a usable font family name for fall-back annotation rendering
    # font_list may contain either font names (strings from preferred_fonts) or
    # a font path returned by fm.findfont(); convert to a family name if needed.
    '''
    if font_list:
        first = font_list[0]
        if isinstance(first, str) and first in available:
            font_family = first
        else:
            try:
                font_family = fm.FontProperties(fname=first).get_name()
            except Exception:
                font_family = 'DejaVu Sans'
    else:
        font_family = 'DejaVu Sans'
    '''

    # let matplotlib use the family + fallback list
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font_list

    '''
    def _font_path_supports_char(path, char):
        try:
            ImageFont.truetype(path, size=16).getmask(char)
            return True
        except Exception:
            return False

    def font_properties_for_text(text, candidates=None):
        """
        Return a FontProperties object pointing to the first candidate font
        that can render at least one non-ASCII character found in `text`.
        If none found, return None (so matplotlib falls back to rcParams).
        """
        if candidates is None:
            candidates = font_list
        # pick a representative character that is outside ASCII if present
        rep_char = next((c for c in text if ord(c) > 127), None)
        if rep_char is None:
            return None
        for name in candidates:
            try:
                path = fm.findfont(name, fallback_to_default=False)
            except Exception:
                continue
            if _font_path_supports_char(path, rep_char):
                return FontProperties(fname=path)
        return None
    '''

    # full-data markers (all points) + top-N text annotations
    top_n = 10
    df_top = df_plot.groupby('lang', sort=False).head(top_n).reset_index(drop=True)

    # marker shapes per language
    #markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    langs = list(dict.fromkeys(df_plot['lang'].tolist()))          # use all langs for markers
    #lang_to_marker = {lang: markers[i % len(markers)] for i, lang in enumerate(langs)}
    lang_to_marker = { 'Chinese':'o', 'English':'s', 'French':'^', 'Urdu':'D'}  # fixed mapping for now

    # cluster -> color mapping (cover whole df_plot)
    unique_clusters = list(dict.fromkeys(df_plot[color_column].tolist()))
    n_clusters = max(1, len(unique_clusters))
    cmap = plt.get_cmap('tab20')
    cluster_colors = [cmap(i / max(1, n_clusters - 1)) for i in range(n_clusters)]
    cluster_to_color = {cat: cluster_colors[i] for i, cat in enumerate(unique_clusters)}

    fig, ax = plt.subplots(figsize=(12, 9))

    # 1) plot all points (small markers), colored by cluster, marker shape by language
    for lang in langs:
        rows_all = df_plot[df_plot['lang'] == lang]
        if rows_all.empty:
            continue
        xs_all = rows_all['x_'+data_column].values
        ys_all = rows_all['y_'+data_column].values
        cols_all = [cluster_to_color[c] for c in rows_all[color_column].tolist()]
        ax.scatter(xs_all, ys_all, marker=lang_to_marker[lang], c=cols_all, s=40, edgecolors='none',
                   label=f'{lang} (pts)', alpha=0.4)

    # give axes some extra margin so labels have room in all directions
    margin_frac = 0.4  # 30% margin on each side (tune as needed)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = (x1 - x0) * margin_frac
    dy = (y1 - y0) * margin_frac
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)

    # 2) overlay top-N points with larger markers, add connecting lines + text labels
    texts = []
    text_points_x = []  # list of x for adjust_text
    text_points_y = []  # list of y for adjust_text
    for lang in list(dict.fromkeys(df_top['lang'].tolist())):
        rows = df_top[df_top['lang'] == lang]
        xs = rows['x_'+data_column].values
        ys = rows['y_'+data_column].values
        cols = [cluster_to_color[c] for c in rows[color_column].tolist()]
        # larger markers for top examples
        ax.scatter(xs, ys, marker=lang_to_marker[lang], c=cols, s=140, edgecolors='black',
                   linewidths=0.3,
                   zorder=3, label=f'{lang} (top)', alpha=0.9)

        # compute a reasonable default offset in data units (a fraction of the axis span)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        span_x = max(1e-9, xlim[1] - xlim[0])
        span_y = max(1e-9, ylim[1] - ylim[0])
        base_offset = 0.04 * max(span_x, span_y)

        # annotate with an initial offset distributed around the point (to give adjust_text room)
        for i, (x, y, txt, col, rlang) in enumerate(zip(xs, ys, rows['text'].tolist(), cols, rows['lang'].tolist())):
            # pick an initial direction so labels don't all pile up; cycle through 8 directions
            angle = (i % 8) * (2 * np.pi / 8)
            dx = np.cos(angle) * base_offset
            dy = np.sin(angle) * base_offset

            try:
                if rlang and str(rlang).lower() == 'chinese':
                    font_path = fm.findfont('Droid Sans Fallback', fallback_to_default=False)
                    fp = FontProperties(fname=font_path)
                elif rlang and str(rlang).lower() == 'urdu':
                    font_path = PosixPath(os.environ.get("MISC", "/scratch/project_462001491/tlundber/misc") + "/NotoNaskhArabic-VariableFont_wght.ttf")
                    fp = FontProperties(fname=font_path)
                else:
                    font_path = fm.findfont('DejaVu Sans', fallback_to_default=False)
                    fp = FontProperties(fname=font_path)
            except Exception:
                fp = None

            text_kwargs = {'fontproperties': fp} if fp is not None else {'fontfamily': 'sans-serif'}

            # place text at the offset position (adjust_text will move it further if needed)
            #t = ax.text(x + dx, y + dy, str(txt),
            t = ax.text(x, y, str(txt),
                        fontsize=16,# ha='left', va='bottom',
                        color='black', weight=800, zorder=4, **text_kwargs)
            texts.append(t)
            text_points_x.append(x)
            text_points_y.append(y)

    # use adjustText to separate overlapping text labels and draw connector lines.
    #'''
    if texts:
        # stronger separation parameters and arrows that avoid striking through text
        arrowprops = dict(arrowstyle='->', linewidth=0.75, color='gray',
                          shrinkA=3, shrinkB=3,
                          #connectionstyle="arc3,rad=0.1"
                          )

        # allow movement in both x and y for both points and text; increase expansion/force so labels move farther
        adjust_text(texts,
                    x=text_points_x,
                    y=text_points_y,
                    #only_move={'points': 'xy', 'text': 'xy'},
                    #autoalign=True,
                    #expand_text=(1.2, 1.2),
                    #expand_points=(1.2, 1.2),
                    expand=(1.6, 1.6),
                    force_text=1.8,
                    force_static=1.0,
                    force_explode=(4.5, 4.5),
                    arrowprops=arrowprops,
                    #arrowprops=dict(arrowstyle='->', linewidth=0.8),
                    #ax=ax
                    )
    #'''

    # Legends:
    from matplotlib.lines import Line2D
    legend_kwargs = dict(frameon=True, fontsize=14, markerscale=1.2, handlelength=1.5, handletextpad=0.6)

    # cluster (color) legend - color handles (keep in main plot)
    cluster_handles = [Line2D([0], [0], marker='o', color='w', label=str(cat),
                              markerfacecolor=cluster_to_color[cat], markersize=9, markeredgecolor='black') for cat in unique_clusters]
    # make cluster legend font larger for readability
    leg_cluster = ax.legend(handles=cluster_handles, title='Cluster',
                            bbox_to_anchor=(1.02, 1), loc='upper left', **legend_kwargs)
    # ensure legend title font matches
    try:
        leg_cluster.get_title().set_fontsize(16)
    except Exception:
        pass
    
    # remove axis ticks/labels (no numeric axes needed)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Save the main figure (with cluster legend only)
    img_file = os.path.join(options.save_dir, column_name, options.labels[0].lower(),
                            f'{options.save_prefix}_wrt_{fig_label.get(color_column, color_column)}_{data_column}.png')
    subpath = os.path.join(options.save_dir, column_name, options.labels[0].lower())
    if not os.path.exists(subpath):
        os.makedirs(subpath, exist_ok=True)
    fig.savefig(img_file, dpi=300, bbox_inches='tight')

    # Create and save a separate legend image for Language (marker) only
    lang_handles = [Line2D([0], [0], marker=lang_to_marker[l], color='w', label=l,
                           markerfacecolor='gray', markersize=9, markeredgecolor='black') for l in langs]
    leg_fig = plt.figure(figsize=(3, max(1.2, 0.35 * len(langs))))
    leg_ax = leg_fig.add_subplot(111)
    leg_ax.axis('off')
    # place legend centered in the legend figure
    lang_leg = leg_ax.legend(handles=lang_handles, title='Language (marker)',
                             loc='center', **legend_kwargs)
    lang_leg.get_title().set_fontsize(16)
	# tighten and save
    leg_fig.tight_layout()
    lang_legend_file = os.path.join(options.save_dir, column_name, f'{options.save_prefix}_language_legend.png')
    leg_fig.savefig(lang_legend_file, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(leg_fig)

    plt.close(fig)

def plot_embeddings_with_hover(df_plot, data_column, color_column, options, column_name, title=None):

    if title is None:
        title = f'Embeddings with {options.model_name} from {options.data_name}'
    df_plot["comments_hover"] =  df_plot.apply(lambda row: f"{wrap_text(row['comments'], 80, truncate=options.truncate_hover)}", axis=1)
    reg = df_plot['label_for_umap'].iloc[0]
    cluster_map_fn = lambda x: cluster_map[reg][int(x)]
    #print("Color column:", color_column)
    #df_plot[f"{color_column}_numerical"] = df_plot[color_column]
    if 'lang' in color_column:
        df_plot[color_column] = df_plot[color_column].apply(lambda x: lang_map[x])
    elif not any(s in color_column for s in ['label_for_umap', 'lang']):
        df_plot[color_column] = df_plot[color_column].apply(cluster_map_fn)
    
	# build color map and also a per-row color column for text color
    unique_cats = sorted(df_plot[color_column].unique().tolist(), key=lambda x: str(x))
    color_map = build_color_map(unique_cats, seed=getattr(options, "seed", None))
    df_plot[f"{color_column}_numerical"] = df_plot[color_column].map(color_map)
    print("Color map:", color_map)
    print("Color column:", df_plot[color_column].head(20))
    print("Color column numerical:", df_plot[f"{color_column}_numerical"].head(20))
    print("Text column:", df_plot['text'].head(20))

    print(f"Now plotting interactive plots for {'x_'+data_column},{'y_'+data_column} with coloring based on {color_column}.")

    fig = px.scatter(df_plot, x='x_'+data_column, y='y_'+data_column, color=color_column,
                     title=title,
                     labels={"label_for_umap": "Register", "lang":"Language", "text":"Word", f"{color_column}":"Cluster", "script_type":"Script type", "translation":"Translation", "comments_hover":"Comments"},
                     text="text",
                     #hover_data={"text":True,"lang":True, "label_for_umap":True, "script_type":True, "translation":True, "comments_hover":True, "comments":False, "x_"+data_column:False, "y_"+data_column:False},
                     width=1200, height=900)  # Increased size for better visibility
    fig.update_layout(legend= {'itemsizing': 'constant'}, xaxis_visible=False, yaxis_visible=False)
    fig.update_traces(mode="text", hoverinfo="skip")#, textfont=dict(family="Arial", size=15, color=df_plot[f"{color_column}_numerical"]))#, marker={"opacity":1, "size":20})
    for trace in fig.data:
        trace.textfont = dict(family="Arial", size=15, color=trace.marker.color)
        #trace.mode = "text"
        #trace.hoverinfo = "skip"
        #trace.marker.update(opacity=1, size=20)
    #fig.update_layout(
    #    legend_title_text='Cluster',
    #    hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell", bordercolor="black"),
    #)
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'paper_bgcolor': 'rgba(255, 255, 255, 1)',
        })
    # Save the figure as an HTML file
    img_file = os.path.join(options.save_dir, column_name, options.labels[0].lower(), f'{options.save_prefix}_wrt_{fig_label.get(color_column, color_column)}_{data_column}.png')
    subpath = os.path.join(options.save_dir, column_name, options.labels[0].lower())
    if not os.path.exists(subpath):
        os.makedirs(subpath)
    fig.write_image(img_file)
    
    # Add custom JavaScript for copying to clipboard
    #with open(img_file, 'a') as f:
    #    f.write("""
    """
<script>
    document.addEventListener('DOMContentLoaded', function() {
        var plot = document.querySelector('.plotly-graph-div');
        plot.on('plotly_click', function(data) {
            var infotext = data.points.map(function(d) {
                return d.customdata[0].replace(/<[^>]+>/g, '');  // Remove HTML tags for clean clipboard content
            });
            copyToClipboard(infotext.join('\\n\\n'));
        });
    });

    function copyToClipboard(text) {
        var el = document.createElement('textarea');
        el.value = text;
        document.body.appendChild(el);
        el.select();
        document.execCommand('copy');
        document.body.removeChild(el);
        var notification = document.createElement('div');
        notification.innerHTML = 'Copied to clipboard';
        notification.style.position = 'fixed';
        notification.style.bottom = '10px';
        notification.style.left = '10px';
        notification.style.padding = '10px';
        notification.style.backgroundColor = '#5cb85c';
        notification.style.color = 'white';
        notification.style.borderRadius = '5px';
        document.body.appendChild(notification);
        setTimeout(function() {
            document.body.removeChild(notification);
        }, 2000);
    }
</script>
    """


if __name__=="__main__":
    options = ap.parse_args(sys.argv[1:])
    print(options)
    print("")
    df = read_and_process_data(options)
    if options.hover_text is not None:  # force this after hover_text is handld in read_and_process_data()
        options.extension = "html"
    if options.seed is not None:
        reducer = umap.UMAP(random_state=options.seed, n_neighbors=options.n_neighbors, min_dist=options.min_dist)
    else:
        reducer = umap.UMAP(n_neighbors=options.n_neighbors, min_dist=options.min_dist)
    # for a pacmap implementation?
    # reducer = pacmap.PaCMAP(n_neighbors=options.n_neighbors, apply_pca=True, MN_ratio=2, FP_ratio=1, random_state=seed)
    
    # apply reducer to embeddings, apply pca if stated in the options
    apply_reducer(df, reducer, options)
    
    # change which function to use based on hover (just rename the function to plot_embeddings)
    plot_embeddings = plot_embeddings_with_hover if options.hover_text is not None else plot_embeddings_normal

    # plot wrt labels
    for column in options.use_column_embeddings:
        color = "label_for_umap"
        plot_embeddings(df, column, color, options)
    # plot wrt language
    for column in options.use_column_embeddings:
        color = "lang"
        plot_embeddings(df, column, color, options)

    
    exit(0)
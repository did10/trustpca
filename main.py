import pandas as pd
import numpy as np
import streamlit as st
import copy
import pickle
import tempfile
from pathlib import Path

import plotly.graph_objects as go
from joblib import Parallel, delayed
from scipy.stats import chi2
import plotly.io as pio
import os
# from weasyprint import HTML
from io import BytesIO
from io import StringIO
import itertools
import plotly.express as px

import streamlit.components.v1 as components
import io


###########################################################################################################
#file handling stuff, parsing

def handle_file_input(file_input):
    if isinstance(file_input, str) and os.path.isfile(file_input):
        return open(file_input, 'r') 
    elif hasattr(file_input, 'getvalue'):
        return StringIO(file_input.getvalue().decode("utf-8"))  
    else:
        raise ValueError("Input must be either a file path or an uploaded file object.")


def parse_geno(input_data):
    geno_lines = []
    try:
        with handle_file_input(input_data) as file:
            geno_lines = file.readlines()
    except Exception as e:
        raise ValueError("Input must be either a file path or a file-like object.") from e

    geno_data = [list(map(int, line.strip())) for line in geno_lines]
    geno_array = np.array(geno_data, dtype=np.uint8).T
    return np.where(geno_array == 9., np.nan, geno_array) 

def parse_ind(file_input):
    try:
        with handle_file_input(file_input) as file:
            ind_df = pd.read_csv(file, delim_whitespace=True, header=None, names=['ID', 'Gender', 'Population'], engine='python')
    except Exception as e:
        raise ValueError("Input must be either a file path or a file-like object.") from e

    return ind_df


def create_subset(input_geno, input_ind, selected_indices, output_geno_path, output_ind_path):
    # Handle IND file
    ind_file = handle_file_input(input_ind)
    with ind_file, open(output_ind_path, 'w') as out_ind_file:
        for idx, line in enumerate(ind_file):
            if idx in selected_indices:
                out_ind_file.write(line)

    # Handle GENO file
    geno_file = handle_file_input(input_geno)
    with geno_file, open(output_geno_path, 'w') as out_geno_file:
        for line in geno_file:
            selected_columns = ''.join(line[i] for i in selected_indices)
            out_geno_file.write(selected_columns + "\n")

###########################################################################################################
def missing_statistics(geno, ind):
    # creates dict from number of missing genotype positions indexed by ID
    total_positions = geno.shape[1]
    missing_data_percentage = {
        ind.iloc[i].values[0]: (np.isnan(geno[i, :]).sum() / total_positions) * 100
        for i in range(geno.shape[0])
    }
    return missing_data_percentage, total_positions

def get_nonvariant_geno(geno, indices):
    print("get nonv feno")
    # filters genos only for variant positions
    # TODO: more general!
    indices = indices["x"].values - 1
    return geno[:, indices]

def plot_missing(nines):
    # creates bar plot of missing percentage per sample
    sample_names = list(nines.keys())
    missing_percentage = list(nines.values())
    #missing_percentage = [100 - i for i in coverage_percentage]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sample_names,
        y=missing_percentage,
        marker=dict(color='#ff9900'),
        hoverinfo="x+y",
    ))
    fig.add_hline(y=100,
        line_color="grey",
        line_dash="dash",
        line_width=2
    )
    fig.update_layout(
        #title="Missing Data per Sample",
        xaxis_title="Samples",
        yaxis_title="Missing Genotypes [%]",
        xaxis=dict(tickangle=45),
        template="simple_white",
        #height=400,
        #width=900,
        #showlegend=True,
        plot_bgcolor='white',  
        paper_bgcolor='white',
        font_size=14,
        font_color='black',

    )
    return fig

def compute_tau(geno, genomean, V, is_not_nani):
    snp_drift = np.sqrt((genomean / 2) * (1 - genomean / 2))
    geno_norm = (geno - genomean) / snp_drift

    V_obs = V[is_not_nani]
    proj_factor = np.linalg.inv(V_obs.T @ V_obs) @ V_obs.T
    tau = proj_factor @ geno_norm[is_not_nani]
    return tau

def pmp_drift_parallel(genos, V, genomean, is_not_nan, n_jobs=-1):
    taus = Parallel(n_jobs=n_jobs)(
        delayed(compute_tau)(geno, genomean, V, is_not_nan[i]) for i, geno in enumerate(genos)
    )
    return taus

def save_fig_as_pdf(fig):
    pdf_buffer = BytesIO()
    fig.write_image(pdf_buffer, format="pdf", engine="kaleido")
    pdf_buffer.seek(0) 
    return pdf_buffer

def color_plot(modern_df, taus, inds, Lambda):
    pc1_label = f"PC1 ({Lambda[0]:.2f}%)"
    pc2_label = f"PC2 ({Lambda[1]:.2f}%)"
    modern_df = modern_df.sort_values(by=['Group'])
    markers = ['circle', 'triangle-up', 'square', 'pentagon', 'star', 'diamond', 'diamond-wide', 'hexagon', 'x']
    palette = px.colors.qualitative.Vivid
    palette = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in palette]

    style = list(itertools.product(palette, markers))
    modern_df['Style'] = modern_df['Group'].map(dict(zip(modern_df['Group'].unique(),
                                       style)))
    fig = go.Figure()

    for sty in modern_df['Style'].unique():
        df_sub = modern_df.loc[modern_df['Style'] == sty]
        fig.add_trace(
            go.Scatter(
                x=df_sub['PC1'],
                y=df_sub['PC2'],
                mode='markers',
                marker=dict(
                    size=5,
                    symbol=sty[1],  # Marker-Stil (z. B. 'o', '^', etc.)
                    color=f"rgb({sty[0][0]*255},{sty[0][1]*255},{sty[0][2]*255})",  # Farbe in RGB
                    line=dict(width=0.1, color='white')  # Kantenfarbe
                ),
                name=df_sub['Group'].unique()[0],  # Gruppenname als Legende,
                meta=df_sub['Group'],
                hovertemplate='%{meta}', 
                legend="legend1"
            )
        )


    for t, tau in enumerate(taus): 
        name = inds.iloc[t]["ID"]
        sample_text = f"{name}"
        # Sample hinzufügen
        fig.add_trace(go.Scatter(
            x=[tau[0]],
            y=[tau[1]],
            mode='markers+text',
            marker=dict(color='black', size=8),
            name=name,
            text=sample_text,
            textposition="top center",
            legendgroup=name, 
            legend="legend2"
        ))

    fig.update_layout(
        width=1300,
        height=900,
        xaxis_title=pc1_label,
        yaxis_title=pc2_label,
        template="simple_white",
        font_size=14,
        font_color='black',
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
        ),
        legend1=dict(
            #title="Modern West Eurasians",
            itemsizing='constant',  
            traceorder="normal",  
            orientation="h",
            x=0, #x=1.05,
            y=0, #y=0.5, 
            xanchor="left",
            yanchor="bottom",
            font=dict(
                size=12,
                family="Arial" 
            ),
            itemwidth=80,
            title_font=dict(
                size=14
            ),
            bordercolor="white",  
            borderwidth=0.1
            ),
        legend2=dict(
            title="Ancient samples <br> &nbsp;",
            title_font=dict(
                size=16
            ),
            font=dict(
            size=14,
            color="black"
        ),
            itemsizing='constant',  
        ),

        legend_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        tickfont_size=14,
        tickcolor='black'
        #tickwidth=5
    )
    fig.update_yaxes(
        range=[-500, 250],
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        tickfont_size=14,
        #tickwidth=5,
        tickcolor='black'
    )

    return fig


def base_plot(modern, taus, inds, color_lookup, Lambda):
    fig = go.Figure()
    pc1_label = f"PC1 ({Lambda[0]:.2f}%)"
    pc2_label = f"PC2 ({Lambda[1]:.2f}%)"
    # Modern samples hinzufügen
    coords_mwe = modern[["PC1", "PC2"]]

    fig.add_trace(go.Scatter(
        x=coords_mwe["PC1"],
        y=coords_mwe["PC2"],
        mode='markers',
        marker=dict(color='rgba(220, 220, 220, 0.8)'),
        name='Modern Samples',
    ))

    for t, tau in enumerate(taus): 
        name = inds.iloc[t]["ID"]
        sample_text = f"{name}"
        # Sample hinzufügen
        fig.add_trace(go.Scatter(
            x=[tau[0]],
            y=[tau[1]],
            mode='markers+text',
            marker=dict(color=color_lookup[name], size=10),
            name=name,
            text=sample_text,
            textposition="top center",
            legendgroup=name,
        ))
    fig.update_layout(
        xaxis_title=pc1_label,
        yaxis_title=pc2_label,
        width=1300,
        height=750,
        template='simple_white',
        font_size=14,
        font_color='black',
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        tickfont_size=14,
        tickcolor='black'
        #tickwidth=5
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        tickfont_size=14,
        #tickwidth=5,
        tickcolor='black'
    )
    return fig

def get_ellipse(mean, Sigma, confidence_level):
    chi2_val = chi2.ppf(confidence_level, df=2)
    eigvals, eigvecs = np.linalg.eigh(Sigma)

    # Sorting eigenvalues and corresponding eigenvectors
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Width and height of the ellipse (2 * sqrt(eigenvalue * chi-square value))
    width = 2 * np.sqrt(eigvals[0] * chi2_val)
    height = 2 * np.sqrt(eigvals[1] * chi2_val)

    # Angle of the ellipse in degrees (in the direction of the largest eigenvector)
    angle_rad = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    x_center, y_center = mean
   
    t = np.linspace(0, 2 * np.pi, 100)
    x = (width / 2) * np.cos(t)
    y = (height / 2) * np.sin(t)

    # Rotation 
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)

    # Translation to center
    x_final = x_rot + x_center
    y_final = y_rot + y_center

    return x_final, y_final
    #return ellipse

def var_discrepency(V_obs, var_tau_r):
  
  #Computes the variance discrepancy between the estimated embedding and the true embedding.
  matrix_of_linear_map = - np.linalg.inv(V_obs[:, 0:2].T @ V_obs[:, 0:2]) @ V_obs[:, 0:2].T @ V_obs[:, 2:]
  return matrix_of_linear_map @ var_tau_r @ matrix_of_linear_map.T

def set_active_tab(tab_name):
    st.session_state["active_tab"] = tab_name

# Callback-Funktion zur Synchronisation
def update_percentiles():
    st.session_state["selected_percentiles"] = st.session_state["percentile_selector"]

# **Function to update button color using JavaScript**
def change_button_color(tabs, inactive_color, active_color):
    js_code = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            elements.forEach(btn => {{
                if ({' || '.join([f"btn.innerText.includes('{tab}')" for tab in tabs])}) {{
                    btn.style.background = "{inactive_color}";
                    btn.style.color = "black";
                    btn.style.borderRadius = "10px 10px 0px 0px"; /* Rounded top, flat bottom */
                    btn.style.borderBottom = "none"; /* Remove bottom border */
                    btn.style.width = "100%"; /* Make button full width */
                    btn.style.margin = "0px"; /* Remove extra margin */
                }}
                if (btn.innerText.includes("{st.session_state['active_tab']}")) {{
                    btn.style.background = "{active_color}";
                    btn.style.color = "white";
                    btn.style.borderRadius = "10px 10px 0px 0px"; /* Keep active tab design */
                    btn.style.borderBottom = "none";
                    btn.style.width = "100%"; /* Make button full width */
                    btn.style.margin = "0px"; /* Remove extra margin */
                }}
            }});
        </script>
    """
    components.html(js_code, height=0, width=0)


title =  """
    <h1 class="title">
        Welcome to TrustPCA!
    </h1>
    """

ellipse_logo_old = """
    <div class="ellipses">
        <svg width="500" height="50" xmlns="http://www.w3.org/2000/svg">
            <ellipse cx="250" cy="25" rx="100" ry="5" fill="red" opacity="0.7" />
            <ellipse cx="250" cy="25" rx="200" ry="8" fill="red" opacity="0.4" />
            <circle cx="250" cy="25" r="5" fill="blue" />
        </svg>
    </div>
    """

ellipse_logo = """
    <img alt="TrustPCA" src="./logo/trustpca_logo.svg" height="90">
    """

subtitle =  """
    <p class="description">
        A <span class="highlight">T</span>ool for <span class="highlight">R</span>eliability and <span class="highlight">U</span>ncertainty in <span class="highlight">S</span>mar<span class="highlight">T</span>PCA projections
    </p>
    """

buttons =  """
    <style>
    div.stButton>button {
    }
    div.stButton>button:hover {
    }
    </style>
    """

buttons= """
<style>
div.stButton>button {
}

div.stButton>button:hover {
}
</style>
"""

############################################################
# Standard files 

#path_to_database = "../../ancientPCA/database/"
path_to_database = "./database/"
modern = pd.read_csv(path_to_database+'coordinates_MWE.csv')
example_data_stats = path_to_database+'test_samples_stats.csv'
modern_df = pd.read_csv(path_to_database+'embedding_modern_refs.csv')
groups = pd.read_csv(path_to_database+'modern_groups_curated.csv', header=0)
modern_df['Group'] = groups['Group']
indices = pd.read_csv(path_to_database+'SNPs_mwe.csv', header=0)
genomean = pd.read_csv(path_to_database+'genomean.csv', header=0)
genomean = genomean['x'].values
V1_path = path_to_database+'eigenvectors1.npy'
V2_path = path_to_database+'eigenvectors2.npy'
Lambda = np.load(path_to_database+'eigenvalues.npy')
magical_factors = np.load(path_to_database+'factors.npy')

############################################################
# set globally 
explained_variance_ratio = Lambda / Lambda.sum() * 100
percentiles = [0.99, 0.9, 0.75, 0.5]
base_palette = px.colors.qualitative.Vivid
  

# App Layout
st.set_page_config(layout="wide",page_title="TrustPCA")

#smaller top margin 
col1, col2 = st.columns([1,10], vertical_alignment="center")
with col1:
    st.image("logo/trustpca_logo.svg", width=150)
with col2:
    st.header('Welcome to TrustPCA')
    st.subheader('A Tool for Reliability and Uncertainty in SmartPCA Projections')

st.divider()


st.markdown('#### About')
st.markdown("TrustPCA is a webtool that implements a probabilistic framework that predicts the uncertainty of SmartPCA projections due to missing genotype information and visualizes the uncertainties in a PC scatter plot.")
st.markdown('#### Tool specification')
st.info('In its current version, TrustPCA computes the PC space from modern West Eurasian populations. Therefore, the uncertainty predictions from TrustPCA are only meaningful for (ancient) human individuals from West Eurasia from the Mesolithic epoch or later.')
st.markdown('''
- **Input:** Genotype information of (ancient) human individuals based on the Human Origins array, covering approx. 600.000 sites.
- **Input format:** EIGENSTRAT format.
- **Output:** 
  - PC scatter plot (PC2 vs. PC1) based on the modern West Eurasian map. 
  - SmartPCA projections of the given (ancient) individuals together with uncertainty predictions of these projections. The uncertainties are visualized as confidence ellipses.
''', unsafe_allow_html=True)

st.divider()

# Initialize session state
if "geno_file" not in st.session_state:
    st.session_state["geno_file"] = None
if "ind_file" not in st.session_state:
    st.session_state["ind_file"] = None
if "ind_file_parsed" not in st.session_state:
    st.session_state["ind_file_parsed"] = None
if "geno" not in st.session_state:
    st.session_state["geno"] = None
if "ind" not in st.session_state:
    st.session_state["ind"] = None
if "nines" not in st.session_state:
    st.session_state["nines"] = None
if "taus" not in st.session_state:
    st.session_state["taus"] = None
if "ellipses" not in st.session_state:
    st.session_state["ellipses"] = None
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Input Data" 
if "selected_percentiles" not in st.session_state:
    st.session_state["selected_percentiles"] = percentiles
if "missing_plot" not in st.session_state:
    st.session_state["missing_plot"] = None
if "ellipse_plot" not in st.session_state:
    st.session_state["ellipse_plot"] = None
if "sample_submitted" not in st.session_state:
    st.session_state["sample_submitted"] = False
if "preprocessing" not in st.session_state:
    st.session_state["preprocessing"] = False
if "checkbox_states" not in st.session_state:
    st.session_state["checkbox_states"] = []
if "select_all" not in st.session_state:
    st.session_state["select_all"] = False
if "example_data" not in st.session_state:
    st.session_state["example_data"] = False
if "parsed" not in st.session_state:
    st.session_state["parsed"] = False
if "color_lookup" not in st.session_state:
    st.session_state["color_lookup"] = False
if "df_pca" not in st.session_state:
    st.session_state["df_pca"] = None

#get download csv stats for modern samples
modern_stats = modern_df[["Group", "PC1", "PC2"]].copy()
modern_stats.rename(columns={"Group": "Sample_ID"}, inplace=True)
modern_stats["Variance_PC1"] = np.nan
modern_stats["Variance_PC2"] = np.nan
modern_stats["Covariance_PC1_PC2"] = np.nan
modern_stats["Sample_Type"] = "Modern"

st.session_state["modern_df"] = modern_stats


tabs = ["Input Data", "Uncertainty Analysis", "More Infos on TrustPCA"]
inactive_color = "#cceaff"  
active_color = "#0070C0"  


columns = st.columns(len(tabs), gap="small")
for col, tab_name in zip(columns, tabs):
    with col:
        if st.button(tab_name, key=f"button_{tab_name}"):
            st.session_state["active_tab"] = tab_name
            #change_button_color(tabs, inactive_color, active_color)
            #st.rerun() 

with st.container(height=0):
    change_button_color(tabs, inactive_color, active_color)

if st.session_state["active_tab"] == "Input Data":
    # file uploads
    with st.container(border=True):
        st.markdown("**Data Upload**")
        with st.expander("Upload your data", icon=":material/upload:"):
            st.markdown("Upload **GENO** and **IND** files as defined by the EIGENSTRAT format.")

            col1, col2 = st.columns(2)

            with col1:
                geno_file = st.file_uploader("Upload GENO file", type=["csv", "txt", "geno"])
                if geno_file:
                    st.session_state["geno_file"] = geno_file

            with col2:
                ind_file = st.file_uploader("Upload IND file", type=["csv", "txt", "ind"])
                if ind_file:
                    st.session_state["ind_file"] = ind_file
            
            if st.session_state["geno_file"] and st.session_state["ind_file"]:
                if st.session_state["ind_file_parsed"] is None:
                    ind_data = parse_ind(st.session_state["ind_file"])
                    st.session_state["ind_file_parsed"] = ind_data
                if len(st.session_state["checkbox_states"]) != len(st.session_state["ind_file_parsed"]):
                    st.session_state["checkbox_states"] = [False] * len(st.session_state["ind_file_parsed"])
                st.subheader("Select Individuals for Analysis")
                
                st.session_state["select_all"] = st.checkbox("Select All Individuals", value=st.session_state["select_all"])
                if st.session_state["select_all"]:
                    st.session_state["checkbox_states"] = [True] * len(st.session_state["ind_file_parsed"])

                num_columns = 3
                columns = st.columns(num_columns)

                for i, row in st.session_state["ind_file_parsed"].iterrows():
                    col_index = i % num_columns
                    with columns[col_index]:
                        entry = f"{row['ID']} - {row['Gender']} - {row['Population']}"
                        st.session_state["checkbox_states"][i] = st.checkbox(
                            entry, value=st.session_state["checkbox_states"][i], key=f"checkbox_{i}"
                        )
                # st.info("This option is disabled in the preliminary version of the TrustPCA web tool due to Streamlit's/Github's file size restrictions. Future releases will be hosted on our own servers, but this is currently not possible due to the double-blind peer review process.")
                if st.button("Submit Selection", disabled=False): #, disabled=True):
                    if st.session_state["parsed"]:
                        print("parsed")
                        st.session_state["parsed"]=False
                        st.session_state["preprocessing"]=False
                    selected_indices = [i for i, selected in enumerate(st.session_state["checkbox_states"]) if selected]
                    if selected_indices:
                        st.session_state["sample_submitted"] = True
                        if not st.session_state.select_all:
                            with tempfile.TemporaryDirectory(dir=".") as temp_dir:
                                output_geno_path = f"{temp_dir}/subset_geno.geno"
                                output_ind_path = f"{temp_dir}/subset_ind.ind"
                                create_subset(st.session_state["geno_file"], st.session_state["ind_file"], selected_indices, output_geno_path, output_ind_path)
                                st.session_state["geno"] = parse_geno(output_geno_path)
                                st.session_state["ind"] = parse_ind(output_ind_path)
                        else:
                            st.session_state["geno"] = parse_geno(geno_file)
                            st.session_state["ind"] = st.session_state["ind_file_parsed"]
                        st.session_state["parsed"]=True
                    status_banner = st.empty()
                    
            if st.session_state["parsed"]:
                status_banner = st.empty()
                if not st.session_state["preprocessing"]:
                    status_banner.info("Filtering genos...")
                    nonv_geno = get_nonvariant_geno(st.session_state["geno"], indices)
                    st.session_state["geno"] = nonv_geno
                    status_banner.info("Calculating missing statistics...")
                    nines, total_positions = missing_statistics(st.session_state["geno"], st.session_state["ind"])
                    st.session_state["nines"] = nines
                    st.session_state["missing_plot"] = plot_missing(st.session_state["nines"])
                    is_not_nan = ~np.isnan(nonv_geno)
                    status_banner.info("Projecting samples...")
                    
                    V1 = np.load(V1_path)
                    V2 = np.load(V2_path)
                    V = np.vstack([V1, V2])
                    taus = pmp_drift_parallel(nonv_geno, V[:, 0:2], genomean, is_not_nan, n_jobs=-1) #pmp_drift(nonv_geno, V, genomean, is_not_nan)
                    st.session_state["taus"] = taus

                    status_banner.info("Generating PCA plot...")
                    num_inds=st.session_state["ind"].shape[0]
                    scaled_palette = [base_palette[i % len(base_palette)] for i in range(num_inds)]
                    color_lookup = dict(zip(st.session_state["ind"]["ID"], scaled_palette))
                    st.session_state["color_lookup"] = color_lookup
                    fig = base_plot(modern, taus, st.session_state["ind"], color_lookup, explained_variance_ratio[0:2])
                    st.session_state["base_plot"] = fig
                    color_plot = color_plot(modern_df, st.session_state["taus"], st.session_state["ind"], explained_variance_ratio[0:2])
                    
                    st.session_state["color_plot"] = color_plot

                    ellipses = {} #ellipses can not be a df bc arrow compatibility or sth
                    ellipse_stats = []  
                    status_banner.info("Calculating uncertainties...") 

                    progress_bar = st.progress(0)  # Start bei 0%
                    total_samples = len(nonv_geno)
                    var_tau_r = np.diag(Lambda[2:] * magical_factors[2:])
                    for index, geno in enumerate(nonv_geno):
                        curr_ind = st.session_state["ind"].iloc[index]
                        progress = (index + 1) / total_samples
                        progress_bar.progress(progress)

                        ellipses[curr_ind["ID"]] = {}
                    
                        is_not_nani = is_not_nan[index]
                        # Predict variance of discrepency
                        V_obs = V[is_not_nani]
                        var_discr = var_discrepency(V_obs=V_obs, var_tau_r=var_tau_r)

                         #for df
                        variance_pc1 = var_discr[0, 0]
                        variance_pc2 = var_discr[1, 1]
                        covariance_pc1_pc2 = var_discr[0, 1]

                        # Speichere diese Werte für den DataFrame
                        ellipse_stats.append({
                            "Sample_ID": curr_ind["ID"],
                            "PC1": taus[index][0],
                            "PC2": taus[index][1],
                            "Variance_PC1": variance_pc1,
                            "Variance_PC2": variance_pc2,
                            "Covariance_PC1_PC2": covariance_pc1_pc2,
                            "Sample_Type": "Ancient"  
                        })

                        st.session_state["df_pca"] = pd.concat([pd.DataFrame(ellipse_stats), st.session_state["modern_df"]], ignore_index=True)


                        #ellipses
                        for percentile in percentiles:
                            x, y = get_ellipse(mean=taus[index], Sigma=var_discr, confidence_level=percentile)
                            ellipses[curr_ind["ID"]][percentile] = {"x": x, "y": y}

                    progress_bar.empty()
                    status_banner.empty()
                    st.session_state["ellipses"] = ellipses
                    status_banner.empty()
                    st.session_state["preprocessing"] = True
                    V = None
                    st.rerun()

         
        if st.button("Use Example Data"):
            if st.session_state["preprocessing"]:
                st.session_state["preprocessing"] = False
            st.session_state["example_data"] = True
            st.session_state["parsed"] = True
            V = None

            
    if st.session_state["preprocessing"]:
        #st.success("GENO and IND data uploaded successfully!")
        with st.expander("Show data characteristics", icon=":material/table_eye:"):
            if not st.session_state["example_data"]:
                st.subheader("Data preview")
                st.write("Geno (First 50x50)")
                st.write(st.session_state["geno"][0:50, 0:50])
                st.write("Ind (First Rows)")
                st.write(st.session_state["ind"].head())
            st.markdown('The following statistics are measured relative to the total of 540,247 genotypes included in the PCA analysis.')
            col1, col2 = st.columns(
            [0.3, 0.7],
            vertical_alignment="center",
            gap="medium"
            )   
            with col1:
                nines_df = pd.DataFrame.from_dict(st.session_state["nines"], orient="index", columns=["Missing Genotypes [%]"])
                nines_df["Missing Genotypes [%]"] = nines_df["Missing Genotypes [%]"].round(2).apply(lambda x: f"{x:.2f}")
                st.dataframe(nines_df,
                    use_container_width=True,

                )
            with col2:
                if st.session_state["missing_plot"]:
                    st.plotly_chart(st.session_state["missing_plot"], theme=None)
        with st.container(border=True):
            st.subheader("SmartPCA Plot")
            st.markdown('This is an interactive plot! Use the legend to (de)select samples or zoom in for a closer look.')
            
            if "current_plot" not in st.session_state:
                st.session_state["current_plot"] = "base"

            if st.button("Change Color Scheme"):
                if st.session_state["current_plot"] == "base":
                    st.session_state["current_plot"] = "color"
                else:
                    st.session_state["current_plot"] = "base"

          
            if st.session_state["current_plot"] == "base":
                st.plotly_chart(st.session_state["base_plot"], theme=None)
                pdf_buffer = save_fig_as_pdf(st.session_state["base_plot"])
                st.download_button(
                    label="Download Figure as PDF",
                    data=pdf_buffer,
                    file_name="TRUST_PCA_download_SmartPCA_projection.pdf",
                    mime="application/pdf"
                )
            elif st.session_state["current_plot"] == "color":
                st.plotly_chart(st.session_state["color_plot"], theme=None)
                pdf_buffer = save_fig_as_pdf(st.session_state["color_plot"])
                st.download_button(
                    label="Download Figure as PDF",
                    data=pdf_buffer,
                    file_name="TRUST_PCA_download_SmartPCA_projection.pdf",
                    mime="application/pdf"
                )

            csv_buffer = io.BytesIO()
            st.session_state["df_pca"].to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)  

            st.download_button(
                label="Download PCA Data as CSV",
                data=csv_buffer,
                file_name="TRUST_PCA_data.csv",
                mime="text/csv")
            
    else:
        if st.session_state["example_data"]:
            ancient_example = pd.read_csv(path_to_database+"example_data_tool.csv", header=0)
            with open(path_to_database+"ellipses_example_data.pkl", "rb") as f:
                ellipses = pickle.load(f)
                st.session_state["ellipses"] = ellipses
            nines = ancient_example[["ID","coverage"]]
            nines["coverage"]=(100 - nines["coverage"]/540247 * 100)
            nines_dict = nines.set_index("ID")["coverage"].to_dict()
            st.session_state["nines"]=nines_dict
            st.session_state["missing_plot"] = plot_missing(st.session_state["nines"])
            st.session_state["ind"] = ancient_example[["ID", "Group_ID"]]
            st.session_state["taus"] = np.array(ancient_example[["x", "y"]])
            
            num_inds = len(ancient_example["ID"])
            scaled_palette = [base_palette[i % len(base_palette)] for i in range(num_inds)]
            color_lookup = dict(zip(ancient_example["ID"], scaled_palette))
            st.session_state["color_lookup"] = color_lookup
            fig = base_plot(modern, st.session_state["taus"], st.session_state["ind"], color_lookup, explained_variance_ratio[0:2])
            color_plot = color_plot(modern_df, st.session_state["taus"], st.session_state["ind"], explained_variance_ratio[0:2])
            st.session_state["base_plot"] = fig
            st.session_state["color_plot"] = color_plot
            st.session_state["df_pca"] = pd.read_csv(example_data_stats)
            st.session_state["preprocessing"] = True
            st.rerun()

# Tab 2
elif st.session_state["active_tab"] == "Uncertainty Analysis":
    if st.session_state["preprocessing"]:
        with st.container(border=True):
            st.subheader("SmartPCA Plot with Uncertainties")
            st.markdown('This is an interactive plot! Use the legend to (de)select samples or zoom in for a closer look.')

            col1, col2 = st.columns([0.4, 0.6])
            with col1:
                # change design muslitselect
                selected = st.multiselect(
                label="Select Percentiles",
                options=[0.99, 0.9, 0.75, 0.5],
                default=st.session_state["selected_percentiles"],
                key="percentile_selector",
                on_change=update_percentiles
                )      
            if st.button("Change Color Scheme"):
                st.session_state["current_plot"] = "base" if st.session_state["current_plot"] == "color" else "color"

            if st.session_state["current_plot"] == "base":
                fig = copy.deepcopy(st.session_state["base_plot"])
            else:
                fig = copy.deepcopy(st.session_state["color_plot"])


            alpha = 0.2
            for index, ind_i in st.session_state["ind"].iterrows():
                color_rgb = st.session_state["color_lookup"][ind_i["ID"]][4:-1].split(",")  # Entferne 'rgb(' und ')'
                fillcolor = f'rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, {alpha})'
                for percentile in st.session_state["selected_percentiles"]:
                    coords = st.session_state["ellipses"][ind_i["ID"]][percentile]
                    fig.add_trace(go.Scatter(
                        x=np.append(coords["x"], coords["x"][0]),  # Ellipse schließen
                        y=np.append(coords["y"], coords["y"][0]),
                        mode="none",
                        fill="toself",
                        fillcolor=fillcolor if st.session_state["current_plot"] == "base" else f'rgba(0, 0, 0, {alpha})',
                        legendgroup=ind_i["ID"], 
                        showlegend=False,
                        name=f"{percentile}"
                    ))

            st.plotly_chart(fig, theme=None)
            st.session_state["ellipse_plot"] = fig

            pdf_buffer = save_fig_as_pdf(st.session_state["ellipse_plot"])
            st.download_button(
                label="Download Figure as PDF",
                data=pdf_buffer,
                file_name="TRUST_PCA_download.pdf",
                mime="application/pdf"
            )

            csv_buffer = io.BytesIO()
            st.session_state["df_pca"].to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)  
            
            st.download_button(
                label="Download PCA Data as CSV",
                data=csv_buffer,
                file_name="TRUST_PCA_data.csv",
                mime="text/csv")
    else:
        st.markdown("No data available. Please upload your file or choose example data from the Input Data tab.")

# Tab 3
elif st.session_state["active_tab"] == "More Infos on TrustPCA":
    st.markdown("TrustPCA is an advanced tool for **Principal Component Analysis (PCA) of ancient human genomic data**, providing an **estimation of projection uncertainty** of genotype individuals.")
    st.markdown("Using a **modern West Eurasian reference PC space**, TrustPCA enables the projection of ancient individuals, similar to SmartPCA. However, TrustPCA goes further by estimating **projection uncertainties** based on missing loci in the provided samples. These uncertainties are visualized as **ellipses on PCA plots**, representing different confidence levels for sample placement.")
    st.markdown("All outputs, including **statistical summaries and visualizations**, can be exported as PDFs for seamless integration into research workflows.")
    
    st.markdown("""
    ### Supported Data Formats
    TRUST PCA works with **EIGENSTRAT-formatted files**, a widely used format for genomic data analysis.
    The following files are required:
    - **GENO file** (`*.geno`): Genotype matrix with SNP data in a compact format.
    - **IND file** (`*.ind`): Information about individuals (ID, population, gender).
    """)
    st.markdown("""
    More details can be found in the official documentation:
    - [EIGENSTRAT Format](https://reich.hms.harvard.edu/software/InputFileFormats)
    - [SmartPCA Documentation](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/chrchang/eigensoft/blob/master/POPGEN)
    ---
    ### How It Works
    1) **Upload** your EIGENSTRAT files or use the provided example dataset.  
    2) **Select** individuals to be included in the analysis and **start the computation**.  
        - Ancient individuals are projected similar to SmartPCA onto the reference PC space (**~1 sec per sample**).  
        - Projection uncertainties are calculated for all individuals (**~1.4 sec per sample**). 
    3) **Explore** the results (be aware of the different tabs!):
        - Inspect the SmartPCA projections of the ancient individuals.
        - Inspect the projection uncertainties visualized as confidence ellipses.
    5) **Download figures** as a .pdf and **download data** as .csv.
    ---
    ### Citation & References
    If you use TRUST PCA in your research, please cite:
    **Susanne Zabel1, Samira Breitling, Cosimo Posth, and Kay Nieselt**, "A Probabilistic Approach to Visualize the Effect of Missing Data on PCA in Ancient Human Genomics", *Published in BMC Genomics*, 2025.
    DOI: [10.1186/s12864-025-11728-1](https://doi.org/10.1186/s12864-025-11728-1)
    ### About This Tool
    - **Developed by:** Integrative Transcriptomics Research Group
    """,
        unsafe_allow_html=True)


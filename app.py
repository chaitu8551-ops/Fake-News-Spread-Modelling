import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
 
# States: 0=S, 1=I(Believers), 2=R, 3=Skeptics, 4=FC
STATES = {0:'#3b82f6',1:'#ef4444',2:'#22c55e',3:'#a855f7',4:'#f59e0b'}
LABELS = {0:'Susceptible',1:'Believers',2:'Recovered',3:'Skeptics',4:'Fact-Checkers'}
 
def load_explanation():
    try:
        with open("explanation.txt","r",encoding="utf-8") as f:
            return f.read()
    except:
        return "⚠️ explanation.txt not found"
 
@st.cache_data
def build_network(N,m,seed=42):
    return nx.barabasi_albert_graph(N,m,seed=seed)
 
def assign_roles(G,skeptic_frac,fc_frac,init_frac,seed=42):
    rng=np.random.default_rng(seed)
    N=G.number_of_nodes()
    nodes=list(G.nodes)
    rng.shuffle(nodes)
    n_sk=int(N*skeptic_frac)
    n_fc=int(N*fc_frac)
    ni=max(1,int(N*init_frac))
    states={n:0 for n in G.nodes}
    for n in nodes[:n_sk]: states[n]=3
    for n in nodes[n_sk:n_sk+n_fc]: states[n]=4
    susc=[n for n in G.nodes if states[n]==0]
    rng.shuffle(susc)
    for n in susc[:ni]: states[n]=1
    return states
 
def sir_step(G,states,beta,gamma,fc_mult,sk_mult):
    new_states=states.copy()
    for node in list(G.nodes):
        s=states[node]
        if s==1:
            for nb in G.neighbors(node):
                nbs=states[nb]
                if nbs in [0,3]:
                    eff_beta=beta*sk_mult if nbs==3 else beta
                    if np.random.random()<eff_beta:
                        new_states[nb]=1
            g=gamma*fc_mult if any(states[nbb]==4 for nbb in G.neighbors(node)) else gamma
            if np.random.random()<g:
                new_states[node]=2
    return new_states
 
def run_simulation(G,states_init,beta,gamma,T,fc_mult,sk_mult,hub_pct=0.0,seed=42):
    np.random.seed(seed)
    Gw=G.copy()
    states=states_init.copy()
    if hub_pct>0.0:
        degs=dict(Gw.degree)
        thr=np.percentile(list(degs.values()),100-hub_pct*100)
        hubs=[n for n,d in degs.items() if d>=thr]
        Gw.remove_nodes_from(hubs)
        for h in hubs: states.pop(h,None)
    history=[]
    N=Gw.number_of_nodes()
    if N==0: return pd.DataFrame(),{}
    for t in range(T):
        c={k:0 for k in range(5)}
        for s in states.values(): c[s]+=1
        history.append({'t':t,'S':c[0]/N*100,'I':c[1]/N*100,
                        'R':c[2]/N*100,'SK':c[3]/N*100,'FC':c[4]/N*100})
        states=sir_step(Gw,states,beta,gamma,fc_mult,sk_mult)
    df=pd.DataFrame(history)
    avg=2*Gw.number_of_edges()/N if N>0 else 0
    r0=beta*avg/gamma if gamma>0 else float('inf')
    return df,{'peak':df['I'].max(),'final_r':df['R'].iloc[-1],'r0':r0,'nodes':N}
 
def make_time_series_fig(df_base,df_int=None):
    fig=go.Figure()
    cols=['S','I','R','SK','FC']
    ids=[0,1,2,3,4]
    for i,col in enumerate(cols):
        fig.add_trace(go.Scatter(x=df_base['t'],y=df_base[col],name=LABELS[ids[i]],
                                 line=dict(color=STATES[ids[i]])))
        if df_int is not None:
            fig.add_trace(go.Scatter(x=df_int['t'],y=df_int[col],
                                     name=f"{LABELS[ids[i]]} (Int)",
                                     line=dict(color=STATES[ids[i]],dash='dash')))
    fig.update_layout(title="SIR Dynamics",xaxis_title="Time",yaxis_title="%",height=400)
    return fig
 
def make_network_fig(G,states):
    pos=nx.spring_layout(G,seed=42)
    edge_x=[]; edge_y=[]
    for edge in G.edges():
        x0,y0=pos[edge[0]]
        x1,y1=pos[edge[1]]
        edge_x.extend([x0,x1,None])
        edge_y.extend([y0,y1,None])
    edge_trace=go.Scatter(x=edge_x,y=edge_y,line=dict(width=0.5,color='#888'),
                          mode='lines',hoverinfo='none')
    node_trace=[]
    for state in range(5):
        x_n,y_n=[],[]
        for node in G.nodes():
            if states.get(node,0)==state:
                x_n.append(pos[node][0])
                y_n.append(pos[node][1])
        if x_n:
            node_trace.append(go.Scatter(x=x_n,y=y_n,mode='markers',
                                         marker=dict(size=8,color=STATES[state]),
                                         name=LABELS[state]))
    fig=go.Figure(data=[edge_trace]+node_trace)
    fig.update_layout(title="Network",showlegend=True,
                      xaxis_showgrid=False,yaxis_showgrid=False,height=450)
    return fig
 
st.title("Infodemic Watch")
st.write("SIR misinformation propagation model on BA networks")
 
with st.sidebar:
    col1,col2=st.columns(2)
    with col1:
        N=st.slider("N",100,400,200)
        m=st.slider("m",2,6,3)
    with col2:
        beta=st.slider("beta",0.01,0.15,0.08)
        gamma=st.slider("gamma",0.01,0.08,0.03)
    T=st.slider("Steps",50,120,80)
    init_pct=st.slider("Init I %",0.5,3.0,1.0)
    skeptic_pct=st.slider("Skeptics %",0.0,0.15,0.08)
    fc_pct=st.slider("FC %",0.0,0.1,0.04)
    st.subheader("Interventions")
    fc_boost=st.checkbox("FC Boost")
    fc_mult=st.slider("FC x",2.0,4.0,2.5)
    hubs=st.checkbox("Remove Hubs")
    hub_pct=st.slider("Hubs %",5,15,10)
 
if st.button("Simulate"):
    with st.spinner("Running..."):
        G=build_network(N,m)
        states=assign_roles(G,skeptic_pct,fc_pct,init_pct/100)
        df_b,m_b=run_simulation(G,states,beta,gamma,T,1.0,0.3)
        fc_g=fc_mult if fc_boost else 1.0
        h_p=hub_pct/100 if hubs else 0
        df_i,m_i=run_simulation(G,states,beta,gamma,T,fc_g,0.3,h_p)
        st.session_state.update(df_b=df_b,df_i=df_i,m_b=m_b,m_i=m_i,G=G,states=states)
        st.success("Done!")
 
if 'df_b' in st.session_state:
    tab1,tab2,tab3,tab4=st.tabs(["📈 Time Series","📊 Results","🕸️ Network","📘 Explanation"])
    with tab1:
        fig=make_time_series_fig(st.session_state['df_b'],st.session_state['df_i'])
        st.plotly_chart(fig,use_container_width=True)
    with tab2:
        df_res=pd.DataFrame({
            'Metric':['Peak I%','Final R%','R0'],
            'Baseline':[f"{st.session_state['m_b']['peak']:.1f}",
                        f"{st.session_state['m_b']['final_r']:.1f}",
                        f"{st.session_state['m_b']['r0']:.1f}"],
            'Intervention':[f"{st.session_state['m_i']['peak']:.1f}",
                            f"{st.session_state['m_i']['final_r']:.1f}",
                            f"{st.session_state['m_i']['r0']:.1f}"]
        })
        st.dataframe(df_res)
        r0_val=st.session_state['m_b']['r0']
        if r0_val>1:
            st.error(f"R0 = {r0_val:.2f} > 1 — epidemic likely to spread")
        else:
            st.success(f"R0 = {r0_val:.2f} <= 1 — epidemic will die out")
    with tab3:
        fig_net=make_network_fig(st.session_state['G'],st.session_state['states'])
        st.plotly_chart(fig_net,use_container_width=True)
    with tab4:
        st.markdown(load_explanation())
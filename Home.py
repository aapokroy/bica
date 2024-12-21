import copy

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from lib.clients.database import SessionLocal, engine, crud
from lib.clients.database.models import Base
from lib.community import Community
from lib.config import cfg


@st.cache_resource
def init_db():
    Base.metadata.create_all(bind=engine)


# Page layout
st.set_page_config(
    page_title='BICA DLE',
    page_icon=cfg.streamlit.page_icon,
    layout='wide'
)

init_db()

plt.style.use('seaborn-v0_8-darkgrid')

st.write('# BICA DLE')

cols = st.columns(2)
with cols[0]:
    n_agents = st.number_input(
        label='Number of agents',
        min_value=1, max_value=50, step=1,
        value=30,
        help='''
        Number of agents in the community
        ''',
    )
with cols[1]:
    n_turns = st.number_input(
        label='Number of turns',
        min_value=1, max_value=None, step=10,
        value=300,
        help='''
        Number of turns in the simulation
        ''',
    )

with st.expander('Advanced parameters', expanded=False):
    cols = st.columns(2)
    with cols[0]:
        arousal_bias = st.number_input(
            label='Arousal bias',
            min_value=0.0, max_value=10.0, step=0.1,
            value=1.0,
            help='''
            How much arousal impacts action selection. Actor is more likely to choose actions towards
            recipient to whom they have higher arousal.
            ''',
        )
        temperature = st.number_input(
            label='Temperature',
            min_value=0.01, max_value=1.0, step=0.1,
            value=0.10,
            help='''
            Randomness of action selection. Higher temperature means more randomnes.
            ''',
        )
        appraisal_decay = st.number_input(
            label='Appraisal decay',
            min_value=0.0, max_value=0.1, step=0.001,
            value=0.02,
            help='''
            How fast appraisal moves to neutral. Higher value means faster decay.
            ''',
        )
    with cols[1]:
        appraisal_change_rate = st.number_input(
            label='Appraisal change rate',
            min_value=0.0, max_value=10.0, step=0.05,
            value=0.3,
            help='''
            How fast appraisal changes with each action. Higher rate means faster change.
            ''',
        )
        feelings_inertia = st.number_input(
            label='Feelings inertia',
            min_value=0.0, max_value=1.0, step=0.05,
            value=0.95,
            help='''
            How fast feelings change towards apraisal each turn. Higher inertia means slower change.
            ''',
        )
        max_actions_per_turn = st.number_input(
            label='Max actions per turn',
            min_value=1, max_value=10, step=1,
            value=3,
            help='''
            Maximum number of actions each agent can be a part of in each turn.
            ''',
        )

if st.button('Start simulation', use_container_width=True, type='primary'):
    with SessionLocal() as db:
        actions = crud.action.get_active(db)
        relationships = crud.relationship.get_active(db)

    community = Community(
        n_agents=n_agents,
        arousal_bias=arousal_bias,
        temperature=temperature,
        appraisal_change_rate=appraisal_change_rate,
        feelings_inertia=feelings_inertia,
        appraisal_decay=appraisal_decay,
        max_actions_per_turn=max_actions_per_turn,
        actions=actions,
        relationships=relationships,
    )

    plot = None
    with st.container(border=True):
        for fig in community.run(n_turns):
            if plot is None:
                plot = st.pyplot(fig, clear_figure=True)
            else:
                plt.close()
                plot.pyplot(fig, clear_figure=True)

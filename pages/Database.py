import streamlit as st
from sqlalchemy.orm import Session

from lib.clients.database import SessionLocal
from lib.clients.database import crud
from lib.config import cfg


def intensionality_input(
    label: str,
    key: str,
    values: list[float] | None = None,
    is_delta: bool = False,
) -> list[float]:
    if values is None:
        values = [0.0, 0.0, 0.0]
    res = []
    with st.container(border=True):
        st.write(label)
        for col, intensionality, value in zip(
            st.columns(3),
            ['Valence', 'Arousal', 'Dominance'],
            values,
        ):
            with col:
                min_value = -1.0
                if not is_delta and intensionality == 'Arousal':
                    min_value = 0.0
                res.append(
                    st.number_input(
                        label=intensionality,
                        min_value=min_value, max_value=1.0, step=0.1,
                        value=value,
                        key=f'{key}_{intensionality}',
                    )
                )
    return res


def handle_actions(db: Session):
    st.header('Manage Actions')

    actions = crud.action.get_all(db)
    actions.sort(key=lambda x: x.id, reverse=True)

    if st.button(
        label='Add',
        key='action_add',
        use_container_width=True,
        icon=':material/add:',
    ):
        try:
            action = crud.action.create(
                db=db,
                title=f'Action {len(actions)}',
                description='',
                actor_appraisal_delta=[0.0, 0.0, 0.0],
                recipient_appraisal_delta=[0.0, 0.0, 0.0],
                actor_relationships=[],
                recipient_relationships=[],
            )
            st.success(f'Action "{action.title}" created successfully!')
            st.rerun()
        except Exception as e:
            st.error(f'Error creating action: {e}')

    if actions:
        for action in actions:
            with st.expander(action.title, expanded=False):
                with st.form(f'edit_action_form_{action.id}', border=False):
                    title = st.text_input('Title', value=action.title)
                    description = st.text_area('Description', value=action.description)
                    actor_appraisal_delta = intensionality_input(
                        label='Change of actor\'s appraisal',
                        key=f'actor_appraisal_delta_edit_{action.id}',
                        values=action.actor_appraisal_delta,
                        is_delta=True,
                    )
                    recipient_appraisal_delta = intensionality_input(
                        label='Change of recipient\'s appraisal',
                        key=f'recipient_appraisal_delta_edit_{action.id}',
                        values=action.recipient_appraisal_delta,
                        is_delta=True,
                    )
                    actor_relationships = st.multiselect(
                        'Allowed actor\'s relationships to recipient',
                        options=crud.relationship.get_all(db),
                        default=action.actor_relationships,
                        format_func=lambda x: x.title
                    )
                    recipient_relationships = st.multiselect(
                        'Allowed recipient\'s relationships to actor',
                        options=crud.relationship.get_all(db),
                        default=action.recipient_relationships,
                        format_func=lambda x: x.title
                    )
                    is_active = st.checkbox('Is Active', value=action.is_active)

                    if st.form_submit_button(
                        label='Update Relationship',
                        use_container_width=True,
                    ):
                        try:
                            crud.action.update(
                                db=db,
                                id=action.id,
                                title=title,
                                description=description,
                                actor_appraisal_delta=actor_appraisal_delta,
                                recipient_appraisal_delta=recipient_appraisal_delta,
                                actor_relationships=[rel.id for rel in actor_relationships],
                                recipient_relationships=[rel.id for rel in recipient_relationships],
                                is_active=is_active,
                            )
                            st.success('Action updated successfully!')
                            st.rerun()
                        except Exception as e:
                            st.error(f'Error updating action: {e}')
                if st.button(
                    label='Delete',
                    key=f'delete_action_{action.id}',
                    use_container_width=True,
                    type='primary',
                ):
                    crud.action.delete(db, action.id)
                    st.success(f'Action "{action.title}" deleted successfully!')
                    st.rerun()
    else:
        st.info('No relationships available.')


def handle_relationships(db: Session):
    st.header('Manage Relationships')

    relationships = crud.relationship.get_all(db)
    relationships.sort(key=lambda x: x.id, reverse=True)

    if st.button(
        label='Add',
        key='relationship_add',
        use_container_width=True,
        icon=':material/add:',
    ):
        try:
            relationship = crud.relationship.create(
                db=db,
                title=f'Relationship {len(relationships)}',
                description='',
                feeling_centroid=[0.0, 0.0, 0.0],
            )
            st.success(f'Relationship "{relationship.title}" created successfully!')
            st.rerun()
        except Exception as e:
            st.error(f'Error creating action: {e}')

    if relationships:
        for relationship in relationships:
            with st.expander(relationship.title, expanded=False):
                with st.form(f'edit_relationship_form_{relationship.id}', border=False):
                    title = st.text_input('Title', value=relationship.title)
                    description = st.text_area('Description', value=relationship.description)
                    feeling_centroid = intensionality_input(
                        label='Feeling centroid',
                        key=f'feeling_centroid_edit_{relationship.id}',
                        values=relationship.feeling_centroid,
                    )
                    is_active = st.checkbox('Is Active', value=relationship.is_active)
                    if st.form_submit_button(
                        label='Update Relationship',
                        use_container_width=True,
                    ):
                        try:
                            crud.relationship.update(
                                db=db,
                                id=relationship.id,
                                title=title,
                                description=description,
                                feeling_centroid=feeling_centroid,
                                is_active=is_active,
                            )
                            st.success('Relationship updated successfully!')
                            st.rerun()
                        except Exception as e:
                            st.error(f'Error updating relationship: {e}')
                if st.button(
                    label='Delete',
                    key=f'delete_relationship_{relationship.id}',
                    use_container_width=True,
                    type='primary',
                ):
                    crud.relationship.delete(db, relationship.id)
                    st.success(f'Relationship "{relationship.title}" deleted successfully!')
                    st.rerun()
    else:
        st.info('No relationships available.')


@st.cache_resource
def get_db():
    db = SessionLocal()
    return db


# Page layout
st.set_page_config(page_title='Database', page_icon=cfg.streamlit.page_icon)

db = get_db()

st.markdown('# Database')

tabs = st.tabs(['Relationships', 'Actions'])

with tabs[0]:
    handle_relationships(db)

with tabs[1]:
    handle_actions(db)
